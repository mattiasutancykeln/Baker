import gc
import json
import os
import socket
import struct
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem  # noqa: F401  - ensure rdkit presence
from sklearn.linear_model import LinearRegression  # noqa: F401 - ensure sklearn presence

from snapshot_utils import snapshot_and_compact


# ----------- Config from env -----------
PROJECT_DIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace")).resolve()
DATA_DIR = PROJECT_DIR / "data"
SOCK_DIR = PROJECT_DIR / ".sandbox"
SOCK_PATH = SOCK_DIR / "kernel.sock"
LOG_PATH = SOCK_DIR / "daemon.log"

IDLE_HIBERNATE_SEC = int(os.environ.get("IDLE_HIBERNATE_SEC", "900"))  # 15 min
PRINT_DEBUG = bool(int(os.environ.get("DEBUG", "0")))

# Ensure dirs exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SOCK_DIR.mkdir(parents=True, exist_ok=True)


# Helpers: atomic parquet IO
def _atomic_write_parquet(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(table, tmp)
    tmp.replace(path)


def _save_df_name_first(name: str, df: pd.DataFrame) -> None:
    path = DATA_DIR / f"{name}.parquet"
    _atomic_write_parquet(pa.Table.from_pandas(df, preserve_index=True), path)


def save_df(df: pd.DataFrame, name: str) -> None:
    # User-facing: save(df, name)
    _save_df_name_first(name, df)


def load_df(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    return pq.read_table(path).to_pandas()


def save_np(name: str, arr: np.ndarray) -> None:
    # Store arrays as Arrow binary table; simple & portable
    t = pa.table({"name": [name], "shape": [str(list(arr.shape))], "data": [arr.tobytes()]})
    _atomic_write_parquet(t, DATA_DIR / f"{name}__np.parquet")


# Expose helper functions to user code
SANDBOX_GLOBALS = {
    "pd": pd,
    "np": np,
    "pa": pa,
    "pq": pq,
    "load": load_df,  # load('train_X')
    "save": save_df,  # save(df, 'train_X')
}

# Persistent user namespace: variables live across calls
PERSIST_NS = dict(SANDBOX_GLOBALS)


# ---- Simple UNIX-socket message framing (len + json) ----
def send_msg(conn, payload: dict) -> None:
    raw = json.dumps(payload).encode("utf-8")
    conn.sendall(struct.pack("!I", len(raw)) + raw)


def recv_msg(conn) -> dict:
    head = conn.recv(4)
    if not head:
        return {}
    n = struct.unpack("!I", head)[0]
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            break
        data += chunk
    if not data:
        return {}
    return json.loads(data.decode("utf-8"))


# ---- Idle hibernation thread (Mode 2) ----
_last_exec_ts = time.time()


def mark_exec() -> None:  # update on every code run
    global _last_exec_ts
    _last_exec_ts = time.time()


def idle_sweeper() -> None:
    while True:
        time.sleep(10)
        idle = time.time() - _last_exec_ts
        if idle >= IDLE_HIBERNATE_SEC:
            try:
                msg, purged = snapshot_and_compact(PERSIST_NS, _save_df_name_first, save_np)
                log(msg)
                if purged:
                    print("EVENT:" + json.dumps({"type": "session_purged", "objects": purged}))
            except Exception as e:  # noqa: BLE001 - log and continue
                log(f"[WARN] hibernate error: {e}")


def log(msg: str) -> None:
    if PRINT_DEBUG:
        print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


# ---- Code execution ----
def exec_code(code: str) -> dict:
    """
    Execute arbitrary Python code in the daemon's global scope.
    Returns stdout/err text and a short status.
    """
    mark_exec()
    import contextlib
    import io

    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            # Execute in persistent namespace so variables survive across calls
            exec(compile(code, "<sandbox>", "exec"), PERSIST_NS, PERSIST_NS)
        out = stdout.getvalue()
        err = stderr.getvalue()
        status = "[OK] exec"
    except Exception:  # noqa: BLE001 - capture traceback into stderr
        out = stdout.getvalue()
        err = stderr.getvalue() + "\n" + traceback.format_exc()
        status = "[ERROR] exec"
    return {"status": status, "stdout": out, "stderr": err}


# ---- Main loop ----
def main() -> None:
    # Clean old socket if present
    if SOCK_PATH.exists():
        SOCK_PATH.unlink()
    # Create UNIX socket server
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(SOCK_PATH))
    os.chmod(str(SOCK_PATH), 0o660)
    server.listen(1)
    log("[BOOT] sandbox daemon started")
    # Start idle sweeper (Mode 2)
    threading.Thread(target=idle_sweeper, daemon=True).start()
    while True:
        conn, _ = server.accept()
        try:
            req = recv_msg(conn)
            if not req:
                send_msg(conn, {"status": "[ERROR] empty"})
                conn.close()
                continue
            if req.get("op") == "exec":
                res = exec_code(req.get("code", ""))
                send_msg(conn, res)
            elif req.get("op") == "ping":
                send_msg(conn, {"status": "[OK] pong"})
            elif req.get("op") == "shutdown":
                # Acknowledge, then exit process to stop container
                send_msg(conn, {"status": "[OK] shutdown"})
                conn.close()
                # Delay exit slightly to allow response to flush
                threading.Timer(0.1, lambda: os._exit(0)).start()
                continue
            else:
                send_msg(conn, {"status": "[ERROR] bad_op"})
        except Exception as e:  # noqa: BLE001 - report to client
            send_msg(conn, {"status": "[ERROR] server", "stderr": str(e)})
        finally:
            conn.close()


if __name__ == "__main__":
    main()



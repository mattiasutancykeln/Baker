import os
import time
from pathlib import Path
import stat

# Ensure we can import sandbox_util from sibling directory
import sys

THIS_DIR = Path(__file__).resolve().parent
SANDBOX_DIR = THIS_DIR.parent
if str(SANDBOX_DIR) not in sys.path:
    sys.path.insert(0, str(SANDBOX_DIR))

from sandbox_util import build_image, exec_code, ping, start_container, stop_container  # noqa: E402


SLUG = "test-sandbox"
PROJ = Path(__file__).resolve().parents[2] / "projects" / SLUG
SOCKET_DIR = None


def _host_socket_dir_for_slug(slug: str) -> str:
    # Use per-user runtime dir to avoid privileged paths or sticky bit surprises
    return str(Path.home() / ".agent_sockets" / slug)


def setup_module(_):
    global SOCKET_DIR
    PROJ.mkdir(parents=True, exist_ok=True)
    (PROJ / "data").mkdir(exist_ok=True)
    build_image()
    # On WSL/DrvFS, use a Linux-native socket dir and bind to /workspace/.sandbox
    SOCKET_DIR = _host_socket_dir_for_slug(SLUG) if str(PROJ).startswith("/mnt/") else None
    if SOCKET_DIR:
        p = Path(SOCKET_DIR)
        p.mkdir(parents=True, exist_ok=True)
        try:
            p.chmod(stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777 to be permissive across uid maps
        except Exception:
            pass
    start_container(str(PROJ), SLUG, idle_hibernate_sec=5, socket_dir=SOCKET_DIR)


def teardown_module(_):
    stop_container(SLUG)


def test_00_ping():
    assert ping(str(PROJ), socket_dir=SOCKET_DIR) is True


def test_01_persistence_within_session_and_pandas_ops():
    # Create DataFrame, do some pandas ops, persist variable across calls
    r1 = exec_code(
        str(PROJ),
        "import pandas as pd; df = pd.DataFrame({'a':[1,2,3], 'b':[10,20,30]}); print(df.shape)",
        socket_dir=SOCKET_DIR,
    )
    assert r1["status"] == "[OK] exec"
    assert "(3, 2)" in (r1.get("stdout") or "")

    # Groupby and aggregation
    r2 = exec_code(
        str(PROJ),
        "g = df.groupby((df['a']%2)==0)['b'].mean(); print(round(g.mean(),1))",
        socket_dir=SOCKET_DIR,
    )
    assert r2["status"] == "[OK] exec"

    # Persistence across calls
    r3 = exec_code(
        str(PROJ),
        "df['c']=df['a']+df['b']; print(df['c'].sum())",
        socket_dir=SOCKET_DIR,
    )
    assert r3["status"] == "[OK] exec"
    assert "66" in (r3.get("stdout") or "")


def test_02_read_write_database_and_reload():
    # Save and load via parquet, then verify persisted file exists
    r1 = exec_code(
        str(PROJ),
        "save(df,'df_persist'); import os; print(os.path.exists('/workspace/data/df_persist.parquet'))",
        socket_dir=SOCKET_DIR,
    )
    assert r1["status"] == "[OK] exec"
    assert "True" in (r1.get("stdout") or "")

    r2 = exec_code(str(PROJ), "x=load('df_persist'); print(x.shape)", socket_dir=SOCKET_DIR)
    assert r2["status"] == "[OK] exec"
    assert "(3, 3)" in (r2.get("stdout") or "")


def test_03_pause_idle_and_hibernate_large_df():
    # Create a big df, wait beyond idle + sweeper, assert it snapshots and clears
    r1 = exec_code(
        str(PROJ),
        "import pandas as pd; big = pd.DataFrame({'x': list(range(3000000))}); print(len(big))",
        socket_dir=SOCKET_DIR,
    )
    assert r1["status"] == "[OK] exec"
    # Wait > idle_hibernate_sec (5s) and > sweeper interval (10s)
    time.sleep(16)
    r2 = exec_code(
        str(PROJ),
        "print('big' in globals()); import os; print(os.path.exists('/workspace/data/big.parquet'))",
        socket_dir=SOCKET_DIR,
    )
    out = (r2.get("stdout") or "")
    assert "False" in out and "True" in out


def test_04_restart_session_and_reload_saved_data():
    # Stop and restart container; verify parquet data persists and is readable
    stop_container(SLUG)
    start_container(str(PROJ), SLUG, idle_hibernate_sec=5, socket_dir=SOCKET_DIR)
    assert ping(str(PROJ), socket_dir=SOCKET_DIR) is True
    r = exec_code(str(PROJ), "y=load('df_persist'); print(y.shape)", socket_dir=SOCKET_DIR)
    assert r["status"] == "[OK] exec"
    assert "(3, 3)" in (r.get("stdout") or "")


def test_05_cleanup_and_close_session():
    # Stop the container and ensure socket is gone and ping fails
    stop_container(SLUG)
    assert ping(str(PROJ), socket_dir=SOCKET_DIR) is False
    sock_path = Path(SOCKET_DIR) / "kernel.sock" if SOCKET_DIR else PROJ / ".sandbox" / "kernel.sock"
    assert not sock_path.exists()



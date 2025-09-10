import json
import os
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Optional


IMAGE = "local/sandbox-py311-rdkit:latest"
CONTAINER_PREFIX = "sandbox_"


def sh(cmd: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True, **kw)


def build_image() -> None:
    here = Path(__file__).parent
    sh(["docker", "build", "-t", IMAGE, str(here)])


def start_container(
    project_dir: str,
    slug: str,
    idle_hibernate_sec: int = 900,
    mem: str = "4g",
    cpus: str = "1.0",
    socket_dir: Optional[str] = None,
) -> str:
    """
    Starts a hardened container with read-only rootfs, no network,
    and /workspace bound to the given project_dir. Returns container name.
    """
    name = f"{CONTAINER_PREFIX}{slug}"
    # stop any existing
    try:
        sh(["docker", "rm", "-f", name], check=False)
    except Exception:
        pass

    envs = [
        f"IDLE_HIBERNATE_SEC={idle_hibernate_sec}",
        "WORKSPACE_DIR=/workspace",
        "DEBUG=0",
    ]
    binds = [f"{project_dir}:/workspace:rw"]

    # If the project dir is on a Windows mount (/mnt/*), the filesystem likely
    # doesn't support AF_UNIX sockets. Use a Linux-native, user-writable socket directory.
    is_drvfs = project_dir.startswith("/mnt/")
    if socket_dir is None and is_drvfs:
        socket_dir = f"/tmp/agent_sockets/{slug}"
    if socket_dir:
        p = Path(socket_dir)
        p.mkdir(parents=True, exist_ok=True)
        # Remove stale socket if present
        sock_path = p / "kernel.sock"
        try:
            if sock_path.exists():
                sock_path.unlink()
        except Exception:
            pass
        binds.append(f"{socket_dir}:/workspace/.sandbox:rw")
    args = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/home/sandboxuser/.cache",
        "--user",
        "1000:1000",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--cpus",
        cpus,
        "--memory",
        mem,
        "--network",
        "none",
    ]
    for e in envs:
        args += ["-e", e]
    for b in binds:
        args += ["-v", b]
    args += [IMAGE]
    sh(args)
    # wait until daemon responds to ping (up to ~20s)
    # Prefer socket_dir if provided, else the project_dir .sandbox
    sock = Path(socket_dir) / "kernel.sock" if socket_dir else Path(project_dir) / ".sandbox" / "kernel.sock"
    for _ in range(80):
        if sock.exists():
            try:
                if ping(project_dir, socket_dir=socket_dir):
                    # Informative event on (re)start
                    _send_req(str(sock), {"op": "exec", "code": "print('EVENT:' + __import__('json').dumps({'type':'session_started'}))"})
                    break
            except Exception:
                pass
        time.sleep(0.25)
    return name


def stop_container(slug: str) -> None:
    name = f"{CONTAINER_PREFIX}{slug}"
    sh(["docker", "rm", "-f", name], check=False)
    # Best-effort cleanup of host socket in common locations
    candidates = [
        Path(f"/var/run/agent_sockets/{slug}"),
        Path(f"/tmp/agent_sockets/{slug}"),
        Path.home() / ".agent_sockets" / slug,
    ]
    for base in candidates:
        try:
            sock = base / "kernel.sock"
            if sock.exists():
                sock.unlink()
        except Exception:
            pass


# --- UNIX socket RPC ---
def _send_req(sock_path: str, payload: dict, timeout: float = 120.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(sock_path)
        data = json.dumps(payload).encode("utf-8")
        s.sendall(struct.pack("!I", len(data)) + data)
        head = s.recv(4)
        if not head:
            return {}
        n = struct.unpack("!I", head)[0]
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8"))


def exec_code(project_dir: str, code: str, timeout: float = 120.0, socket_dir: Optional[str] = None) -> dict:
    if socket_dir:
        sock_path = Path(socket_dir) / "kernel.sock"
    else:
        sock_path = Path(project_dir) / ".sandbox" / "kernel.sock"
    return _send_req(str(sock_path), {"op": "exec", "code": code}, timeout=timeout)


def ping(project_dir: str, socket_dir: Optional[str] = None) -> bool:
    sock = str((Path(socket_dir) if socket_dir else Path(project_dir) / ".sandbox") / "kernel.sock")
    try:
        r = _send_req(sock, {"op": "ping"}, timeout=5.0)
        return r.get("status") == "[OK] pong"
    except Exception:
        return False


def shutdown(project_dir: str, socket_dir: Optional[str] = None) -> bool:
    sock = str((Path(socket_dir) if socket_dir else Path(project_dir) / ".sandbox") / "kernel.sock")
    try:
        r = _send_req(sock, {"op": "shutdown"}, timeout=5.0)
        return r.get("status") == "[OK] shutdown"
    except Exception:
        return False


if __name__ == "__main__":
    # quick manual smoke
    proj = os.path.abspath("../projects/demo-slug")
    Path(proj).mkdir(parents=True, exist_ok=True)
    build_image()
    # On DrvFS, map a Linux-native socket dir
    sockdir = "/var/run/agent_sockets/demo-slug" if proj.startswith("/mnt/") else None
    cid = start_container(proj, "demo-slug", socket_dir=sockdir)
    print("ping:", ping(proj, socket_dir=sockdir))
    print(
        exec_code(
            proj,
            "import pandas as pd; df=pd.DataFrame({'x':[1,2]}); save(df,'toy'); print(load('toy').to_dict())",
            socket_dir=sockdir,
        )
    )
    stop_container("demo-slug")



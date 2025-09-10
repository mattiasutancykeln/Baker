# Sandbox (KernelLite) – Containerized Python Sandbox

A secure local sandbox that executes Python code snippets for data analysis with very low latency across many calls.

- Persistent daemon (“KernelLite”) inside a hardened Docker container
- Preloaded scientific stack: pandas, numpy, scikit-learn, pyarrow, rdkit
- Helper IO:
  - `load(name)` → pandas DataFrame from `/workspace/data/{name}.parquet`
  - `save(df, name)` → atomic write to `/workspace/data/{name}.parquet`
- Communication via UNIX domain socket at `/workspace/.sandbox/kernel.sock` (no TCP)
- Hibernation on idle: snapshots large DataFrames/arrays to disk and frees RAM

## Build image

```bash
# From repo root
docker build -t local/sandbox-py311-rdkit:latest sandbox
```

## Run container

Hardened settings: non-root, read-only rootfs, no network, tmpfs for /tmp and cache.

- Linux host (native ext4):
```bash
SLUG=my-project
PROJ=/abs/path/to/projects/$SLUG

docker run -d \
  --name sandbox_$SLUG \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /home/sandboxuser/.cache \
  --user 1000:1000 \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --pids-limit 256 \
  --cpus 1.0 \
  --memory 4g \
  --network none \
  -e IDLE_HIBERNATE_SEC=900 \
  -e WORKSPACE_DIR=/workspace \
  -e DEBUG=0 \
  -v "$PROJ:/workspace:rw" \
  local/sandbox-py311-rdkit:latest
```

- WSL/DrvFS (Windows path under /mnt/...): UNIX sockets are not supported on DrvFS. Mount a Linux-native socket dir into `/workspace/.sandbox`:
```bash
SLUG=my-project
PROJ=/mnt/c/Users/<you>/.../projects/$SLUG
SOCK=/var/run/agent_sockets/$SLUG
sudo mkdir -p "$SOCK" && sudo chown 1000:1000 "$SOCK" && sudo chmod 770 "$SOCK"

docker run -d \
  --name sandbox_$SLUG \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /home/sandboxuser/.cache \
  --user 1000:1000 \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --pids-limit 256 \
  --cpus 1.0 \
  --memory 4g \
  --network none \
  -e IDLE_HIBERNATE_SEC=900 \
  -e WORKSPACE_DIR=/workspace \
  -e DEBUG=0 \
  -v "$PROJ:/workspace:rw" \
  -v "$SOCK:/workspace/.sandbox:rw" \
  local/sandbox-py311-rdkit:latest
```

## Python control API (start, exec, shutdown)

Use `sandbox/sandbox_util.py` from the host.

```python
from sandbox.sandbox_util import start_container, ping, exec_code, shutdown, stop_container

slug = "my-project"
proj = "/abs/path/to/projects/" + slug
sock = f"/var/run/agent_sockets/{slug}"  # optional; recommended on WSL

# Start
start_container(proj, slug, idle_hibernate_sec=900, socket_dir=sock)
assert ping(proj, socket_dir=sock)

# Execute code (session persists across calls)
r = exec_code(proj, "import pandas as pd; df=pd.DataFrame({'x':[1,2,3]}); print(df.shape)", socket_dir=sock)
print(r["status"], r["stdout"])  # [OK] exec (3, 1)

# Graceful shutdown (daemon exits, container stops)
shutdown(proj, socket_dir=sock)

# Hard stop/remove (requires Docker permission)
stop_container(slug)
```

## LangGraph tool: `pandastool`

`generated_tools/pandastool.py` exposes a tool callable by an agent to execute code in the sandbox. Sessions persist.

Signature:
```python
pandastool(code: str, project_dir: str, socket_dir: str | None = None, *, db=None, config=None) -> str
```
Returns a JSON string: `{ status, stdout, stderr }`.

Example:
```python
from generated_tools.pandastool import pandastool
out = pandastool(
    code="import pandas as pd; df=pd.DataFrame({'a':[1,2,3]}); save(df,'demo'); print(load('demo').shape)",
    project_dir=proj,
    socket_dir=sock,
)
print(out)
```

## Hibernation (Mode 2)

- Configure with `IDLE_HIBERNATE_SEC` at container start (default 900s)
- Background sweeper runs every ~10s. On idle threshold:
  - Snapshots large DataFrames/arrays to `/workspace/data/`
  - Removes them from memory

## Tests

- Minimal sandbox verification:
```bash
pytest -q sandbox/tests/test_sandbox_minimal.py
```

- Tool-level tests (uses already running container/socket; will skip if unavailable unless forced):
```bash
export PANDAS_SANDBOX_SLUG=my-project
export PANDAS_SANDBOX_PROJECT_DIR=/abs/path/to/projects/$PANDAS_SANDBOX_SLUG
export PANDAS_SANDBOX_SOCKET_DIR=/var/run/agent_sockets/$PANDAS_SANDBOX_SLUG
# optionally force run even if pre-checks would skip
# export PANDAS_SANDBOX_FORCE=1
pytest -q tests/test_pandastool.py
```

## Security hardening

- Non-root user, read-only rootfs, no-new-privileges
- No network (`--network none`)
- Limited pids, CPU, memory; tmpfs mounts for `/tmp` and cache
- Optional: run with gVisor (`--runtime=runsc`), seccomp/AppArmor profiles, and `--cap-drop ALL` (already in examples)

## Troubleshooting

- OSError 95 (Operation not supported) on socket bind
  - Cause: UNIX sockets on DrvFS (`/mnt/...`)
  - Solution: mount a Linux-native host dir (e.g., `/var/run/agent_sockets/<slug>`) to `/workspace/.sandbox`

- Connection refused to socket
  - Cause: stale socket file or daemon not running
  - Fix: remove stale socket, restart container, ensure socket dir owned by uid 1000
  - Commands:
    ```bash
    sudo rm -f /var/run/agent_sockets/<slug>/kernel.sock
    docker rm -f sandbox_<slug> || true
    sudo chown -R 1000:1000 /var/run/agent_sockets/<slug> && sudo chmod 770 /var/run/agent_sockets/<slug>
    ```

- Permission denied writing outside `/workspace`
  - Expected due to read-only rootfs

- Network unreachable
  - Expected due to `--network none`

- Code changes not taking effect
  - Rebuild image: `docker build -t local/sandbox-py311-rdkit:latest sandbox`
  - Restart container

## Notes

- The daemon maintains a persistent namespace, so variables survive across `exec` calls.
- `save(df, name)` expects the DataFrame as the first argument.
- Hibernation thresholds can be tuned in `snapshot_utils.py` (`DF_BYTES`, `NP_BYTES`).

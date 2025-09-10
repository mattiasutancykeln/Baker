from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
import traceback

from sandbox.sandbox_util import exec_code


def _sanitize_config(config: dict | None) -> dict:
    if not isinstance(config, dict):
        return {}
    redacted = {}
    for k, v in config.items():
        if any(s in k.lower() for s in ("key", "secret", "token", "password")):
            redacted[k] = "***"
        else:
            redacted[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return redacted


def pandastool(
    code: str,
    *,
    project_dir: Optional[str] = None,
    socket_dir: Optional[str] = None,
    db=None,
    config=None,
) -> str:
    """
    Execute concise pandas/numpy code in the project sandbox.

    - Injects DB/CFG context preamble without exposing data to the LLM context
    - Returns a JSON string: { status, stdout, stderr }
    """
    if not isinstance(code, str) or not code.strip():
        return json.dumps({"status": "[ERROR] exec", "stdout": "", "stderr": "empty code"})

    # Resolve project_dir: prefer explicit arg, else env, else db root if available
    proj = (
        project_dir
        or os.environ.get("PANDAS_SANDBOX_PROJECT_DIR")
        or (str(getattr(db, "root_dir", "")) if db is not None else None)
    )
    if not proj or not Path(proj).exists():
        return json.dumps({"status": "[ERROR] exec", "stdout": "", "stderr": "invalid project_dir"})

    # Compute dataset hints and config for the sandbox code preamble
    data_dir = Path(proj) / "data"
    datasets = [p.stem for p in data_dir.glob("*.parquet")] if data_dir.exists() else []
    cfg = _sanitize_config(config if isinstance(config, dict) else {})

    preamble = (
        "# ---- injected context ----\n"
        f"DB_ROOT = {json.dumps(str(data_dir))}\n"
        f"DATASETS = {json.dumps(sorted(datasets))}\n"
        f"CFG = {json.dumps(cfg)}\n"
        "# --------------------------\n"
    )

    try:
        res = exec_code(proj, preamble + "\n" + code, socket_dir=socket_dir, timeout=300.0)
        return json.dumps({
            "status": res.get("status"),
            "stdout": res.get("stdout") or "",
            "stderr": res.get("stderr") or "",
        })
    except Exception as e:  # noqa: BLE001 - return error to caller
        return json.dumps({"status": "[ERROR] exec", "stdout": "", "stderr": str(e) + "\n" + traceback.format_exc()})



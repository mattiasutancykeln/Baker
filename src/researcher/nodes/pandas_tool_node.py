from __future__ import annotations

from typing import Any, Dict
import json
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph import MessagesState

from src.researcher.tools.pandastool import pandastool as _pandastool
from sandbox.sandbox_util import build_image as sandbox_build, start_container as sandbox_start, ping as sandbox_ping
from src.researcher.db import ArrowDatabase
from src.researcher.base import UnifiedNodeBase


class PandasToolNode(UnifiedNodeBase):
    """
    Executes sandbox pandas tool calls emitted by the LLM.

    - Exposes no data to the LLM; injects dummy db/config internally
    - Appends ToolMessage responses for each pandastool call
    """

    def __init__(self, *, project_dir: str, socket_dir: str, db: ArrowDatabase, cfg: Dict[str, Any]) -> None:
        super().__init__(db=db)
        self._project_dir = project_dir
        self._socket_dir = socket_dir
        self._cfg = cfg

    def __call__(self, state: MessagesState) -> dict:
        last: AnyMessage = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": []}

        # Ensure sandbox is ready; restart once on failure
        try:
            sandbox_build()
        except Exception:
            pass
        if not sandbox_ping(self._project_dir, socket_dir=self._socket_dir):
            sandbox_start(self._project_dir, Path(self._project_dir).name, socket_dir=self._socket_dir)
            if not sandbox_ping(self._project_dir, socket_dir=self._socket_dir):
                err = json.dumps({"status":"[ERROR] exec","stdout":"","stderr":"sandbox unavailable"})
                return {"messages": [ToolMessage(tool_call_id="", content=err, name="pandastool")]}

        out: list[AnyMessage] = []
        for tc in last.tool_calls:
            name = tc.get("name")
            args = tc.get("args") or {}

            if name == "pandastool":
                code = args.get("code", "")
                raw = _pandastool(
                    code=code,
                    project_dir=self._project_dir,
                    socket_dir=self._socket_dir,
                    db=self._db,
                    config=self._cfg,
                )
                # Parse session EVENT lines (start/purge) for user info; do not update registry
                try:
                    res = json.loads(raw)
                except Exception:
                    res = {"status": "[ERROR] exec", "stdout": "", "stderr": raw}
                stdout = res.get("stdout") or ""
                cleaned_lines = []
                for line in stdout.splitlines():
                    if line.startswith("EVENT:"):
                        try:
                            ev = json.loads(line[len("EVENT:"):].strip())
                            if ev.get("type") == "session_started":
                                cleaned_lines.append("[sandbox] session started or restarted; state is fresh.")
                            elif ev.get("type") == "session_purged":
                                objs = ev.get("objects", [])
                                cleaned_lines.append(f"[sandbox] idle hibernation purged large objects: {', '.join(objs)}")
                        except Exception:
                            pass
                    else:
                        cleaned_lines.append(line)
                res["stdout"] = "\n".join(cleaned_lines)
                out.append(ToolMessage(tool_call_id=tc.get("id"), content=json.dumps(res), name="pandastool"))

            elif name == "update_data_description":
                dataset_name = (args.get("name") or "").strip()
                description = (args.get("description") or "").strip()
                if not dataset_name:
                    content = json.dumps({"status": "[ERROR] update_description", "stderr": "missing dataset name"})
                else:
                    try:
                        self._db.set_description(dataset_name, description)
                        content = json.dumps({"status": "[OK] update_description", "dataset": dataset_name})
                    except Exception as e:
                        content = json.dumps({"status": "[ERROR] update_description", "stderr": str(e)})
                out.append(ToolMessage(tool_call_id=tc.get("id"), content=content, name="update_data_description"))

            elif name == "list_datasets":
                content = self._tool_list_datasets()
                out.append(ToolMessage(tool_call_id=tc.get("id"), content=content, name=name))

            elif name == "dataset_describe":
                content = self._tool_dataset_describe(args.get("name", ""))
                out.append(ToolMessage(tool_call_id=tc.get("id"), content=content, name=name))

            elif name == "dataset_head":
                content = self._tool_dataset_head(args.get("name", ""), int(args.get("n", 5)))
                out.append(ToolMessage(tool_call_id=tc.get("id"), content=content, name=name))

            else:
                # Always return a tool_result for any tool_use to satisfy the protocol
                content = json.dumps({"status": "[ERROR] unknown_tool", "tool": name})
                out.append(ToolMessage(tool_call_id=tc.get("id"), content=content, name=name or "unknown"))
        return {"messages": out}



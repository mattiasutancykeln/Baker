from __future__ import annotations

import importlib.util
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model

from src.researcher.db import ArrowDatabase
from src.researcher.project_config import ProjectConfig


@dataclass
class LoadedTool:
    name: str
    fn: Callable[..., str]
    description: str
    tool: StructuredTool


def _load_module(path: Path) -> Optional[ModuleType]:
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if not spec or not spec.loader:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _build_args_schema(fn: Callable[..., Any]) -> type[BaseModel]:
    sig = inspect.signature(fn)
    fields: Dict[str, tuple[type, Any]] = {}
    for name, param in sig.parameters.items():
        if name in {"db", "config"}:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = param.annotation if param.annotation is not inspect._empty else str
        default = param.default if param.default is not inspect._empty else ...
        # Fallback unknown annotations to str
        if not isinstance(ann, type) or ann not in {str, int, float, bool}:
            ann = str
        fields[name] = (ann, default)
    if not fields:
        # At least one dummy field to satisfy StructuredTool
        fields["_"] = (str, "")
    model = create_model(fn.__name__ + "Args", **fields)  # type: ignore[arg-type]
    return model


def _wrap_tool(fn: Callable[..., str], *, db: ArrowDatabase, cfg: ProjectConfig) -> Callable[..., str]:
    def _wrapped(**kwargs: Any) -> str:  # noqa: ANN401 - tool boundary
        try:
            # Remove placeholder key if present
            kwargs.pop("_", None)
            result = fn(**kwargs, db=db, config=cfg.__dict__)
            if not isinstance(result, str):
                return f"Error: tool '{fn.__name__}' returned non-string response."
            return result
        except Exception as e:  # noqa: BLE001 - surface error to LLM as string
            return f"Error in tool '{fn.__name__}': {str(e)}"
    return _wrapped


def load_generated_tools(tool_dir: str, *, db: ArrowDatabase, cfg: ProjectConfig) -> Dict[str, LoadedTool]:
    base = Path(tool_dir)
    if not base.exists():
        return {}

    # Load descriptions from JSON toolcards
    descriptions: Dict[str, str] = {}
    for card in base.glob("*.json"):
        try:
            with card.open("r", encoding="utf-8") as f:
                data = json.load(f)
            name = (data.get("name") or card.stem).strip()
            desc = (data.get("description") or "").strip()
            descriptions[name] = desc
        except Exception:
            continue

    registry: Dict[str, LoadedTool] = {}
    for py in base.glob("*.py"):
        module = _load_module(py)
        if not module:
            continue
        for attr_name, obj in vars(module).items():
            if not callable(obj):
                continue
            if attr_name.startswith("_"):
                continue
            # Heuristic: only include functions that follow our pattern
            try:
                sig = inspect.signature(obj)
                if "db" not in sig.parameters or "config" not in sig.parameters:
                    continue
            except Exception:
                continue

            tool_name = attr_name
            desc = descriptions.get(tool_name, f"Generated tool: {tool_name}")
            args_schema = _build_args_schema(obj)
            wrapped = _wrap_tool(obj, db=db, cfg=cfg)
            structured = StructuredTool.from_function(
                lambda **kwargs: wrapped(**kwargs),  # type: ignore[misc]
                name=tool_name,
                description=desc,
                args_schema=args_schema,
            )
            registry[tool_name] = LoadedTool(name=tool_name, fn=obj, description=desc, tool=structured)

    return registry



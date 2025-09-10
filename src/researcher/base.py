from __future__ import annotations

from typing import Any, List

import pandas as pd
from langchain_core.tools import StructuredTool

from src.researcher.db import ArrowDatabase, DESCR_TABLE, REGISTRY_TABLE
from src.researcher.project_config import ProjectConfig


class UnifiedNodeBase:
    """Base for all LLM nodes to share dataset tools and context.

    - Provides read-only dataset tools bound per turn
    - Keeps db/cfg internal and out of LLM-facing signatures
    """

    _HIDDEN_DATASETS = {DESCR_TABLE, REGISTRY_TABLE}

    def __init__(self, *, db: ArrowDatabase, cfg: ProjectConfig | None = None, node_name: str | None = None) -> None:
        self._db = db
        self._cfg = cfg
        self._node_name = node_name or self.__class__.__name__

    # ---------- Internal helpers ----------
    def _truncate_text(self, text: str, limit: int = 1000) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "\u2026"

    def _format_head(self, name: str, n: int) -> str:
        if not self._db.has_table(name):
            return f"Error: dataset '{name}' not found."
        df: pd.DataFrame = self._db.get_table(name).to_pandas()
        n = max(1, min(int(n), 25))
        df = df.head(n)

        cols = list(df.columns)
        if len(cols) > 25:
            # Show first 21, sentinel '.', last 3 → total 25
            left = cols[:21]
            right = cols[-3:]
            view = pd.DataFrame()
            if left:
                view[left] = df[left]
            view["."] = "\u2026"
            for c in right:
                view[c] = df[c]
        else:
            view = df
        out = view.to_string(index=False)
        return self._truncate_text(out, 1000)

    # ---------- Default tools (read-only) ----------
    def _tool_list_datasets(self) -> str:
        names = [n for n in self._db.list_tables() if n not in self._HIDDEN_DATASETS]
        text = ", ".join(sorted(names)) if names else "<empty>"
        return self._truncate_text(text, 1000)

    def _tool_dataset_describe(self, name: str) -> str:  # noqa: ANN001
        name = (name or "").strip()
        if not name:
            return "Error: missing dataset name."
        if not self._db.has_table(name):
            return f"Error: dataset '{name}' not found."
        d = self._db.get_description(name) or "No description available."
        return self._truncate_text(d, 1000)

    def _tool_dataset_head(self, name: str, n: int = 5) -> str:  # noqa: ANN001
        name = (name or "").strip()
        if not name:
            return "Error: missing dataset name."
        return self._format_head(name, n)

    def get_default_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(self._tool_list_datasets, name="list_datasets", description="List available datasets in the project database (excludes system tables)."),
            StructuredTool.from_function(self._tool_dataset_describe, name="dataset_describe", description="Show the natural-language description of a dataset from the 'descr' table."),
            StructuredTool.from_function(self._tool_dataset_head, name="dataset_head", description="Show the first N rows of a dataset (max 25 rows, 25 columns; middle columns truncated)."),
        ]

    def bind_default_tools(self, llm: Any, extra_tools: list[StructuredTool] | None = None) -> Any:
        tools = self.get_default_tools()
        if extra_tools:
            tools.extend(extra_tools)
        return llm.bind_tools(tools)



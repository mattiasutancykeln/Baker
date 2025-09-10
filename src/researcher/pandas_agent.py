from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.researcher.db import ArrowDatabase
from src.researcher.nodes.data_formatter import build_data_formatter_app
from src.researcher.project_config import ensure_project_config


def create_app() -> Any:
    """Factory for `langgraph dev` to run the Data Formatter app in isolation."""
    # Default to the sample researcher project; override by editing as needed
    project_path = str(Path(__file__).resolve().parents[2] / "projects" / "researcher_20250908_093746")
    cfg = ensure_project_config(project_path)
    db = ArrowDatabase(root_dir=Path(cfg.data_dir))
    model_name = os.environ.get("ANTHROPIC_MODEL", "anthropic:claude-sonnet-4-20250514")
    return build_data_formatter_app(
        project_dir=cfg.project_dir,
        socket_dir=cfg.socket_dir,
        db=db,
        model_name=model_name,
    )



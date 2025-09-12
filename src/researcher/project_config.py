from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ProjectConfig:
    slug: str
    project_dir: str
    data_dir: str
    output_path: str
    socket_dir: str
    sandbox_image: str
    idle_hibernate_sec: int


def ensure_project_config(project_path: str) -> ProjectConfig:
    proj = Path(project_path)
    slug = proj.name
    cfg_path = proj / "config.json"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            raw: Dict[str, Any] = json.load(f)
        # Ensure defaults and required paths exist
        data_dir = raw.get("data_dir", str(proj / "data"))
        output_path = raw.get("output_path", str(proj / "output"))
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        # Persist output_path if it was missing
        if "output_path" not in raw:
            raw["output_path"] = output_path
            with cfg_path.open("w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2)
        return ProjectConfig(
            slug=raw.get("slug", slug),
            project_dir=str(proj),
            data_dir=data_dir,
            output_path=output_path,
            socket_dir=raw.get("socket_dir", str(Path.home() / ".agent_sockets" / slug)),
            sandbox_image=raw.get("sandbox_image", "local/sandbox-py311-rdkit:latest"),
            idle_hibernate_sec=int(raw.get("idle_hibernate_sec", 900)),
        )

    data_dir = proj / "data"
    output_dir = proj / "output"
    socket_dir = Path.home() / ".agent_sockets" / slug
    socket_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = {
        "slug": slug,
        "project_dir": str(proj),
        "data_dir": str(data_dir),
        "output_path": str(output_dir),
        "socket_dir": str(socket_dir),
        "sandbox_image": "local/sandbox-py311-rdkit:latest",
        "idle_hibernate_sec": 900,
    }
    with (proj / "config.json").open("w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
    return ProjectConfig(
        slug=slug,
        project_dir=str(proj),
        data_dir=str(data_dir),
        output_path=str(output_dir),
        socket_dir=str(socket_dir),
        sandbox_image="local/sandbox-py311-rdkit:latest",
        idle_hibernate_sec=900,
    )



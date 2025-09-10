from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DESCR_TABLE = "descr"
REGISTRY_TABLE = "registry"


def _parquet_path(root: Path, name: str) -> Path:
    return root / f"{name}.parquet"


@dataclass
class ArrowDatabase:
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        # Ensure support tables exist
        if not self.has_table(DESCR_TABLE):
            self.save_table(DESCR_TABLE, pd.DataFrame({"dataset": [], "description": []}))
        if not self.has_table(REGISTRY_TABLE):
            self.save_table(
                REGISTRY_TABLE,
                pd.DataFrame({
                    "dataset": [],
                    "owner_node": [],
                    "lineage": [],
                    "version": [],
                    "schema_hash": [],
                    "created_at": [],
                }),
            )

    # Basic CRUD
    def list_tables(self) -> List[str]:
        return [p.stem for p in self.root_dir.glob("*.parquet")]

    def has_table(self, name: str) -> bool:
        return _parquet_path(self.root_dir, name).exists()

    def get_table(self, name: str) -> pa.Table:
        path = _parquet_path(self.root_dir, name)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {name}")
        return pq.read_table(path)

    def save_table(self, name: str, table: pd.DataFrame | pa.Table) -> None:
        if isinstance(table, pd.DataFrame):
            pa_table = pa.Table.from_pandas(table)
        else:
            pa_table = table
        pq.write_table(pa_table, _parquet_path(self.root_dir, name))

    def delete_table(self, name: str) -> None:
        path = _parquet_path(self.root_dir, name)
        if path.exists():
            path.unlink()

    # Compatibility methods for sandbox/tooling
    def path_of(self, name: str) -> Path:
        return _parquet_path(self.root_dir, name)

    def read(self, name: str) -> pa.Table:
        return self.get_table(name)

    def write(self, name: str, table: pa.Table) -> None:
        pq.write_table(table, self.path_of(name))

    # Descriptions
    def get_description(self, name: str) -> Optional[str]:
        descr = self.get_table(DESCR_TABLE).to_pandas()
        row = descr.loc[descr["dataset"] == name]
        if row.empty:
            return None
        return str(row.iloc[0]["description"])

    def set_description(self, name: str, description: str) -> None:
        descr = self.get_table(DESCR_TABLE).to_pandas()
        if name in descr["dataset"].values:
            descr.loc[descr["dataset"] == name, "description"] = description
        else:
            descr = pd.concat([descr, pd.DataFrame({"dataset": [name], "description": [description]})], ignore_index=True)
        self.save_table(DESCR_TABLE, descr)

    def remove_description(self, name: str) -> None:
        descr = self.get_table(DESCR_TABLE).to_pandas()
        descr = descr.loc[descr["dataset"] != name]
        self.save_table(DESCR_TABLE, descr)

    # Registry and provenance
    def _hash_schema(self, table: pd.DataFrame | pa.Table) -> str:
        if isinstance(table, pa.Table):
            columns = [f"{f.name}:{f.type}" for f in table.schema]
        else:
            columns = [f"{c}:{str(table[c].dtype)}" for c in table.columns]
        return hashlib.sha256("|".join(columns).encode()).hexdigest()[:16]

    def register(self, *, dataset: str, owner_node: str, lineage: str, version: str, table: pd.DataFrame | pa.Table) -> None:
        registry = self.get_table(REGISTRY_TABLE).to_pandas()
        registry = pd.concat([
            registry,
            pd.DataFrame({
                "dataset": [dataset],
                "owner_node": [owner_node],
                "lineage": [lineage],
                "version": [version],
                "schema_hash": [self._hash_schema(table)],
                "created_at": [datetime.utcnow().isoformat()],
            }),
        ], ignore_index=True)
        self.save_table(REGISTRY_TABLE, registry)

    def record_provenance(self, tool: str, inputs: List[str], outputs: List[str], lineage: str) -> None:
        try:
            reg = self.get_table(REGISTRY_TABLE).to_pandas()
            now = datetime.utcnow().isoformat()
            rows = []
            for ds in outputs:
                rows.append({
                    "dataset": ds,
                    "owner_node": tool,
                    "lineage": lineage or ",".join(inputs),
                    "version": now,
                    "schema_hash": "",
                    "created_at": now,
                })
            if rows:
                reg = pd.concat([reg, pd.DataFrame(rows)], ignore_index=True)
                self.save_table(REGISTRY_TABLE, reg)
        except Exception:
            # Best-effort; do not raise in tooling path
            pass



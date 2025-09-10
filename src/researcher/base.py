from __future__ import annotations

from typing import List

from src.researcher.db import ArrowDatabase, DESCR_TABLE


class UnifiedNodeBase:
    def __init__(self, *, db: ArrowDatabase) -> None:
        self._db = db

    # Shared dataset utilities (not exposed outside nodes by default)
    def list_datasets(self) -> List[str]:
        return [n for n in self._db.list_tables() if n != DESCR_TABLE]

    def dataset_describe(self, name: str) -> str:
        d = self._db.get_description(name)
        return d if d else "No description available."

    def dataset_head(self, name: str, n: int = 5) -> str:
        if not self._db.has_table(name):
            return f"Error: dataset '{name}' not found."
        df = self._db.get_table(name).to_pandas()
        limited = df.iloc[: max(0, n)].iloc[:, :10]
        return limited.to_string(index=False)



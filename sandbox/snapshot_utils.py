import gc
from typing import Any, Dict

import numpy as np
import pandas as pd


# Heuristics for "large" objects (tune thresholds)
DF_BYTES = 20_000_000  # 20 MB
NP_BYTES = 10_000_000  # 10 MB


def _df_size(df: pd.DataFrame) -> int:
    try:
        return int(df.memory_usage(index=True, deep=True).sum())
    except Exception:
        return 0


def is_big_df(x: Any) -> bool:
    return isinstance(x, pd.DataFrame) and _df_size(x) >= DF_BYTES


def is_big_np(x: Any) -> bool:
    try:
        return isinstance(x, np.ndarray) and int(x.nbytes) >= NP_BYTES
    except Exception:
        return False


def snapshot_and_compact(glob: Dict[str, Any], save_df, save_np) -> tuple[str, list[str]]:
    """
    save_df(name:str, df:DataFrame) → persist a DataFrame
    save_np(name:str, arr:np.ndarray) → persist an array (optional)
    """
    to_drop: list[str] = []
    for k, v in list(glob.items()):
        if k.startswith("_"):
            continue
        if is_big_df(v):
            try:
                save_df(k, v)
                to_drop.append(k)
            except Exception:
                pass
        elif is_big_np(v):
            try:
                save_np(k, v)
                to_drop.append(k)
            except Exception:
                pass
    for k in to_drop:
        try:
            del glob[k]
        except Exception:
            pass
    gc.collect()
    return (f"[OK] snapshot | saved={len(to_drop)} | compacted", to_drop)



"""I/O helpers — path constants and parquet loading."""

from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MASTER_PARQUET = PROCESSED_DIR / "all_admissions.parquet"


def master_exists() -> bool:
    return MASTER_PARQUET.exists()


def master_info() -> dict:
    """Return basic info about the master parquet without loading it fully."""
    if not master_exists():
        return {"exists": False}
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(MASTER_PARQUET)
    meta = pf.metadata
    return {
        "exists": True,
        "rows": meta.num_rows,
        "columns": meta.num_columns,
        "size_mb": round(MASTER_PARQUET.stat().st_size / 1e6, 1),
    }


def load_master(use_streamlit_cache: bool = False) -> pd.DataFrame:
    """Load the master all_admissions parquet.

    Parameters
    ----------
    use_streamlit_cache : bool
        If True, wrap with @st.cache_data (call from Streamlit context).
    """
    if use_streamlit_cache:
        try:
            import streamlit as st

            @st.cache_data(show_spinner="Loading master dataset...")
            def _cached_load():
                return pd.read_parquet(MASTER_PARQUET)

            return _cached_load()
        except ImportError:
            pass

    return pd.read_parquet(MASTER_PARQUET)


def list_configs() -> list[str]:
    """Return sorted list of config YAML filenames in configs/."""
    if not CONFIGS_DIR.exists():
        return []
    return sorted(p.name for p in CONFIGS_DIR.glob("*.yaml") if not p.name.startswith("_"))

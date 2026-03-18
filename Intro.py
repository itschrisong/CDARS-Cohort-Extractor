"""DoEMDataWrangler — Streamlit multipage entry point."""

import streamlit as st
from core.io import master_exists, master_info, list_configs, CONFIGS_DIR
from core.config import CohortConfig
from core.ui_helpers import clear_cohort_state

_FILTER_BUILDER_KEYS = (
    "incl_pick_filters",
    "excl_pick_filters",
    "incl_pick_logic",
    "excl_pick_logic",
    "incl_editing_idx",
    "excl_editing_idx",
    "inclusion_mode",
    "exclusion_mode",
)

st.set_page_config(
    page_title="CDARS DataWrangler",
    page_icon=":hospital:",
    layout="wide",
)

# --- Sidebar: Config management + data status ---
with st.sidebar:
    st.title("CDARS DataWrangler")

    # Master data status
    if master_exists():
        info = master_info()
        st.success(
            f"Master data: **{info['rows']:,}** rows, "
            f"{info['size_mb']} MB"
        )
    else:
        st.warning("No data loaded yet. Go to **0. Upload Data** to upload CDARS files.")

    st.divider()

    # Config management
    st.subheader("Configuration")

    configs = list_configs()
    config_options = ["(New config)"] + configs

    selected = st.selectbox("Load a saved config", config_options)

    if selected != "(New config)" and selected:
        config_path = CONFIGS_DIR / selected
        if st.button("Load", use_container_width=True):
            st.session_state["config"] = CohortConfig.from_yaml(config_path)
            st.session_state["config_name"] = selected.replace(".yaml", "")
            # Clear any previous results
            clear_cohort_state(st.session_state)
            # Atomically clear all filter-builder state (SESS-03)
            for key in _FILTER_BUILDER_KEYS:
                st.session_state.pop(key, None)
            # Pre-set mode to Advanced so loaded config rules are visible (SESS-01)
            st.session_state["inclusion_mode"] = "Advanced rules"
            st.session_state["exclusion_mode"] = "Advanced rules"
            st.toast(f"Loaded {selected}")
            st.rerun()

    config_name = st.text_input(
        "Config name",
        value=st.session_state.get("config_name", "my_cohort"),
    )
    st.session_state["config_name"] = config_name

    if st.button("Save current config", use_container_width=True):
        cfg = st.session_state.get("config", CohortConfig())
        save_path = CONFIGS_DIR / f"{config_name}.yaml"
        cfg.to_yaml(save_path)
        st.toast(f"Saved to {save_path.name}")

    st.divider()
    st.caption("Navigate pages using the sidebar above.")

# --- Initialise session state ---
if "config" not in st.session_state:
    st.session_state["config"] = CohortConfig()

# --- Main page ---
st.header("CDARS DataWrangler")
st.markdown(
    """
Extract any clinical cohort from the CDARS A&E master dataset using
flexible filter rules, then export as CSV or Parquet for analysis.

### Workflow

| Page | What it does | Data effect |
|------|-------------|-------------|
| **0. Upload Data** | Upload CDARS XLS/XLSX files. Files are parsed, deduplicated, and cached automatically. | Creates master dataset (all rows, all columns) |
| **1. Define Cohort** | Build inclusion/exclusion rules to define your cohort (e.g. ICD-9 codes, age, residency). See a STROBE flow of how many rows are removed at each step. | **Removes rows** only — all columns are preserved |
| **2. Download & Export** | View cohort summary stats, select columns, and export as Parquet/CSV plus config YAML. | **Choose columns** for the download file |

### Getting started

1. Go to **0. Upload Data** and upload your CDARS XLS/XLSX files (or load a saved config from the sidebar)
2. Go to **1. Define Cohort** and add your inclusion/exclusion rules
3. Click **Run Filter** to extract the cohort
4. Go to **2. Download & Export** to download your cohort
"""
)

# Show current config
cfg = st.session_state["config"]

# Show cohort status if available
if "cohort_df" in st.session_state:
    cdf = st.session_state["cohort_df"]
    st.success(f"Active cohort: **{len(cdf):,}** presentations, **{cdf[cfg.patient_id_column].nunique():,}** patients")

with st.expander("Current configuration (YAML)", expanded=False):
    st.code(cfg.to_yaml_str(), language="yaml")

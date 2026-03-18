"""Page 2 — Download & Export: cohort summary and download."""

import io
import streamlit as st
import pandas as pd

from core.config import EXPORT_PRESETS
from core.ui_helpers import year_bar_chart_data

st.header("2. Download & Export")

if "cohort_df" not in st.session_state:
    st.warning("Run **1. Define Cohort** first.")
    st.stop()

cfg = st.session_state["config"]
cdf = st.session_state["cohort_df"]

# =====================================================================
# Cohort Summary Statistics
# =====================================================================
st.subheader("Cohort Summary")

m1, m2, m3 = st.columns(3)
m1.metric("Rows", f"{len(cdf):,}")
m2.metric("Columns", f"{len(cdf.columns)}")
m3.metric("Unique patients", f"{cdf[cfg.patient_id_column].nunique():,}")

# Demographics summary
with st.expander("Demographics & Clinical Profile", expanded=True):
    demo_cols = st.columns(3)

    # Age
    if "Admission Age (Year) (episode based)" in cdf.columns:
        age = pd.to_numeric(cdf["Admission Age (Year) (episode based)"], errors="coerce")
        with demo_cols[0]:
            st.markdown("**Age**")
            st.markdown(
                f"- Mean: {age.mean():.1f}\n"
                f"- Median: {age.median():.0f}\n"
                f"- Range: {age.min():.0f}–{age.max():.0f}\n"
                f"- Missing: {age.isna().sum():,}"
            )

    # Sex
    if "Sex" in cdf.columns:
        sex = cdf["Sex"].astype(str).str.strip().str.upper()
        sex_vc = sex.value_counts()
        n_total = len(sex)
        with demo_cols[1]:
            st.markdown("**Sex**")
            for val in ["M", "F"]:
                count = sex_vc.get(val, 0)
                pct = 100 * count / n_total if n_total > 0 else 0
                st.markdown(f"- {val}: {count:,} ({pct:.1f}%)")

    # Triage
    if "Triage Category" in cdf.columns:
        triage = pd.to_numeric(cdf["Triage Category"], errors="coerce")
        with demo_cols[2]:
            st.markdown("**Triage Category**")
            for cat in sorted(triage.dropna().unique()):
                count = int((triage == cat).sum())
                pct = 100 * count / len(triage) if len(triage) > 0 else 0
                st.markdown(f"- Cat {int(cat)}: {count:,} ({pct:.1f}%)")

    # Date range
    if "attendance_date" in cdf.columns:
        date_min = cdf["attendance_date"].min()
        date_max = cdf["attendance_date"].max()
        st.markdown(f"**Date range**: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")

    # UIPOL-03: presentations-per-year bar chart
    year_counts = year_bar_chart_data(cdf)
    if year_counts is not None:
        st.markdown("**Presentations per year**")
        st.bar_chart(year_counts.set_index("Year"), use_container_width=True, height=200)

    # Top diagnosis codes
    diag_col = "Principal Diagnosis Code"
    if diag_col in cdf.columns:
        st.markdown("**Top 10 Principal Diagnosis Codes**")
        top_dx = cdf[diag_col].astype(str).replace({"nan": pd.NA}).dropna().value_counts().head(10)
        dx_df = pd.DataFrame({"Code": top_dx.index, "Count": top_dx.values,
                              "Percent": (100 * top_dx.values / len(cdf)).round(1)})
        st.dataframe(dx_df, use_container_width=True, hide_index=True)

    # Missing data rates for key columns
    st.markdown("**Missing Data Rates (key columns)**")
    key_cols = [
        "Admission Age (Year) (episode based)", "Sex", "Triage Category",
        "GCS Total Score", "Principal Diagnosis Code",
        "Poison Nature Description", "Episode Death (Y/N)",
    ]
    key_cols = [c for c in key_cols if c in cdf.columns]
    if key_cols:
        missing_data = []
        for col in key_cols:
            s = cdf[col]
            str_s = s.astype(str).str.strip().str.lower()
            n_missing = int(s.isna().sum() + str_s.isin(["", "nan", "none"]).sum())
            pct = 100 * n_missing / len(cdf) if len(cdf) > 0 else 0
            missing_data.append({"Column": col, "Missing": f"{n_missing:,}", "Rate": f"{pct:.1f}%"})
        st.dataframe(pd.DataFrame(missing_data), use_container_width=True, hide_index=True)

st.divider()

# =====================================================================
# Column Selection & Download
# =====================================================================
st.subheader("Select Columns for Export")
st.caption(
    "Your cohort has all original CDARS columns. "
    "Select which to include in the download — this does not affect the underlying data."
)

all_cols = list(cdf.columns)

meta_cols = [cfg.patient_id_column, "attendance_date", "year", "presentation_id"]
meta_cols = [c for c in meta_cols if c in all_cols]
clinical_cols = sorted(c for c in all_cols if c not in meta_cols and c != "presentation_id")

# Column preset selector (UIPOL-04)
preset = st.radio(
    "Column preset",
    list(EXPORT_PRESETS.keys()),
    index=list(EXPORT_PRESETS.keys()).index("Full"),  # default to Full
    horizontal=True,
    key="export_preset",
    help="Minimal: IDs and dates only. Demographics: adds age, sex, district. Full: all columns.",
)

# Preset drives default selection; only update when preset changes to preserve user tweaks
_preset_key = f"_preset_cols_{preset}"
if _preset_key not in st.session_state:
    if EXPORT_PRESETS[preset] is None:
        st.session_state[_preset_key] = all_cols
    else:
        st.session_state[_preset_key] = [c for c in EXPORT_PRESETS[preset] if c in all_cols]

default_cols = st.session_state[_preset_key]

selected_cols = st.multiselect(
    "Columns to export",
    options=all_cols,
    default=[c for c in default_cols if c in all_cols],
)

if not selected_cols:
    st.warning("Select at least one column.")
    st.stop()

export_df = cdf[selected_cols]

st.caption(f"Exporting {len(selected_cols)} columns, {len(export_df):,} rows")

# --- Data quality notes ---
st.divider()
st.subheader("Data Quality Notes")
st.caption(
    "The exported cohort preserves original CDARS values. "
    "You may need to handle the following in your analysis code:"
)
st.markdown(
    "- **Missing values**: numeric columns (age, GCS, waiting times) may contain NaN\n"
    "- **String `nan`**: text columns contain the literal string `\"nan\"` for missing values — treat as missing\n"
    "- **Date `NaT`**: unparseable dates (death, discharge) are NaT\n"
    "- **No outlier handling**: values like age 0/120, negative waiting times are kept as-is\n"
    "- **Raw diagnosis codes**: ICD-9 codes are not validated — typos in CDARS pass through"
)

# --- Downloads ---
st.divider()
st.subheader("Download")

config_name = st.session_state.get("config_name", "cohort")

try:
    col1, col2, col3 = st.columns(3)

    with col1:
        parquet_buffer = io.BytesIO()
        export_df.to_parquet(parquet_buffer, index=False)
        size_mb = len(parquet_buffer.getvalue()) / 1e6
        st.download_button(
            label=f"Parquet ({size_mb:.1f} MB)",
            data=parquet_buffer.getvalue(),
            file_name=f"{config_name}_cohort.parquet",
            mime="application/octet-stream",
            use_container_width=True,
        )

    with col2:
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue()
        csv_size_mb = len(csv_bytes.encode()) / 1e6
        st.download_button(
            label=f"CSV ({csv_size_mb:.1f} MB)",
            data=csv_bytes,
            file_name=f"{config_name}_cohort.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col3:
        yaml_str = cfg.to_yaml_str()
        st.download_button(
            label="Config YAML",
            data=yaml_str,
            file_name=f"{config_name}_config.yaml",
            mime="text/yaml",
            use_container_width=True,
        )
except Exception as exc:
    st.error(
        "Export failed. This can happen if the cohort data contains unexpected values "
        "or if the system is low on memory for large exports. "
        "Try downloading a smaller column selection, or re-run the filter."
    )
    with st.expander("Technical details (for reporting)", expanded=False):
        st.code(str(exc))

st.caption("The config YAML records all filter rules — share it with teammates to reproduce this exact cohort.")

# --- Preview ---
st.divider()
st.subheader("Preview")
st.dataframe(export_df.head(50), use_container_width=True, hide_index=True)

"""Page 0 — Upload Data: upload XLS/XLSX files and manage the master dataset."""

import io
import os
import streamlit as st
import pandas as pd

from core.io import PROCESSED_DIR, MASTER_PARQUET, master_exists, master_info
from core.ui_helpers import _execute_confirmed_delete, clear_cohort_state
from core.ingest import (
    parse_single_file_bytes,
    REQUIRED_COLUMNS,
    ANCHOR_COLUMN,
)

st.header("0. Upload Data")


# ---------------------------------------------------------------------------
# Audit Trail helper
# ---------------------------------------------------------------------------
class AuditTrail:
    """Accumulates structured step records during data processing."""

    def __init__(self):
        self.steps: list[dict] = []

    def add_step(
        self,
        name: str,
        description: str,
        rows_before: int | None = None,
        rows_after: int | None = None,
        details: dict | None = None,
    ):
        self.steps.append(
            {
                "name": name,
                "description": description,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "details": details,
            }
        )


def _read_raw(uploaded_file):
    """Read an uploaded file into a raw DataFrame. Returns (df, engine) or (None, error_str)."""
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()

    raw_df = None
    parse_engine = None
    try:
        if ext == ".xlsx":
            parse_engine = "openpyxl"
            raw_df = pd.read_excel(uploaded_file, engine="openpyxl", header=None)
        else:
            parse_engine = "xlrd"
            raw_df = pd.read_excel(uploaded_file, engine="xlrd", header=None)
    except Exception as e:
        error_msg = str(e)
        if "Expected BOF record" in error_msg or "b'<meta" in error_msg:
            try:
                uploaded_file.seek(0)
                dfs = pd.read_html(uploaded_file, header=None)
                for temp_df in dfs:
                    if (temp_df.astype(str)
                        .apply(lambda x: x.str.contains(ANCHOR_COLUMN, na=False))
                        .any().any()):
                        raw_df = temp_df
                        parse_engine = "html-fallback"
                        break
            except Exception:
                pass

    if raw_df is None:
        return None, "Could not read file (unsupported format)"
    return raw_df, parse_engine


def _scan_file(uploaded_file):
    """Quick-scan a file: read it, find header, report stats. Returns a scan dict."""
    name = uploaded_file.name
    result = {"name": name, "error": None, "rows": 0, "columns_found": 0,
              "missing_columns": [], "status": "ok"}

    raw_df, engine_or_err = _read_raw(uploaded_file)
    if raw_df is None:
        result["error"] = engine_or_err
        result["status"] = "error"
        return result

    # Find header row
    matches = raw_df.isin([ANCHOR_COLUMN])
    if not matches.any().any():
        result["error"] = f"No '{ANCHOR_COLUMN}' column found"
        result["status"] = "error"
        return result

    header_idx = matches.stack().idxmax()[0]
    file_header = raw_df.iloc[header_idx].astype(str).str.strip().tolist()

    # Count data rows (rough — before cleaning)
    data_rows = len(raw_df) - header_idx - 1
    result["rows"] = max(data_rows, 0)
    result["columns_found"] = len(set(file_header) - {"nan", ""})

    # Check required columns
    missing = sorted(set(REQUIRED_COLUMNS) - set(file_header))
    result["missing_columns"] = missing
    if missing:
        result["status"] = "warning"

    return result


def _parse_uploaded_file(uploaded_file, audit: AuditTrail | None = None):
    """Delegate to core parse function. Returns (df, error_str)."""
    fname = getattr(uploaded_file, "name", str(uploaded_file))
    return parse_single_file_bytes(uploaded_file, fname, audit=audit)


# --- Current data status ---
st.subheader("Current Dataset")

if master_exists():
    info = master_info()
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", f"{info['rows']:,}")
    m2.metric("Columns", f"{info['columns']}")
    m3.metric("Size", f"{info['size_mb']} MB")

    # Show date range and patient count (cached to avoid re-reading 19M rows)
    @st.cache_data(show_spinner=False)
    def _dataset_summary(mtime: float):
        """Compute date range and patient count. mtime busts cache on file change."""
        cols = ["Attendance Date (yyyy-mm-dd)", "Reference Key"]
        sample = pd.read_parquet(MASTER_PARQUET, columns=cols)
        return {
            "date_min": sample["Attendance Date (yyyy-mm-dd)"].min(),
            "date_max": sample["Attendance Date (yyyy-mm-dd)"].max(),
            "n_patients": int(sample["Reference Key"].nunique()),
        }

    try:
        mtime = MASTER_PARQUET.stat().st_mtime
        summary = _dataset_summary(mtime)
        m4, m5 = st.columns(2)
        m4.metric("Date range", f"{summary['date_min'].strftime('%Y-%m-%d')} to {summary['date_max'].strftime('%Y-%m-%d')}")
        m5.metric("Unique patients", f"{summary['n_patients']:,}")
    except Exception:
        pass

    # Step 1 — trigger the confirmation flow
    if st.button("Delete all data", type="primary", key="btn_delete_trigger"):
        st.session_state["confirm_delete_pending"] = True
        st.rerun()

    # Step 2 — show confirmation UI if pending
    if st.session_state.get("confirm_delete_pending"):
        st.warning(
            "**This will permanently delete the master dataset.** "
            "You will need to re-upload all original XLS/XLSX files to rebuild it. "
            "Download a backup first if you want to keep a copy."
        )

        # Cache backup bytes once — avoid re-reading 98 MB on every rerun
        if "delete_backup_bytes" not in st.session_state:
            with st.spinner("Preparing backup..."):
                buf = io.BytesIO()
                pd.read_parquet(MASTER_PARQUET).to_parquet(buf, index=False)
                st.session_state["delete_backup_bytes"] = buf.getvalue()

        st.download_button(
            "Download backup before deleting",
            data=st.session_state["delete_backup_bytes"],
            file_name="all_admissions_backup.parquet",
            mime="application/octet-stream",
            key="btn_backup_download",
        )

        confirm_text = st.text_input(
            "Type DELETE to confirm",
            key="delete_confirm_text",
            placeholder="DELETE",
        )
        col1, col2 = st.columns(2)
        if col1.button(
            "Confirm delete",
            type="primary",
            key="btn_delete_confirm",
            disabled=(confirm_text != "DELETE"),
        ):
            if _execute_confirmed_delete(st.session_state, confirm_text, MASTER_PARQUET):
                st.rerun()
        if col2.button("Cancel", key="btn_delete_cancel"):
            for key in ["confirm_delete_pending", "delete_backup_bytes"]:
                st.session_state.pop(key, None)
            st.rerun()
else:
    st.info("No data loaded yet. Upload CDARS XLS/XLSX files below to get started.")

st.divider()

# --- Upload ---
st.subheader("Upload CDARS Files")
st.caption("Drag and drop XLS or XLSX files. After scanning you can choose which files to include.")

# Initialise session state
if "uploaded_file_stash" not in st.session_state:
    st.session_state["uploaded_file_stash"] = {}   # name -> bytes
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = None        # list[dict] from _scan_file
if "file_selection" not in st.session_state:
    st.session_state["file_selection"] = {}        # name -> bool

uploaded_files = st.file_uploader(
    "Upload XLS/XLSX files",
    type=["xls", "xlsx"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

# Stash any newly uploaded files
if uploaded_files:
    new_files = False
    for uf in uploaded_files:
        if uf.name not in st.session_state["uploaded_file_stash"]:
            st.session_state["uploaded_file_stash"][uf.name] = uf.getvalue()
            new_files = True
    # Clear old scan when new files are added
    if new_files:
        st.session_state["scan_results"] = None

stash = st.session_state["uploaded_file_stash"]

# ---- Step 1: Scan ----
if stash and st.session_state["scan_results"] is None:
    if st.button("Scan files", type="primary", use_container_width=True):
        scans = []
        progress = st.progress(0, text="Scanning files...")
        stash_names = sorted(stash.keys())
        for i, fname in enumerate(stash_names):
            progress.progress((i + 1) / len(stash_names), text=f"Scanning {fname}...")
            buf = io.BytesIO(stash[fname])
            buf.name = fname
            scans.append(_scan_file(buf))
        progress.empty()
        st.session_state["scan_results"] = scans
        # Default: select all non-error files
        st.session_state["file_selection"] = {
            s["name"]: s["status"] != "error" for s in scans
        }
        st.rerun()

# ---- Step 2: Review & select ----
scans = st.session_state["scan_results"]
if scans is not None:
    st.subheader("Review Files")
    st.caption("Tick the files you want to include. Files with warnings are missing some columns but can still be loaded.")

    selection = st.session_state["file_selection"]

    for scan in scans:
        fname = scan["name"]
        disabled = scan["status"] == "error"

        # Status icon
        if scan["status"] == "error":
            icon = "~~"
            tag = f"Error: {scan['error']}"
        elif scan["status"] == "warning":
            icon = "~~"
            tag = f"{len(scan['missing_columns'])} missing columns"
        else:
            tag = ""
            icon = ""

        cols = st.columns([0.5, 3, 1.5, 1, 2])

        checked = cols[0].checkbox(
            "sel", value=selection.get(fname, False),
            key=f"sel_{fname}", label_visibility="collapsed",
            disabled=disabled,
        )
        selection[fname] = checked

        cols[1].text(fname)
        cols[2].text(f"{scan['rows']:,} rows" if scan["rows"] else "—")
        cols[3].text(f"{scan['columns_found']} cols" if scan["columns_found"] else "—")

        if scan["status"] == "error":
            cols[4].caption(f":red[{scan['error']}]")
        elif scan["status"] == "warning":
            cols[4].caption(f":orange[{tag}]")
        else:
            cols[4].caption(":green[Ready]")

        # Expandable detail for warnings
        if scan["missing_columns"]:
            with st.expander(f"Missing columns in {fname}", expanded=False):
                for mc in scan["missing_columns"]:
                    st.text(f"  • {mc}")

    st.session_state["file_selection"] = selection

    # Quick select/deselect
    sel_cols = st.columns(3)
    if sel_cols[0].button("Select all"):
        for s in scans:
            if s["status"] != "error":
                selection[s["name"]] = True
        st.rerun()
    if sel_cols[1].button("Deselect all"):
        for s in scans:
            selection[s["name"]] = False
        st.rerun()
    if sel_cols[2].button("Clear uploads", type="secondary"):
        st.session_state["uploaded_file_stash"] = {}
        st.session_state["scan_results"] = None
        st.session_state["file_selection"] = {}
        st.rerun()

    # Count selected
    selected_names = [n for n, v in selection.items() if v]
    n_selected = len(selected_names)

    st.divider()

    # ---- Step 3: Process selected ----
    if n_selected == 0:
        st.info("Select at least one file to process.")
    elif st.button(
        f"Process {n_selected} selected file(s)",
        type="primary", use_container_width=True,
    ):
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        progress = st.progress(0, text="Parsing files...")
        parsed_chunks = []
        errors = []
        file_audits: list[tuple[str, AuditTrail]] = []

        for i, fname in enumerate(selected_names):
            progress.progress((i + 1) / len(selected_names), text=f"Parsing {fname}...")
            audit = AuditTrail()
            buf = io.BytesIO(stash[fname])
            buf.name = fname
            df, err = _parse_uploaded_file(buf, audit=audit)
            if df is not None:
                parsed_chunks.append(df)
                file_audits.append((fname, audit))
            else:
                errors.append((fname, err))

        if errors:
            for name, err in errors:
                st.warning(f"Skipped **{name}**: {err}")

        if not parsed_chunks:
            st.error("No files could be parsed.")
            st.stop()

        progress.progress(0.8, text="Deduplicating...")

        try:
            global_audit = AuditTrail()

            new_data = pd.concat(parsed_chunks, ignore_index=True)
            global_audit.add_step(
                "Concatenation",
                f"Combined {len(parsed_chunks)} parsed files",
                rows_after=len(new_data),
            )

            # Merge with existing data if present
            if master_exists():
                existing = pd.read_parquet(MASTER_PARQUET)
                rows_existing = len(existing)
                combined = pd.concat([existing, new_data], ignore_index=True)
                global_audit.add_step(
                    "Merge with existing",
                    f"Merged {len(new_data):,} new rows with {rows_existing:,} existing rows",
                    rows_before=rows_existing,
                    rows_after=len(combined),
                )
            else:
                combined = new_data
                global_audit.add_step(
                    "New dataset",
                    "No existing data — starting fresh",
                    rows_after=len(combined),
                )

            # Deduplicate
            before = len(combined)
            dedup_cols = ["Reference Key", "AE Number", "Attendance Date (yyyy-mm-dd)"]
            available_dedup = [c for c in dedup_cols if c in combined.columns]
            combined = combined.drop_duplicates(subset=available_dedup)
            dupes_removed = before - len(combined)
            n_patients_after = int(combined["Reference Key"].nunique()) if "Reference Key" in combined.columns else None
            global_audit.add_step(
                "Deduplication",
                f"Removed {dupes_removed:,} exact duplicate rows on ({', '.join(available_dedup)}). "
                "Duplicates arise when uploaded files cover overlapping date ranges.",
                rows_before=before,
                rows_after=len(combined),
                details={"unique_patients": n_patients_after} if n_patients_after else None,
            )

            # Infer Meta_Year from attendance date if available
            if "Meta_Year" not in combined.columns and "Attendance Date (yyyy-mm-dd)" in combined.columns:
                combined["Meta_Year"] = combined["Attendance Date (yyyy-mm-dd)"].dt.year
                global_audit.add_step(
                    "Meta_Year inference",
                    "Added Meta_Year column from Attendance Date",
                )

            progress.progress(0.95, text="Saving...")

            combined.to_parquet(MASTER_PARQUET, index=False)
            file_size_mb = round(MASTER_PARQUET.stat().st_size / (1024 * 1024), 1)
            n_patients_final = int(combined["Reference Key"].nunique()) if "Reference Key" in combined.columns else None
            global_audit.add_step(
                "Save to parquet",
                f"Saved to {MASTER_PARQUET.name} ({file_size_mb} MB)",
                rows_after=len(combined),
                details={"unique_patients": n_patients_final, "columns": len(combined.columns)} if n_patients_final else None,
            )

            # Clear cached data
            clear_cohort_state(st.session_state)
            st.session_state["data_updated_since_last_run"] = True  # SESS-04

            progress.progress(1.0, text="Done!")

            new_rows = len(new_data)
            st.success(
                f"Added **{new_rows:,}** rows from **{len(parsed_chunks)}** files. "
                f"Removed **{dupes_removed:,}** duplicates. "
                f"Total dataset: **{len(combined):,}** rows."
            )

            # Clear file stash & scan after successful processing
            st.session_state["uploaded_file_stash"] = {}
            st.session_state["scan_results"] = None
            st.session_state["file_selection"] = {}

            # Store audit trail
            st.session_state["audit_trail"] = {
                "files": [(fname, fa.steps) for fname, fa in file_audits],
                "global": global_audit.steps,
            }
            st.rerun()
        except Exception as exc:
            st.error(
                "Something went wrong while saving your data. "
                "No data was written to disk. "
                "Please check your files and try again, or report the error below."
            )
            with st.expander("Technical details (for reporting)", expanded=False):
                st.code(str(exc))


# --- Persistent data cleaning summary ---
if master_exists():
    st.divider()
    st.subheader("Data Cleaning Summary")
    st.caption(
        "The following automatic cleaning steps are applied during ingestion. "
        "These remove invalid/duplicate records only — **no clinical or demographic filtering is applied here**. "
        "All cohort-level filtering happens on the Filter page."
    )

    # Extract counts from last audit trail if available
    _trail = st.session_state.get("audit_trail")
    _empty_rows_dropped = None
    _empty_before = None
    _empty_after = None
    _dedup_dropped = None
    _dedup_before = None
    _dedup_after = None
    _dedup_patients = None
    _final_rows = None
    _final_patients = None

    if _trail:
        # Per-file: sum up empty row removals across all files
        _empty_rows_dropped = 0
        _empty_before = 0
        _empty_after = 0
        for _fname, _steps in _trail["files"]:
            for _s in _steps:
                if _s["name"] == "Empty row removal":
                    if _s["rows_before"] is not None:
                        _empty_before += _s["rows_before"]
                    if _s["rows_after"] is not None:
                        _empty_after += _s["rows_after"]
                    if _s["rows_before"] is not None and _s["rows_after"] is not None:
                        _empty_rows_dropped += _s["rows_before"] - _s["rows_after"]

        # Global steps
        for _s in _trail["global"]:
            if _s["name"] == "Deduplication":
                _dedup_before = _s.get("rows_before")
                _dedup_after = _s.get("rows_after")
                if _dedup_before is not None and _dedup_after is not None:
                    _dedup_dropped = _dedup_before - _dedup_after
                if _s.get("details"):
                    _dedup_patients = _s["details"].get("unique_patients")
            if _s["name"] == "Save to parquet":
                _final_rows = _s.get("rows_after")
                if _s.get("details"):
                    _final_patients = _s["details"].get("unique_patients")

    _no_trail_msg = "Upload and process files to see row counts"

    with st.container(border=True):
        st.markdown("**1. Empty row removal**")
        st.markdown(
            "Rows where Reference Key is null or blank are dropped. "
            "These are padding or footer rows from the XLS spreadsheet — not real patient records."
        )
        if _empty_before is not None:
            st.markdown(
                f"  {_empty_before:,} → {_empty_after:,} rows "
                f"(**{_empty_rows_dropped:,}** removed)"
            )
        else:
            st.caption(_no_trail_msg)

    with st.container(border=True):
        st.markdown("**2. Deduplication**")
        st.markdown(
            "Exact duplicate rows on (Reference Key, AE Number, Attendance Date) are removed. "
            "Duplicates arise when uploaded files cover overlapping date ranges."
        )
        if _dedup_before is not None:
            _dedup_detail = (
                f"  {_dedup_before:,} → {_dedup_after:,} rows "
                f"(**{_dedup_dropped:,}** removed)"
            )
            if _dedup_patients is not None:
                _dedup_detail += f" — **{_dedup_patients:,}** unique patients"
            st.markdown(_dedup_detail)
        else:
            st.caption(_no_trail_msg)

    with st.container(border=True):
        st.markdown("**3. Type coercion**")
        st.markdown(
            "Non-numeric values in numeric fields (e.g. age, GCS) are converted to NaN. "
            "The row is preserved — only the invalid cell value is marked as missing. **No rows are dropped.**"
        )

    if _final_rows is not None and _final_patients is not None:
        st.success(f"Final dataset: **{_final_rows:,}** rows, **{_final_patients:,}** unique patients, **{info['columns']}** columns")

    st.caption("No rows are removed for clinical reasons (age, residency, diagnosis, etc.) at this stage.")


# --- Render audit trail if available ---
def _render_audit_step(step: dict):
    """Render a single audit step as styled markdown."""
    row_info = ""
    if step["rows_before"] is not None and step["rows_after"] is not None:
        diff = step["rows_after"] - step["rows_before"]
        diff_str = f" ({diff:+,})" if diff != 0 else ""
        row_info = f" — {step['rows_before']:,} → {step['rows_after']:,} rows{diff_str}"
    elif step["rows_after"] is not None:
        row_info = f" — {step['rows_after']:,} rows"
    st.markdown(f"- **{step['name']}**: {step['description']}{row_info}")
    if step.get("details"):
        for detail_key, detail_val in step["details"].items():
            if isinstance(detail_val, (int, float)):
                st.caption(f"  {detail_key}: {detail_val:,}")
            elif isinstance(detail_val, dict) and detail_val:
                items = ", ".join(f"{k}: {v}" for k, v in detail_val.items())
                st.caption(f"  {detail_key}: {items}")
            elif isinstance(detail_val, list) and detail_val:
                st.caption(f"  {detail_key}: {', '.join(str(v) for v in detail_val)}")


if "audit_trail" in st.session_state:
    trail = st.session_state["audit_trail"]
    with st.expander("Last Upload — Detailed Audit Trail", expanded=True):
        # Per-file details
        for fname, steps in trail["files"]:
            st.markdown(f"#### {fname}")
            for step in steps:
                _render_audit_step(step)

        # Global steps
        st.markdown("#### Combined Processing")
        for step in trail["global"]:
            _render_audit_step(step)

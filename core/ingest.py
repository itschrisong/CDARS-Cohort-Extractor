"""XLS → parquet ingestion.

Near-verbatim port of self-harm_model_decay/src/python/00_raw_to_parquet.py
with parameterised paths and optional progress callback.
"""

import glob
import io
import os
import warnings

import pandas as pd

from core.normalize import normalize_ws_series as _normalize_ws_series, normalize_ws_scalar as _normalize_ws_scalar

ANCHOR_COLUMN = "Reference Key"

REQUIRED_COLUMNS = [
    "Reference Key",
    "Institution (IPAS)",
    "AE Number",
    "Sex",
    "Admission Age (Year) (episode based)",
    "Paycode (at discharge)",
    "District of Residence (system code)",
    "District of Residence Description",
    "Race Description",
    "Date of Registered Death",
    "Exact date of death",
    "Death Cause (Main Cause)",
    "Death Cause (Supplementary Cause)",
    "Attendance Date (yyyy-mm-dd)",
    "Attendance Date (Hour)",
    "Admission from OAH (Y/N)",
    "Ambulance Case (Y/N)",
    "Attendance Specialty (EIS)",
    "Discharge Date (yyyy-mm-dd)",
    "Discharge Hour (00-23)",
    "Traumatic Type",
    "Domestic Type",
    "Domestic Nature",
    "Animal Bite Type Description",
    "Poison Nature Description",
    "Poison Type Description",
    "Poison Description",
    "Triage Category",
    "Mobility Status",
    "Conscious Level",
    "GCS Total Score",
    "Waiting Time (to cubicle)(Minute)",
    "Waiting Time (to triage)(Minute)",
    "Consultation Start Time (Hour, 00-23)",
    "Observation Room Case (Y/N)",
    "Observation Room Staying Time (Minute)",
    "Trauma Team Activation (Y/N)",
    "Episode Death (Y/N)",
    "Total Staying Time (Minute)",
    "Discharge Status (EIS)",
    "Discharge Destination (AEIS)",
    "A&E to IP Ward: Admission Decision Time (yyyy-mm-dd HH:MM)",
    "A&E to IP Ward: Waiting Time for Admission (Min)",
    "A&E to IP Ward: Length of Stay",
    "A&E to IP Ward: Admission Date",
    "A&E to IP Ward: Principal Diagnosis Code",
    "A&E to IP Ward: Diagnosis (rank 2)",
    "A&E to IP Ward: Diagnosis (rank 3)",
    "A&E to IP Ward: Diagnosis (rank 4)",
    "A&E to IP Ward: Diagnosis (rank 5)",
    "A&E to IP Ward: Principal Diagnosis Description (HAMDCT)",
    "A&E to IP Ward: Diagnosis HAMDCT Description (rank 2)",
    "A&E to IP Ward: Diagnosis HAMDCT Description (rank 3)",
    "A&E to IP Ward: Diagnosis HAMDCT Description (rank 4)",
    "A&E to IP Ward: Diagnosis HAMDCT Description (rank 5)",
    "Principal Diagnosis Code",
    "Diagnosis (rank 2)",
    "Diagnosis (rank 3)",
    "Diagnosis (rank 4)",
    "Diagnosis (rank 5)",
    "Principal Diagnosis Description (HAMDCT)",
    "Diagnosis HAMDCT Description (rank 2)",
    "Diagnosis HAMDCT Description (rank 3)",
    "Diagnosis HAMDCT Description (rank 4)",
    "Diagnosis HAMDCT Description (rank 5)",
    "A&E to IP Ward: Institution",
    "A&E to IP Ward: HN Number",
    "A&E to IP Ward: Admission Specialty (IPAS)",
]

NUMERIC_COLUMNS = [
    "Admission Age (Year) (episode based)",
    "Attendance Date (Hour)",
    "Discharge Hour (00-23)",
    "Triage Category",
    "GCS Total Score",
    "Waiting Time (to cubicle)(Minute)",
    "Waiting Time (to triage)(Minute)",
    "Consultation Start Time (Hour, 00-23)",
    "Observation Room Staying Time (Minute)",
    "Total Staying Time (Minute)",
    "A&E to IP Ward: Waiting Time for Admission (Min)",
    "A&E to IP Ward: Length of Stay",
]

DATE_COLUMNS = [
    "Attendance Date (yyyy-mm-dd)",
    "Discharge Date (yyyy-mm-dd)",
    "Date of Registered Death",
    "Exact date of death",
    "A&E to IP Ward: Admission Date",
]

# Identifier columns that must NOT be normalized
_STRING_ID_COLUMNS = frozenset([
    "Reference Key",
    "AE Number",
    "Institution (IPAS)",
    "A&E to IP Ward: Institution",
    "A&E to IP Ward: HN Number",
])


# ---------------------------------------------------------------------------
# Core parse function (shared by CLI and upload page)
# ---------------------------------------------------------------------------

def parse_single_file_bytes(buf, name: str, audit=None):
    """Parse one XLS/XLSX BytesIO buffer into a cleaned DataFrame.

    Lenient: keeps available columns if some REQUIRED_COLUMNS are absent.
    Returns (df, None) on success; (None, error_str) on failure.

    Args:
        buf: io.BytesIO of the file contents
        name: filename with extension (e.g. "cdars_2023.xlsx")
        audit: optional audit trail object (passed through to metadata)
    """
    ext = os.path.splitext(name)[1].lower()

    # --- Step 1: Raw read with format detection and HTML fallback ---
    raw_df = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if ext == ".xlsx":
                raw_df = pd.read_excel(buf, engine="openpyxl", header=None)
            else:
                raw_df = pd.read_excel(buf, engine="xlrd", header=None)
    except Exception as e:
        error_msg = str(e)
        if "Expected BOF record" in error_msg or "b'<meta" in error_msg or "Unsupported format" in error_msg:
            try:
                buf.seek(0)
                dfs = pd.read_html(buf, header=None)
                for temp_df in dfs:
                    if (
                        temp_df.astype(str)
                        .apply(lambda x: x.str.contains(ANCHOR_COLUMN, na=False))
                        .any()
                        .any()
                    ):
                        raw_df = temp_df
                        break
            except Exception:
                pass
        if raw_df is None:
            return None, f"Could not read '{name}': {error_msg}"

    if raw_df is None:
        return None, f"Could not read '{name}': no parseable content found"

    # --- Step 2: Header row detection via ANCHOR_COLUMN ---
    header_row = None
    for idx, row in raw_df.iterrows():
        if any(str(cell) == ANCHOR_COLUMN for cell in row):
            header_row = idx
            break
    if header_row is None:
        return None, f"'{name}': header row with '{ANCHOR_COLUMN}' not found"

    df = raw_df.iloc[header_row + 1:].copy()
    df.columns = [str(c) for c in raw_df.iloc[header_row]]
    df = df.reset_index(drop=True)

    # --- Step 3: Column validation (lenient — keep what's available) ---
    available = [c for c in REQUIRED_COLUMNS if c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        warnings.warn(
            f"'{name}': {len(missing)} REQUIRED_COLUMNS absent — "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
            stacklevel=2,
        )
    df = df[available] if available else df

    # --- Step 4: Remove empty rows ---
    df = df.dropna(how="all").reset_index(drop=True)

    # --- Step 5: Reference Key whitespace clean ---
    if ANCHOR_COLUMN in df.columns:
        df[ANCHOR_COLUMN] = df[ANCHOR_COLUMN].astype(str).str.strip()

    # --- Step 6: Numeric casting ---
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Step 7: Date casting ---
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- Step 8: String casting + Unicode normalization (non-ID columns only) ---
    str_cols = [
        c for c in df.columns
        if c not in NUMERIC_COLUMNS
        and c not in DATE_COLUMNS
        and c not in _STRING_ID_COLUMNS
    ]
    for col in str_cols:
        df[col] = _normalize_ws_series(df[col].astype(str))

    # --- Step 9: Metadata columns ---
    if audit is not None and hasattr(audit, "source_file"):
        df["Meta_Source_File"] = audit.source_file
    else:
        df["Meta_Source_File"] = name

    att_col = "Attendance Date (yyyy-mm-dd)"
    if att_col in df.columns:
        df["Meta_Year"] = pd.to_datetime(df[att_col], errors="coerce").dt.year
    df["Meta_Week"] = (
        pd.to_datetime(df.get(att_col), errors="coerce").dt.isocalendar().week
        if att_col in df.columns
        else pd.Series([pd.NA] * len(df), dtype="Int32")
    )

    return df, None


def parse_single_file(file_path, year_label):
    """Parse one xls/xlsx file. Returns a DataFrame or None. (CLI path — strict)"""
    try:
        with open(file_path, "rb") as f:
            buf = io.BytesIO(f.read())
        name = os.path.basename(file_path)
        df, err = parse_single_file_bytes(buf, name)
        if err is not None:
            print(f"  [ERROR] Could not read: {name} — {err}")
            return None
        # CLI strict check: reject if critical columns are missing
        missing_critical = [c for c in REQUIRED_COLUMNS[:10] if c not in df.columns]
        if missing_critical:
            print(f"  [SKIP] Missing critical cols in: {name}")
            return None
        # Attach year_label as Meta_Year override (CLI-specific behavior)
        df["Meta_Year"] = year_label
        return df
    except Exception as e:
        print(f"  [ERROR] Exception reading {os.path.basename(file_path)}: {e}")
        return None


def process_year(year_folder, year_label, progress_cb=None):
    """Process all files in one year folder → one DataFrame."""
    files = glob.glob(os.path.join(year_folder, "*.xls")) + glob.glob(
        os.path.join(year_folder, "*.xlsx")
    )

    print(f"\n{'=' * 60}")
    print(f"Processing {year_label}: {len(files)} files in {year_folder}")
    print(f"{'=' * 60}")

    chunks = []
    for i, f in enumerate(sorted(files)):
        if i % 10 == 0:
            print(f"  File {i}/{len(files)}: {os.path.basename(f)}")
        if progress_cb:
            progress_cb(i / max(len(files), 1))
        result = parse_single_file(f, year_label)
        if result is not None:
            chunks.append(result)

    if not chunks:
        print(f"  [WARNING] No data for {year_label}")
        return None

    df = pd.concat(chunks, ignore_index=True)

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(
        subset=["Reference Key", "AE Number", "Attendance Date (yyyy-mm-dd)"]
    )
    after = len(df)
    if before != after:
        print(f"  Removed {before - after} duplicates")

    print(
        f"  Result: {len(df):,} rows, {df['Reference Key'].nunique():,} unique patients"
    )
    return df


def run_ingestion(raw_dir, output_dir, progress_cb=None):
    """Run full XLS→parquet ingestion.

    Parameters
    ----------
    raw_dir : str or Path
    output_dir : str or Path
    progress_cb : callable(pct), optional
    """
    raw_dir = str(raw_dir)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    year_folders = sorted(glob.glob(os.path.join(raw_dir, "*_raw")))
    if not year_folders:
        print(
            "No year folders found. Expected: data/raw/2015_raw, data/raw/2016_raw, etc."
        )
        return

    print(f"Found {len(year_folders)} year folders")
    all_years = []

    for fi, folder in enumerate(year_folders):
        year_label = os.path.basename(folder).replace("_raw", "")
        df = process_year(folder, year_label, progress_cb=progress_cb)
        if df is not None:
            out_path = os.path.join(output_dir, f"{year_label}.parquet")
            df.to_parquet(out_path, index=False)
            print(f"  Saved: {out_path}")
            all_years.append(df)

        if progress_cb:
            progress_cb((fi + 1) / len(year_folders))

    if all_years:
        print(f"\n{'=' * 60}")
        print("Merging all years...")
        master = pd.concat(all_years, ignore_index=True)
        master_path = os.path.join(output_dir, "all_admissions.parquet")
        master.to_parquet(master_path, index=False)
        print(f"DONE: {len(master):,} total rows across {len(all_years)} years")
        print(f"Unique patients: {master['Reference Key'].nunique():,}")
        print(f"Saved: {master_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest CDARS XLS files to parquet")
    parser.add_argument(
        "--raw-dir", default="data/raw", help="Directory containing YYYY_raw folders"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for parquet files",
    )
    args = parser.parse_args()
    run_ingestion(args.raw_dir, args.output_dir)

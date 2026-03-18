# CDARS Cohort Extractor

A local-only Streamlit tool for extracting clinical cohorts from CDARS A&E data. Upload XLS/XLSX files from CDARS, define filter rules, and export as CSV or Parquet.

**Data never leaves your machine.** Everything runs locally in your browser — no server, no cloud.

---

## For Teammates: Setup from Scratch

Follow these steps exactly. You need either **Docker** (easiest) or **Python 3.10+** installed.

### Option A: Docker (recommended — nothing else to install)

1. **Install Docker Desktop** if you don't have it:
   - Mac: https://docs.docker.com/desktop/install/mac-install/
   - Windows: https://docs.docker.com/desktop/install/windows-install/
   - Open Docker Desktop and make sure it's running (whale icon in your menu bar / taskbar)

2. **Get the code** — open Terminal (Mac) or PowerShell (Windows):
   ```bash
   git clone <repo-url>
   cd CDARSDataWrangler
   ```

3. **Start the app** (first time takes ~2 minutes to build):
   ```bash
   docker compose up --build
   ```
   Wait until you see `You can now view your Streamlit app in your browser`.

4. **Open your browser** and go to:
   ```
   http://localhost:8501
   ```

5. **When you're done**, press `Ctrl+C` in the terminal to stop the app. Your uploaded data and configs are saved — next time just run:
   ```bash
   docker compose up
   ```

### Option B: Python (no Docker)

1. **Check your Python version**:
   ```bash
   python3 --version
   ```
   You need 3.10 or higher. If not, install from https://www.python.org/downloads/

2. **Get the code**:
   ```bash
   git clone <repo-url>
   cd CDARSDataWrangler
   ```

3. **Create a virtual environment and install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # macOS/Linux
   # .venv\Scripts\activate         # Windows PowerShell
   pip install -r requirements.txt
   ```

4. **Launch the app**:
   ```bash
   python3 -m streamlit run Intro.py
   ```

5. **Open your browser** and go to:
   ```
   http://localhost:8501
   ```

6. **Next time**, activate the environment first then run:
   ```bash
   source .venv/bin/activate        # macOS/Linux
   # .venv\Scripts\activate         # Windows PowerShell
   python3 -m streamlit run Intro.py
   ```

---

## Using the App: Step by Step

### Step 1 — Upload your CDARS data

1. Click **0. Upload Data** in the left sidebar
2. Drag your `.xls` or `.xlsx` files from CDARS into the upload box (you can drop multiple files at once)
3. Click **Scan files** — the tool checks each file has the right columns
4. Review the results: green = ready, orange = some columns missing (still usable), red = can't read the file
5. Tick the files you want, then click **Process selected file(s)**
6. Wait for parsing to finish — you'll see a success message with the total row count

**You only need to do this once.** The data is cached locally. You can upload more files later to add data incrementally.

### Step 2 — Define your cohort

1. Click **1. Define Cohort** in the sidebar
2. **If someone shared a config YAML with you**: select it from the sidebar dropdown and click **Load**, then skip to step 2e
3. Under **Inclusion Rules**, add filters for the rows you want to keep:
   - Pick a category (e.g. Diagnosis, Poison, Patient)
   - Pick a column
   - Select the values you want, or switch to Advanced mode for regex/ICD-9 ranges
   - Click **Add filter**
4. Under **Exclusion Rules**, add filters for rows to remove (e.g. age < 18, non-residents)
5. Click **Run Filter** at the bottom
6. Review the **STROBE Flow** table — it shows exactly how many rows each step removed
7. Check the **Quick Summary** (total presentations, patients, mean age) to sanity-check

### Step 3 — Export your cohort

1. Click **2. Download & Export** in the sidebar
2. Review the summary statistics (demographics, top diagnoses, missing data rates)
3. Choose a column preset: **Minimal** (IDs only), **Demographics** (+ age/sex), or **Full** (everything)
4. Click one of the download buttons:
   - **Parquet** — best for Python/R analysis (small, fast)
   - **CSV** — opens in Excel
   - **Config YAML** — save this to share your exact filter rules with teammates

### Step 4 — Share your cohort definition

To let a teammate reproduce your exact cohort:

1. Download the **Config YAML** from the Export page
2. Send them the `.yaml` file (email, Slack, Teams, etc.)
3. They put it in their `configs/` folder
4. They load it from the sidebar dropdown, click **Run Filter**, and get the same cohort

The YAML file is small (< 1 KB) and contains only the rules — no patient data.

---

## Step-by-Step Guide

The app has three pages, used in order:

### Page 0: Upload Data

This is where you load your raw CDARS files into the tool.

1. Navigate to **0. Upload Data** in the sidebar
2. Drag and drop one or more `.xls` or `.xlsx` files from CDARS into the upload area
3. Click **Scan files** — the tool checks each file for the expected 68 CDARS columns and reports any issues
4. Review the scan results:
   - **Green (Ready)**: file is good to go
   - **Orange (Warning)**: some columns are missing — the file can still be loaded, but those columns will be empty
   - **Red (Error)**: file could not be parsed (wrong format, corrupted, etc.)
5. Tick the files you want to include, then click **Process selected file(s)**

The tool will:
- Find the header row automatically (looks for the "Reference Key" column)
- Cast numeric columns (age, GCS, waiting times) and date columns to proper types
- Remove empty/padding rows
- Deduplicate on (Reference Key, AE Number, Attendance Date)
- Save everything to a local Parquet cache for fast reloading

**You can upload more files later** — new data is merged with existing data and deduplicated again. If your CDARS exports overlap in date range, duplicates are handled automatically.

The **Data Cleaning Summary** section shows exactly what was removed and why (empty rows, duplicates, type coercions). No clinical filtering happens here.

To start fresh, click **Delete all data** (you'll be prompted to download a backup first).

---

### Page 1: Define Cohort

This is where you define which rows to keep (inclusion) and which to remove (exclusion).

#### Inclusion Rules

Inclusion rules define your cohort — only rows matching these criteria are kept. You have two modes:

**Pick Values mode** (default):
1. Select a category (e.g. Poison, Diagnosis, Patient)
2. Select a column within that category
3. Browse the actual values in your data — each value shows how many rows contain it
4. Tick the values you want to include
5. Click **Add filter**
6. Add more filters if needed — choose AND (must match all) or OR (must match any)

**Advanced Rules mode** (for power users):
1. Select a column or a preset group (marked with the hamburger icon) that searches across multiple columns at once
2. Choose an operator:

| Operator | What it does | Example |
|----------|-------------|---------|
| `starts_with` | Value begins with... | `E95` matches E950, E959, etc. |
| `contains` | Value contains substring | `attempt suicide` |
| `equals` | Exact match | `M` |
| `not_equals` | Does not equal | `F` |
| `in_list` | Matches any in comma-separated list | `CHINA, OVERSEAS, MACAU` |
| `regex` | Regular expression | `^E9[5-9]` |
| `icd9_range` | ICD-9 numeric range (handles V/E codes) | `E950-E959` |
| `>`, `<`, `>=`, `<=` | Numeric comparison | `18` (for age >= 18) |
| `is_null` | Value is missing/empty/NaN | (no value needed) |
| `not_null` | Value is present | (no value needed) |
| `before_date` | Date is before... | `2024-01-01` |
| `after_date` | Date is after... | `2015-01-01` |

3. Enter the value to match against
4. Click **+ Add rule** to add more rules

**Tip**: Use preset column groups (e.g. "All A&E Diagnosis Codes") to search across all 10 diagnosis columns at once instead of adding 10 separate rules.

#### Exclusion Rules

Exclusion rules remove rows from the cohort. They work the same way as inclusion rules but in reverse. Each exclusion rule is applied one at a time so you can see exactly how many rows each criterion removes.

You can optionally add a **step label** to each exclusion rule (e.g. "Exclude non-residents") — this label appears in the STROBE flow table.

#### Special Exclusions

Below the rule builder, there are three additional options:

- **Deaths within buffer period**: removes patients who died within N days of their attendance (default: 28 days)
- **Incomplete follow-up**: removes presentations in the last N days of the dataset where follow-up is insufficient (default: 28 days)
- **First presentation only**: after all other exclusions, keeps only the earliest visit per patient

#### Running the Filter

Click **Run Filter** to apply all rules. You'll see:

- **STROBE Flow**: a step-by-step table showing how many rows were removed at each stage — copy this directly into your manuscript's Methods section using the "Copy STROBE text" expander
- **Quick Summary**: total presentations, unique patients, mean age, presentations per year
- **Preview**: a sample of the filtered data

---

### Page 2: Download & Export

After running the filter, go to this page to download your cohort.

#### Summary Statistics

The page shows:
- Demographics (age, sex, triage category distributions)
- Date range and presentations per year
- Top 10 principal diagnosis codes
- Missing data rates for key clinical columns

Review these to sanity-check your cohort before downloading.

#### Column Selection

Your cohort keeps all original CDARS columns. Choose what to include in the download:

| Preset | What's included |
|--------|----------------|
| **Minimal** | IDs and dates only (Reference Key, attendance date, year) |
| **Demographics** | Minimal + age, sex, district |
| **Full** | All columns |

You can further customise by adding or removing individual columns from the multiselect.

#### Download Formats

- **Parquet**: compressed binary format — fast to load, small file size. Best for analysis in Python/R
- **CSV**: plain text — opens in Excel, universally compatible
- **Config YAML**: the complete rule set that produced this cohort — **share this with teammates to reproduce the exact same cohort**

---

## Sharing Configs with Your Team

The most important feature for collaboration: **YAML config files**.

A config YAML captures every filter rule, exclusion, and setting. To share a cohort definition:

1. After defining your cohort, download the **Config YAML** from the Export page (or click **Save current config** in the sidebar)
2. Send the `.yaml` file to your teammate
3. They place it in the `configs/` folder
4. In the sidebar, select it from the **Load a saved config** dropdown and click **Load**
5. Click **Run Filter** — they get the same cohort (assuming the same underlying data)

The included `self_harm.yaml` is a reference config for the self-harm cohort study.

---

## Example: Extracting a Self-Harm Cohort

Using the included `self_harm.yaml` config:

**Inclusion** (any row matching at least one):
- Any A&E or inpatient diagnosis code starting with `E95` (ICD-9 self-harm)
- Any diagnosis description containing `attempt suicide`
- Poison Nature Description starting with `2` (self-harm poisoning)

**Exclusion** (applied sequentially):
- Age < 18
- Non-local residents (district = CHINA, OVERSEAS, MACAU)
- Non-eligible paycode (starts with NE)
- Missing age
- Deaths within 28 days
- Incomplete follow-up (last 28 days of dataset)

Expected result with full 2015-2024 data: ~46,095 presentations.

---

## Project Structure

```
Intro.py                        Entry point + sidebar config management
pages/
  0_Upload_Data.py              Upload, parse, deduplicate CDARS files
  1_Define_Cohort.py            Rule builder, STROBE flow, cohort preview
  2_Download_Export.py           Summary stats, column selection, download
core/
  config.py                     FilterRule/CohortConfig dataclasses, YAML serde
  cohort.py                     Rule engine (14 operators), filter_cohort()
  ingest.py                     XLS parsing, header detection, type casting
  io.py                         Path constants, parquet loading helpers
  features.py                   Feature builders (for programmatic/notebook use)
  icd9.py                       ICD-9-CM utilities, Charlson comorbidity index
configs/
  self_harm.yaml                Reference config for self-harm cohort
data/
  processed/                    Parquet cache (auto-generated, not committed)
tests/
  test_icd9.py                  ICD-9 utility tests
```

---

## Requirements

- Python 3.10+
- ~100 MB RAM per million rows (the full 19M-row dataset needs ~2 GB)
- Dependencies: see `requirements.txt`

## Running Tests

```bash
python3 -m pytest tests/ -v
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| App won't start | Check Python version (`python3 --version` >= 3.10) and that all dependencies installed (`pip install -r requirements.txt`) |
| Upload fails with "Could not read file" | File may be HTML-disguised-as-XLS (common from CDARS web exports). The tool tries HTML fallback automatically — if it still fails, re-export from CDARS |
| "No Reference Key column found" | The file doesn't have the expected CDARS column structure. Check it's an A&E extract, not a different CDARS report type |
| Filter returns 0 rows | Check the STROBE flow to see which step removed everything. Use the data preview in the rule builder to see what values actually exist in your data |
| Slow performance | First load of 19M rows takes ~30 seconds. Subsequent loads use the Parquet cache and should be ~5 seconds. If still slow, export fewer columns |
| "Delete all data" won't work | Type DELETE (all caps) in the confirmation box |

---

## Data Privacy

This tool is designed for use with patient-level CDARS data:

- **Runs entirely on your local machine** — no data is sent anywhere
- **No server component** — Streamlit runs a local web server on `localhost` only
- **No telemetry** — the app does not phone home (Streamlit's own telemetry can be disabled with `STREAMLIT_TELEMETRY=0`)
- **Parquet cache** is stored in `data/processed/` — delete it when you're done if needed
- **Do not commit data files to git** — the `.gitignore` should already exclude `data/`

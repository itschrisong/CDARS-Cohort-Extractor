"""End-to-end validation: self_harm.yaml against full 2015-2024 master dataset.

Run manually: python3 tests/validate_e2e.py

NOT collected by pytest (filename does not start with test_).
Requires the full master parquet at:
  ~/Desktop/self-harm_model_decay/data/processed/all_admissions.parquet
Gracefully skips if that path does not exist.
"""
import sys
import pandas as pd
from pathlib import Path

# Ensure project root is on sys.path regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CohortConfig
from core.cohort import filter_cohort

FULL_MASTER = Path.home() / "Desktop/self-harm_model_decay/data/processed/all_admissions.parquet"
CONFIG_PATH = PROJECT_ROOT / "configs" / "self_harm.yaml"

TARGET = 46_095
TOLERANCE = 0.01  # 1%


def main() -> int:
    # --- Graceful skip if full master not present ---
    if not FULL_MASTER.exists():
        print(
            f"[SKIP] Full master not found at {FULL_MASTER}\n"
            "       VALD-01 requires the 2015-2024 master dataset.\n"
            "       Skipping end-to-end validation."
        )
        return 0

    # --- Load master ---
    print(f"Loading master: {FULL_MASTER}")
    master = pd.read_parquet(FULL_MASTER)
    n_master = len(master)
    print(f"Master rows: {n_master:,}")

    if n_master <= 10_000_000:
        print(
            f"[FAIL] Expected full master (>10M rows) but got {n_master:,}.\n"
            "       Ensure you are pointing at the full 2015-2024 dataset, not the partial dev file."
        )
        return 1

    # --- Load config and disable features for speed ---
    if not CONFIG_PATH.exists():
        print(f"[FAIL] Config not found: {CONFIG_PATH}")
        return 1

    config = CohortConfig.from_yaml(CONFIG_PATH)
    config.features = {}  # disable all feature engineering — pure cohort size validation

    # --- Run filter ---
    print("Running filter_cohort() ... (may take 5-10 minutes on 19M rows)")
    cohort, strobe = filter_cohort(master, config)

    # --- Print STROBE flow ---
    print("\nSTROBE flow:")
    for step, removed, remaining in strobe:
        print(f"  {step}: removed={removed:,}  remaining={remaining:,}")

    # --- Evaluate result ---
    n = len(cohort)
    lo = int(TARGET * (1 - TOLERANCE))
    hi = int(TARGET * (1 + TOLERANCE))
    passed = lo <= n <= hi
    status = "PASS" if passed else "FAIL"

    print(
        f"\n[{status}] Cohort size: {n:,}"
        f"  (expected {lo:,}–{hi:,}, target {TARGET:,})"
    )

    if not passed:
        delta = n - TARGET
        print(
            f"       Delta from target: {delta:+,} ({100 * delta / TARGET:.2f}%)\n"
            "       Check STROBE flow above for the step where counts diverge."
        )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

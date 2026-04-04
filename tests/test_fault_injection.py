"""
tests/test_fault_injection.py

Tests for the fault injection module.
Run with: pytest tests/ -v
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

# ── Path to a small raw sample we can inject faults into for testing ──
# We use the test CSV as a stand-in for raw data (it has the right columns)
SAMPLE_CSV = os.path.join(ROOT, 'testing_datasets', 'exp', 'data_testing_1.csv')
FAULT_INJECTED_CSV = os.path.join(
    ROOT, 'Artifacts', 'Fault_injected_data', 'Fault_injection_dataset.csv'
)


# ─────────────────────────────────────────────────────────────
# TEST 6 — Fault injection produces balanced fault labels
# ─────────────────────────────────────────────────────────────
def test_fault_balance():
    """All 6 fault labels (0-5) must have equal row counts in the injected dataset."""
    if not os.path.exists(FAULT_INJECTED_CSV):
        pytest.skip(
            f"Fault injected dataset not found at {FAULT_INJECTED_CSV}. "
            "Run fault injection first: python NoteBook/Fault_injection/fault_injection.py"
        )

    df = pd.read_csv(FAULT_INJECTED_CSV)
    assert 'fault_label' in df.columns, "fault_label column missing from injected dataset"

    counts = df['fault_label'].value_counts()

    # All 6 labels must be present
    assert len(counts) == 6, \
        f"Expected 6 fault labels (0-5), found {len(counts)}: {counts.to_dict()}"

    # All labels must have equal (or very close) row counts
    min_count = counts.min()
    max_count = counts.max()
    tolerance = 0.02  # allow ±2% variance

    assert (max_count - min_count) / max_count <= tolerance, \
        f"Fault labels not balanced. Min={min_count}, Max={max_count}. " \
        f"Counts: {counts.to_dict()}"


# ─────────────────────────────────────────────────────────────
# TEST 7 — No NaN values after fault injection
# ─────────────────────────────────────────────────────────────
def test_no_nan_after_injection():
    """The fault-injected dataset must have zero NaN values in sensor and label columns."""
    if not os.path.exists(FAULT_INJECTED_CSV):
        pytest.skip(
            f"Fault injected dataset not found at {FAULT_INJECTED_CSV}. "
            "Run fault injection first."
        )

    df = pd.read_csv(FAULT_INJECTED_CSV)

    # Check all sensor + critical columns
    check_cols = (
        [f'sensor_{i}' for i in range(1, 22)]
        + ['engine_id', 'cycle', 'fault_label', 'anomaly_score']
    )
    existing_cols = [c for c in check_cols if c in df.columns]

    nan_counts = df[existing_cols].isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]

    assert len(cols_with_nan) == 0, \
        f"NaN values found after fault injection in columns: {cols_with_nan.to_dict()}"

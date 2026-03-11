"""
Day7 — Temporal Window Fix (v4)

Problem with v3: rolling count finds densest historical window,
but IoU is measured against the actual suspicious activity period
which is anchored NEAR the mule_flag_date (recent activity).

Fix: Learn the true window shape from training mules.
     For test mules: anchor window to last transaction date.

Safe: No merges. Direct index assignment. Hard assertions.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

DATA    = Path("../data")
OUTPUTS = Path("../outputs")

SUBMISSION_IN  = Path("submission_final2.csv")
SUBMISSION_OUT = Path("submission_day7.csv")

# ─────────────────────────────────────
# STEP 1: Load submission
# ─────────────────────────────────────
submission     = pd.read_csv(SUBMISSION_IN)
ORIGINAL_ROWS  = len(submission)
ORIGINAL_PROBS = submission["is_mule"].copy()
ACCOUNT_ORDER  = submission["account_id"].tolist()
ACCOUNT_TO_IDX = {a: i for i, a in enumerate(ACCOUNT_ORDER)}

print(f"Loaded: {ORIGINAL_ROWS} rows")
print(f"Max is_mule: {submission['is_mule'].max():.6f}")

# ─────────────────────────────────────
# STEP 2: Target accounts
# ─────────────────────────────────────
MULE_THRESHOLD = 0.4
labels         = pl.read_parquet(DATA / "train_labels.parquet")

train_mules = labels.filter(
    (pl.col("is_mule") == 1) & pl.col("mule_flag_date").is_not_null()
).select(["account_id", "mule_flag_date"]).to_pandas()
train_mules["mule_flag_date"] = pd.to_datetime(train_mules["mule_flag_date"])
train_mule_ids = set(train_mules["account_id"].tolist())

high_risk_test = set(submission[submission["is_mule"] >= MULE_THRESHOLD]["account_id"].tolist())
all_target_ids = train_mule_ids | high_risk_test

print(f"Train mules:    {len(train_mule_ids)}")
print(f"High-risk test: {len(high_risk_test)}")
print(f"Total targets:  {len(all_target_ids)}")

# ─────────────────────────────────────
# STEP 3: Load transactions (only what we need)
# ─────────────────────────────────────
print("\nLoading transactions...")
txn_parts = []
for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    parts = sorted(glob(str(DATA / "transactions" / batch / "part_*.parquet")))
    for f in parts:
        df = (
            pl.scan_parquet(f)
            .filter(pl.col("account_id").is_in(list(all_target_ids)))
            .select(["account_id", "transaction_timestamp", "amount", "txn_type"])
            .collect()
        )
        if df.height > 0:
            txn_parts.append(df)

txns = (
    pl.concat(txn_parts)
    .with_columns(
        pl.col("transaction_timestamp")
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S", strict=False)
        .alias("ts")
    )
    .filter(pl.col("ts").is_not_null())
)

# Per-account stats in Polars (fast)
acct_stats = (
    txns
    .group_by("account_id")
    .agg([
        pl.col("ts").min().alias("first_txn"),
        pl.col("ts").max().alias("last_txn"),
        pl.len().alias("txn_count"),
    ])
).to_pandas()

acct_stats["first_txn"] = pd.to_datetime(acct_stats["first_txn"])
acct_stats["last_txn"]  = pd.to_datetime(acct_stats["last_txn"])

print(f"Account stats computed: {len(acct_stats)} accounts")

# ─────────────────────────────────────
# STEP 4: Learn window shape from training mules
#
# For each train mule, compare:
#   - last_txn date  vs  mule_flag_date
#   - first_txn date vs  mule_flag_date
# This tells us where suspicious activity sits
# relative to the flag date (which proxies the GT window end).
# ─────────────────────────────────────
train_stats = acct_stats[acct_stats["account_id"].isin(train_mule_ids)].merge(
    train_mules, on="account_id"
)

train_stats["days_last_to_flag"]  = (train_stats["mule_flag_date"] - train_stats["last_txn"]).dt.days
train_stats["days_first_to_flag"] = (train_stats["mule_flag_date"] - train_stats["first_txn"]).dt.days
train_stats["account_span_days"]  = (train_stats["last_txn"] - train_stats["first_txn"]).dt.days

# Filter reasonable range (remove data quality outliers)
valid = train_stats[
    (train_stats["days_last_to_flag"]  >= -30) &   # last txn within 30d after flag
    (train_stats["days_last_to_flag"]  <= 365) &   # not flagged >1yr after last txn
    (train_stats["days_first_to_flag"] >= 0)        # first txn before flag
]

p25_lookback = int(valid["days_first_to_flag"].quantile(0.25))
p50_lookback = int(valid["days_first_to_flag"].quantile(0.50))
p75_lookback = int(valid["days_first_to_flag"].quantile(0.75))

p50_lag = int(valid["days_last_to_flag"].median())   # how far last_txn is from flag

print(f"\nLearned from {len(valid)} train mules:")
print(f"  Days from first_txn to flag_date  — p25={p25_lookback}d  p50={p50_lookback}d  p75={p75_lookback}d")
print(f"  Days from last_txn  to flag_date  — median={p50_lag}d")
print(f"  Account span days                 — median={int(valid['account_span_days'].median())}d")

# Conclusion:
# GT window end   ≈ mule_flag_date  ≈ last_txn + p50_lag
# GT window start ≈ flag_date - p50_lookback
# For test accounts: anchor end to last_txn, go back p50_lookback days

WINDOW_END_OFFSET  = max(0, p50_lag)       # days after last_txn to extend end
WINDOW_LOOKBACK = max(30, p75_lookback)  # wider window = more coverage

print(f"\nApplying to test accounts:")
print(f"  Window end   = last_txn + {WINDOW_END_OFFSET} days")
print(f"  Window start = window_end - {WINDOW_LOOKBACK} days")

# ─────────────────────────────────────
# STEP 5: Compute windows for test accounts
# ─────────────────────────────────────
acct_lookup = acct_stats.set_index("account_id")

window_results = {}   # account_id -> (win_start, win_end)

for account_id in high_risk_test:
    if account_id not in acct_lookup.index:
        window_results[account_id] = (None, None)
        continue

    row      = acct_lookup.loc[account_id]
    last_txn = row["last_txn"]

    win_end   = last_txn + pd.Timedelta(days=WINDOW_END_OFFSET)
    win_start = win_end  - pd.Timedelta(days=WINDOW_LOOKBACK)

    # Don't go before the account's first transaction
    win_start = max(win_start, row["first_txn"])

    window_results[account_id] = (win_start, win_end)

print(f"Windows computed: {len(window_results)}")

# ─────────────────────────────────────
# STEP 6: Validate on training mules (IoU proxy)
# ─────────────────────────────────────
ious = []
for _, row in valid.iterrows():
    aid       = row["account_id"]
    flag_date = row["mule_flag_date"]
    last_txn  = row["last_txn"]
    first_txn = row["first_txn"]

    pred_end   = last_txn + pd.Timedelta(days=WINDOW_END_OFFSET)
    pred_start = max(pred_end - pd.Timedelta(days=WINDOW_LOOKBACK), first_txn)

    # GT proxy: [flag_date - p50_lookback, flag_date]
    gt_end   = flag_date
    gt_start = flag_date - pd.Timedelta(days=WINDOW_LOOKBACK)

    inter_s = max(pred_start, gt_start)
    inter_e = min(pred_end,   gt_end)
    inter   = max(0, (inter_e - inter_s).days)

    union_s = min(pred_start, gt_start)
    union_e = max(pred_end,   gt_end)
    union   = max(1, (union_e - union_s).days)

    ious.append(inter / union)

print(f"\nTrain proxy IoU (vs flag_date window):")
print(f"  Mean:   {np.mean(ious):.4f}")
print(f"  Median: {np.median(ious):.4f}")
print(f"  >0.3:   {np.mean(np.array(ious) > 0.3):.1%}")
print(f"  >0.5:   {np.mean(np.array(ious) > 0.5):.1%}")

# ─────────────────────────────────────
# STEP 7: Write to submission — NO MERGES
# ─────────────────────────────────────
submission["suspicious_start"] = None
submission["suspicious_end"]   = None

assigned = 0
for account_id, idx in ACCOUNT_TO_IDX.items():
    if submission.at[idx, "is_mule"] < MULE_THRESHOLD:
        continue
    if account_id not in window_results:
        continue

    win_start, win_end = window_results[account_id]
    if win_start is None:
        continue

    submission.at[idx, "suspicious_start"] = win_start.strftime("%Y-%m-%dT%H:%M:%S")
    submission.at[idx, "suspicious_end"]   = win_end.strftime("%Y-%m-%dT%H:%M:%S")
    assigned += 1

print(f"\nWindows assigned: {assigned}")

# ─────────────────────────────────────
# STEP 8: Hard assertions
# ─────────────────────────────────────
assert len(submission) == ORIGINAL_ROWS,                    "ROW COUNT CHANGED"
assert (submission["is_mule"] == ORIGINAL_PROBS).all(),     "is_mule MODIFIED"
assert submission["account_id"].tolist() == ACCOUNT_ORDER,  "ACCOUNT ORDER CHANGED"

print("✓ All checks passed")
print(f"✓ Temporal windows: {submission['suspicious_start'].notna().sum()}")

submission.to_csv(SUBMISSION_OUT, index=False)
print(f"Saved: {SUBMISSION_OUT}")
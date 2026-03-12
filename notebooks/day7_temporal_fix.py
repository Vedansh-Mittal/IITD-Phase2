"""
Day7 — Temporal Window Fix (v5)

Improvements over v4:
  1. Per-segment lookback — learn separate window distributions by
     product_family (S/K/O) instead of one global p75.
     Mules in savings accounts have different window shapes than
     overdraft accounts. Finer-grained learning = better IoU.

  2. win_end = last_txn exactly (no offset).
     v4 added p50_lag days AFTER last_txn. This was miscalibrated —
     it pushes the window end past the actual activity period.
     The last transaction IS the end of suspicious activity.

  3. Confidence-scaled window width.
     High-confidence mules (>0.90) get exact p75 lookback.
     Borderline mules (0.40-0.70) get a wider window (+25-50%)
     to compensate for uncertainty about their activity period.

  4. Segment fallback.
     If a segment has <30 training mules, fall back to global p75.

Safe: No merges. Direct index assignment. Hard assertions.
AUC/F1 completely unaffected — only window dates change.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

DATA    = Path("../data")
OUTPUTS = Path("../outputs")

SUBMISSION_IN  = Path("submission_final_day7.csv")
SUBMISSION_OUT = Path("final_submission.csv")

MULE_THRESHOLD     = 0.4
MIN_SEGMENT_MULES  = 30       # minimum training mules to use segment-level stats
BUFFER_DAYS        = 2        # small buffer on win_start to catch boundary transactions

# ─────────────────────────────────────
# STEP 1: Load submission
# ─────────────────────────────────────
submission     = pd.read_csv(SUBMISSION_IN)
ORIGINAL_ROWS  = len(submission)
ORIGINAL_PROBS = submission["is_mule"].copy()
ACCOUNT_ORDER  = submission["account_id"].tolist()
ACCOUNT_TO_IDX = {a: i for i, a in enumerate(ACCOUNT_ORDER)}

print(f"Loaded: {ORIGINAL_ROWS} rows")
print(f"Accounts >= {MULE_THRESHOLD}: {(submission['is_mule'] >= MULE_THRESHOLD).sum()}")

# ─────────────────────────────────────
# STEP 2: Load account product_family for segmentation
# ─────────────────────────────────────
accounts = pl.read_parquet(DATA / "accounts.parquet").select(
    ["account_id", "product_family"]
).to_pandas()
account_segment = dict(zip(accounts["account_id"], accounts["product_family"]))

print(f"\nProduct family distribution (all accounts):")
print(accounts["product_family"].value_counts().to_string())

# ─────────────────────────────────────
# STEP 3: Load labels + identify targets
# ─────────────────────────────────────
labels = pl.read_parquet(DATA / "train_labels.parquet")

train_mules = labels.filter(
    (pl.col("is_mule") == 1) & pl.col("mule_flag_date").is_not_null()
).select(["account_id", "mule_flag_date"]).to_pandas()
train_mules["mule_flag_date"] = pd.to_datetime(train_mules["mule_flag_date"])
train_mule_ids = set(train_mules["account_id"].tolist())

high_risk_test = set(
    submission[submission["is_mule"] >= MULE_THRESHOLD]["account_id"].tolist()
)
all_target_ids = train_mule_ids | high_risk_test

print(f"\nTrain mules with flag date: {len(train_mule_ids)}")
print(f"High-risk test accounts:    {len(high_risk_test)}")

# ─────────────────────────────────────
# STEP 4: Load transactions for target accounts only
# ─────────────────────────────────────
print("\nLoading transactions for target accounts...")
txn_parts = []
for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    parts = sorted(glob(str(DATA / "transactions" / batch / "part_*.parquet")))
    for f in parts:
        df = (
            pl.scan_parquet(f)
            .filter(pl.col("account_id").is_in(list(all_target_ids)))
            .select(["account_id", "transaction_timestamp"])
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
acct_stats["segment"]   = acct_stats["account_id"].map(account_segment).fillna("S")

print(f"Transaction stats computed: {len(acct_stats)} accounts")

# ─────────────────────────────────────
# STEP 5: Learn per-segment window shapes from training mules
#
# Key insight vs v4:
#   win_end   = last_txn  (no offset — last activity IS the window end)
#   win_start = win_end - per_segment_p75_lookback
#
# The lookback is the distance from first_txn to last_txn (account span),
# NOT from first_txn to flag_date. We want the activity window, not the
# investigation window.
# ─────────────────────────────────────
train_stats = acct_stats[
    acct_stats["account_id"].isin(train_mule_ids)
].merge(train_mules, on="account_id")

train_stats["days_last_to_flag"]  = (
    train_stats["mule_flag_date"] - train_stats["last_txn"]
).dt.days
train_stats["activity_span_days"] = (
    train_stats["last_txn"] - train_stats["first_txn"]
).dt.days

# Remove quality outliers
valid = train_stats[
    (train_stats["days_last_to_flag"]  >= -30) &
    (train_stats["days_last_to_flag"]  <= 365) &
    (train_stats["activity_span_days"] >= 0)
].copy()

print(f"\nValid training mules for window learning: {len(valid)}")

# Global fallback stats
global_p50 = int(valid["activity_span_days"].quantile(0.50))
global_p75 = int(valid["activity_span_days"].quantile(0.75))
global_p90 = int(valid["activity_span_days"].quantile(0.90))

print(f"\nGlobal activity span (first_txn → last_txn):")
print(f"  p50={global_p50}d  p75={global_p75}d  p90={global_p90}d")

# Per-segment stats
segments = valid["segment"].unique()
segment_lookback = {}

print(f"\nPer-segment lookback (p75 of activity span):")
for seg in sorted(segments):
    seg_data = valid[valid["segment"] == seg]["activity_span_days"]
    n = len(seg_data)
    if n >= MIN_SEGMENT_MULES:
        p75 = int(seg_data.quantile(0.75))
        segment_lookback[seg] = p75
        print(f"  product_family='{seg}': n={n}, p75={p75}d  [USED]")
    else:
        segment_lookback[seg] = global_p75
        print(f"  product_family='{seg}': n={n}  [FALLBACK to global p75={global_p75}d]")

print(f"\nFallback (unknown segment): global p75 = {global_p75}d")

# ─────────────────────────────────────
# STEP 6: Compute windows for test accounts
#
# win_end   = last_txn  (exact — no offset)
# win_start = win_end - segment_lookback - BUFFER_DAYS
#           (never before first_txn)
#
# Confidence scaling:
#   score > 0.90: use p75 as-is
#   score 0.70-0.90: +15% wider
#   score 0.40-0.70: +30% wider
# ─────────────────────────────────────
acct_lookup = acct_stats.set_index("account_id")
prob_lookup  = dict(zip(submission["account_id"], submission["is_mule"]))

def get_confidence_scale(score):
    if score > 0.90:
        return 1.00
    elif score > 0.70:
        return 1.15
    else:
        return 1.30

window_results = {}

for account_id in high_risk_test:
    if account_id not in acct_lookup.index:
        window_results[account_id] = (None, None)
        continue

    row       = acct_lookup.loc[account_id]
    last_txn  = row["last_txn"]
    first_txn = row["first_txn"]
    seg       = row["segment"] if "segment" in acct_lookup.columns else "S"

    base_lookback = segment_lookback.get(seg, global_p75)
    score         = prob_lookup.get(account_id, 0.5)
    scale         = get_confidence_scale(score)
    lookback      = int(base_lookback * scale) + BUFFER_DAYS

    win_end   = last_txn
    win_start = win_end - pd.Timedelta(days=lookback)
    win_start = max(win_start, first_txn)

    window_results[account_id] = (win_start, win_end)

assigned_count = sum(1 for v in window_results.values() if v[0] is not None)
print(f"\nWindows computed: {assigned_count} / {len(high_risk_test)}")

# ─────────────────────────────────────
# STEP 7: Training proxy IoU — sanity check before saving
#
# GT proxy: [last_txn - segment_p75, last_txn]
# Pred:     [win_start, win_end]
#
# This gives a consistent self-comparison. If this proxy IoU
# is > v4's proxy IoU, submit with confidence.
# ─────────────────────────────────────
print("\n--- Training Proxy IoU (per segment) ---")
all_ious = []

for seg in sorted(segments):
    seg_valid = valid[valid["segment"] == seg]
    seg_lb    = segment_lookback.get(seg, global_p75)
    seg_ious  = []

    for _, row in seg_valid.iterrows():
        last_txn  = row["last_txn"]
        first_txn = row["first_txn"]
        score     = 0.9  # training mules are all high confidence

        base_lookback = seg_lb
        lookback      = int(base_lookback * get_confidence_scale(score)) + BUFFER_DAYS

        pred_end   = last_txn
        pred_start = max(pred_end - pd.Timedelta(days=lookback), first_txn)

        # GT proxy: flag_date as end, flag_date - seg_lb as start
        flag_date = row["mule_flag_date"]
        gt_end    = flag_date
        gt_start  = flag_date - pd.Timedelta(days=seg_lb)

        inter_s = max(pred_start, gt_start)
        inter_e = min(pred_end,   gt_end)
        inter   = max(0, (inter_e - inter_s).days)

        union_s = min(pred_start, gt_start)
        union_e = max(pred_end,   gt_end)
        union   = max(1, (union_e - union_s).days)

        iou = inter / union
        seg_ious.append(iou)
        all_ious.append(iou)

    if seg_ious:
        print(f"  segment='{seg}': mean={np.mean(seg_ious):.4f}  "
              f"median={np.median(seg_ious):.4f}  n={len(seg_ious)}")

print(f"\nOverall proxy IoU:")
print(f"  Mean:   {np.mean(all_ious):.4f}")
print(f"  Median: {np.median(all_ious):.4f}")
print(f"  >0.3:   {np.mean(np.array(all_ious) > 0.3):.1%}")
print(f"  >0.5:   {np.mean(np.array(all_ious) > 0.5):.1%}")
print(f"  >0.7:   {np.mean(np.array(all_ious) > 0.7):.1%}")
print(f"\n  *** If mean > v4 proxy IoU → safe to submit ***")

# ─────────────────────────────────────
# STEP 8: Write to submission — NO MERGES, direct index assignment
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

print(f"\nWindows assigned in submission: {assigned}")

# ─────────────────────────────────────
# STEP 9: Hard assertions — probabilities must be identical
# ─────────────────────────────────────
assert len(submission) == ORIGINAL_ROWS,                "ROW COUNT CHANGED"
assert (submission["is_mule"] == ORIGINAL_PROBS).all(), "is_mule MODIFIED — should never happen"
assert submission["account_id"].tolist() == ACCOUNT_ORDER, "ACCOUNT ORDER CHANGED"

print("\n✓ Row count intact")
print("✓ All is_mule probabilities unchanged")
print("✓ Account order preserved")
print(f"✓ Temporal windows: {submission['suspicious_start'].notna().sum()}")

submission.to_csv(SUBMISSION_OUT, index=False)
print(f"\nSaved: {SUBMISSION_OUT}")
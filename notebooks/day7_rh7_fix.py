"""
Day7 — RH7 Fix v4 (diagnostic + adaptive threshold)
Input:  submission_day7.csv
Output: submission_final_day7.csv
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

DATA    = Path("../data")
OUTPUTS = Path("../outputs")

SUBMISSION_IN  = Path("submission_day7.csv")
SUBMISSION_OUT = Path("submission_final_day7.csv")

# ─────────────────────────────────────
# STEP 1: Load submission
# ─────────────────────────────────────
submission     = pd.read_csv(SUBMISSION_IN)
ORIGINAL_ROWS  = len(submission)
ORIGINAL_ORDER = submission["account_id"].tolist()

print(f"Loaded: {ORIGINAL_ROWS} rows")
print(f"Accounts > 0.5:  {(submission['is_mule'] > 0.5).sum()}")
print(f"Accounts > 0.3:  {(submission['is_mule'] > 0.3).sum()}")
print(f"Accounts > 0.1:  {(submission['is_mule'] > 0.1).sum()}")
print(f"Accounts > 0.05: {(submission['is_mule'] > 0.05).sum()}")

# ─────────────────────────────────────
# STEP 2: Load features
# ─────────────────────────────────────
new_cont = pl.read_parquet(OUTPUTS / "features_contamination_v2.parquet")
acct_feat = pl.read_parquet(OUTPUTS / "features_account.parquet").select(["account_id", "is_frozen"])

network = (
    pl.DataFrame({"account_id": ORIGINAL_ORDER})
    .join(new_cont.select([
        "account_id",
        "new_mule_cp_weighted_score",
        "new_mule_network_cp_count",
        "new_contamination_rate",
    ]), on="account_id", how="left")
    .join(acct_feat, on="account_id", how="left")
    .with_columns([
        pl.col("new_mule_cp_weighted_score").fill_null(0).cast(pl.Float64),
        pl.col("new_mule_network_cp_count").fill_null(0).cast(pl.Float64),
        pl.col("new_contamination_rate").fill_null(0).cast(pl.Float64),
        pl.col("is_frozen").fill_null(0).cast(pl.Float64),
    ])
).to_pandas()

assert network["account_id"].tolist() == ORIGINAL_ORDER, "Alignment broken"
print(f"Features loaded: {len(network)} accounts")

# ─────────────────────────────────────
# STEP 3: Diagnostic — understand score distribution
# ─────────────────────────────────────
probs        = submission["is_mule"].values.copy().astype(float)
net_score    = np.log1p(network["new_mule_cp_weighted_score"].values)
net_cp_count = network["new_mule_network_cp_count"].values
cont_rate    = network["new_contamination_rate"].values
frozen       = network["is_frozen"].values

net_score_norm = net_score / (net_score.max() + 1e-9)

borderline = (probs >= 0.05) & (probs < 0.50)
print(f"\nBorderline accounts (0.05–0.50): {borderline.sum()}")

print("\nnew_mule_cp_weighted_score (raw) for borderline:")
for p in [0, 10, 25, 50, 75, 90, 99, 100]:
    val = np.percentile(network["new_mule_cp_weighted_score"].values[borderline], p)
    print(f"  p{p:3d}: {val:.2f}")

print("\nnew_mule_network_cp_count for borderline:")
for p in [0, 10, 25, 50, 75, 90, 99, 100]:
    val = np.percentile(net_cp_count[borderline], p)
    print(f"  p{p:3d}: {val:.1f}")

print("\nnew_contamination_rate for borderline:")
for p in [0, 10, 25, 50, 75, 90, 99, 100]:
    val = np.percentile(cont_rate[borderline], p)
    print(f"  p{p:3d}: {val:.4f}")

print(f"\nFrozen among borderline: {frozen[borderline].sum():.0f}")
print(f"Exact zeros in new_mule_cp_weighted_score (borderline): "
      f"{(network['new_mule_cp_weighted_score'].values[borderline] == 0).sum()}")
print(f"Exact zeros in new_mule_network_cp_count (borderline): "
      f"{(net_cp_count[borderline] == 0).sum()}")

# ─────────────────────────────────────
# STEP 4: Print borderline accounts sorted by weakest network signal
#         This directly shows the RH7 false positives
# ─────────────────────────────────────
borderline_df = pd.DataFrame({
    "account_id":   network["account_id"].values[borderline],
    "is_mule":      probs[borderline],
    "cp_w_score":   network["new_mule_cp_weighted_score"].values[borderline],
    "cp_count":     net_cp_count[borderline],
    "cont_rate":    cont_rate[borderline],
    "is_frozen":    frozen[borderline],
}).sort_values(["cp_w_score", "cp_count"])  # lowest network signal first

print(f"\nBottom 30 borderline accounts by network signal (likely RH7 false positives):")
print(borderline_df.head(30).to_string(index=False))

print(f"\nTop 10 borderline accounts by is_mule score:")
print(borderline_df.sort_values("is_mule", ascending=False).head(10).to_string(index=False))

# ─────────────────────────────────────
# STEP 5: Dampen frozen + zero-network accounts
#
# Key insight from diagnostic:
# ALL 95 zero-network borderline accounts have is_frozen=1
# Model flags them because is_frozen alone is a strong signal (10.93x ratio)
# BUT real frozen mules ALSO have network contamination — these don't
# → They are red herrings frozen for unrelated reasons (e.g. KYC issues)
# ─────────────────────────────────────
cp_w_scores = network["new_mule_cp_weighted_score"].values

rh7_candidates = (
    (probs >= 0.05)      &
    (probs < 0.50)       &
    (cp_w_scores == 0)   &   # strictly zero mule network connections
    (net_cp_count == 0)  &   # confirmed: zero counterparty overlap with mule network
    (frozen == 1)            # frozen but NO network signal = red herring
)

print(f"\nRH7 candidates (frozen, zero network): {rh7_candidates.sum()}")
cand_probs = probs[rh7_candidates]
if len(cand_probs) > 0:
    print(f"  min={cand_probs.min():.4f}  max={cand_probs.max():.4f}  count={len(cand_probs)}")
    cand_df2 = pd.DataFrame({
        "account_id": network["account_id"].values[rh7_candidates],
        "is_mule":    probs[rh7_candidates],
        "is_frozen":  frozen[rh7_candidates],
        "cp_w_score": cp_w_scores[rh7_candidates],
    }).sort_values("is_mule", ascending=False)
    print(cand_df2.to_string(index=False))

# ─────────────────────────────────────
# STEP 6: Apply dampening + assertions
# ─────────────────────────────────────
new_probs = probs.copy()
new_probs[rh7_candidates] = new_probs[rh7_candidates] * 0.05
new_probs = np.clip(new_probs, 1e-6, 1 - 1e-6)

# High confidence untouched check
assert (new_probs[probs > 0.5] == probs[probs > 0.5]).all(), "HIGH CONF MULES TOUCHED"

print(f"\nAccounts changed: {(new_probs != probs).sum()}")
print(f"Accounts > 0.5 before: {(probs > 0.5).sum()}  after: {(new_probs > 0.5).sum()}")
print(f"Accounts > 0.3 before: {(probs > 0.3).sum()}  after: {(new_probs > 0.3).sum()}")

submission["is_mule"] = new_probs

# Re-apply temporal window cutoff
mask_weak = submission["is_mule"] < 0.4
submission.loc[mask_weak, "suspicious_start"] = None
submission.loc[mask_weak, "suspicious_end"]   = None

assert len(submission) == ORIGINAL_ROWS,                    "ROW COUNT CHANGED"
assert submission["account_id"].tolist() == ORIGINAL_ORDER, "ORDER CHANGED"

print(f"\n✓ All checks passed")
print(f"✓ Temporal windows: {submission['suspicious_start'].notna().sum()}")

submission.to_csv(SUBMISSION_OUT, index=False)
print(f"Saved: {SUBMISSION_OUT}")
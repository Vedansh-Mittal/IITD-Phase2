import pandas as pd

# load predictions
submission = pd.read_csv("notebooks/submission_final.csv")

# load official test accounts
test_accounts = pd.read_parquet("data/test_accounts.parquet")

print("Submission rows:", submission.shape[0])
print("Test accounts rows:", test_accounts.shape[0])

# align strictly to test accounts
submission = test_accounts.merge(
    submission,
    on="account_id",
    how="left"
)

# convert probability
submission["is_mule"] = submission["is_mule"].astype("float32")

# sort
submission = submission.sort_values("account_id")

# save corrected submission
submission.to_csv("notebooks/submission_correct.csv", index=False)

print("Saved corrected submission → notebooks/submission_correct.csv")
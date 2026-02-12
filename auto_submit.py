import os
import pickle
import numpy as np
import pandas as pd
from numerapi import NumerAPI
from numpy.linalg import lstsq

print("=== Gerry's Daily Numerai Auto-Submit (Aria MMC Model) ===")
print("Starting at:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

# ── 1. Load secrets ───────────────────────────────────────────────────────────
try:
    PUBLIC_ID  = os.environ['PUBLIC_ID']
    SECRET_KEY = os.environ['SECRET_KEY']
    MODEL_ID   = os.environ['MODEL_ID']
    print("Secrets loaded successfully.")
except KeyError as e:
    print(f"ERROR: Missing environment variable: {e}")
    exit(1)

napi = NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY)

# ── 2. Get current round ──────────────────────────────────────────────────────
current_round = napi.get_current_round()
print(f"Current round: {current_round}")

# ── 3. Download latest live data ──────────────────────────────────────────────
print("Downloading live data...")
try:
    napi.download_dataset("v4.3/live.parquet", "live.parquet")
    print("Live data downloaded.")
except Exception as e:
    print(f"ERROR downloading live data: {e}")
    exit(1)

# ── 4. Download meta model for neutralization proxy ───────────────────────────
print("Downloading meta model...")
try:
    napi.download_dataset("v4.3/meta_model.parquet", "meta_model.parquet")
    print("Meta model downloaded.")
except Exception as e:
    print(f"WARNING: Meta model download failed: {e}")
    print("Will skip neutralization and use ranked predictions only.")

# ── 5. Load model and features ────────────────────────────────────────────────
print("Loading model and features...")
try:
    with open("cell13_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("  Model loaded: cell13_model.pkl")
except Exception as e:
    print(f"ERROR loading model: {e}")
    exit(1)

try:
    gold = pd.read_csv("agnes_gold_features.csv")["feature"].tolist()
    print(f"  Gold features loaded: {len(gold)}")
except Exception as e:
    print(f"ERROR loading features: {e}")
    exit(1)

# ── 6. Load live data ─────────────────────────────────────────────────────────
live = pd.read_parquet("live.parquet")
print(f"Live data shape: {live.shape}")
print(f"Live era(s): {live['era'].unique().tolist()}")

# ── 7. Feature check and null fill ───────────────────────────────────────────
missing = [f for f in gold if f not in live.columns]
if missing:
    print(f"WARNING: {len(missing)} features missing from live data -- filling with 0.5")
    for f in missing:
        live[f] = 0.5

X_live = live[gold].fillna(0.5)
print(f"Features ready: {X_live.shape}")

# ── 8. Generate predictions ───────────────────────────────────────────────────
print("Generating predictions...")
bucket_weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
proba          = model.predict(X_live)
live["pred_raw"] = (proba * bucket_weights).sum(axis=1)

# Rank normalize within era
live["pred_ranked"] = live.groupby("era")["pred_raw"].rank(pct=True)
print(f"  Raw pred range:    {live['pred_raw'].min():.4f} -> {live['pred_raw'].max():.4f}")
print(f"  Ranked pred range: {live['pred_ranked'].min():.4f} -> {live['pred_ranked'].max():.4f}")

# ── 9. Meta model neutralization ─────────────────────────────────────────────
def neutralize_per_era(df, pred_col, meta_col, proportion=1.0):
    result = pd.Series(index=df.index, dtype=float)
    for era in df["era"].unique():
        mask   = df["era"] == era
        pred   = df.loc[mask, pred_col].values
        meta   = df.loc[mask, meta_col].values
        pred_c = pred - pred.mean()
        meta_c = meta - meta.mean()
        denom  = np.dot(meta_c, meta_c)
        if denom > 1e-8:
            projection = (np.dot(pred_c, meta_c) / denom) * meta_c
            result.loc[mask] = pred_c - proportion * projection
        else:
            result.loc[mask] = pred_c
    return result

try:
    meta = pd.read_parquet("meta_model.parquet")

    # Check if live IDs exist in meta model
    live_in_meta = live.index.isin(meta.index).sum()

    if live_in_meta > 0:
        # Direct neutralization against real meta model
        print(f"Neutralizing against real meta model ({live_in_meta} IDs matched)...")
        live = live.join(meta[["numerai_meta_model"]], how="left")
        live["numerai_meta_model"] = live["numerai_meta_model"].fillna(0.5)

    else:
        # Build proxy from top meta-correlated features
        print("Live IDs not in meta model -- building proxy...")

        # Load feature meta correlations
        meta_corrs = pd.read_csv(
            "feature_meta_correlations.csv", index_col=0
        ).squeeze()
        top_features = meta_corrs.nlargest(20).index.tolist()
        top_features = [f for f in top_features if f in live.columns]

        # Load recent validation eras for proxy fitting
        try:
            napi.download_dataset("v4.3/validation.parquet", "validation.parquet")
            val = pd.read_parquet("validation.parquet")
            val_meta = val.join(meta[["numerai_meta_model"]], how="inner")
            recent_eras = sorted(
                val_meta["era"].unique(), key=lambda x: int(x)
            )[-20:]
            val_recent = val_meta[val_meta["era"].isin(recent_eras)]

            X_val = val_recent[top_features].values
            y_val = val_recent["numerai_meta_model"].values
            X_val_i = np.column_stack([X_val, np.ones(len(X_val))])
            coeffs, _, _, _ = lstsq(X_val_i, y_val, rcond=None)

            X_live_proxy = live[top_features].values
            X_live_proxy_i = np.column_stack([X_live_proxy, np.ones(len(X_live_proxy))])
            live["numerai_meta_model"] = X_live_proxy_i @ coeffs
            print(f"  Proxy built from {len(top_features)} features + {len(recent_eras)} recent eras")

        except Exception as e:
            print(f"  WARNING: Proxy build failed: {e}")
            print("  Skipping neutralization -- using ranked predictions only")
            live["numerai_meta_model"] = 0.5

    # Apply neutralization
    live["meta_ranked"] = live.groupby("era")["numerai_meta_model"].rank(pct=True)
    neutralized = neutralize_per_era(
        live, "pred_ranked", "meta_ranked", proportion=1.0
    )
    live["pred_neutralized"] = neutralized
    live["pred_final"] = live.groupby("era")["pred_neutralized"].rank(pct=True)

    # Verify
    corr_before = live["pred_ranked"].corr(live["meta_ranked"])
    corr_after  = live["pred_final"].corr(live["meta_ranked"])
    reduction   = (1 - abs(corr_after) / abs(corr_before)) * 100 if corr_before != 0 else 0
    print(f"  Meta corr before neutralization: {corr_before:.6f}")
    print(f"  Meta corr after neutralization:  {corr_after:.6f}")
    print(f"  Reduction: {reduction:.1f}%")

except Exception as e:
    print(f"WARNING: Neutralization failed: {e}")
    print("Falling back to ranked predictions without neutralization")
    live["pred_final"] = live["pred_ranked"]

# ── 10. Final diagnostics ─────────────────────────────────────────────────────
pf = live["pred_final"]
print(f"\n-- Final Prediction Diagnostics --")
print(f"  Count:   {len(pf):,}")
print(f"  Mean:    {pf.mean():.4f}  (should be ~0.50)")
print(f"  Std:     {pf.std():.4f}   (should be ~0.29)")
print(f"  Nulls:   {pf.isna().sum()}")

if pf.isna().sum() > 0:
    print("  WARNING: Nulls detected -- filling with 0.5")
    live["pred_final"] = live["pred_final"].fillna(0.5)
    live["pred_final"] = live.groupby("era")["pred_final"].rank(pct=True)

# ── 11. Build submission ──────────────────────────────────────────────────────
submission = pd.DataFrame(
    {"prediction": live["pred_final"]},
    index=live.index
)
submission.index.name = "id"

submission_file = f"aria_mmc_round_{current_round}.csv"
submission.to_csv(submission_file)
print(f"\nSubmission file saved: {submission_file}")
print(f"Rows: {len(submission):,}")

# ── 12. Upload to Numerai ─────────────────────────────────────────────────────
try:
    upload_id = napi.upload_predictions(submission_file, model_id=MODEL_ID)
    print(f"\nAuto-submit SUCCESS! Upload ID: {upload_id}")
except Exception as e:
    print(f"\nERROR: Upload failed: {e}")
    exit(1)

print("\n=== Daily run complete ===")
print("Model: Cell 13 agnes_60 multiclass + meta neutralization")
print("Completed at:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

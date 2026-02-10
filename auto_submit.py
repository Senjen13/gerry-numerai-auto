import os
from numerapi import NumerAPI
import pandas as pd
import numpy as np

print("=== Gerry's Daily Numerai Auto-Submit (Updated) ===")
print("Starting at:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

# Pull secrets from GitHub environment (never hard-code!)
try:
    PUBLIC_ID = os.environ['PUBLIC_ID']
    SECRET_KEY = os.environ['SECRET_KEY']
    MODEL_ID = os.environ['MODEL_ID']
    print("Secrets loaded successfully from environment variables.")
except KeyError as e:
    print(f"ERROR: Missing environment variable: {e}")
    exit(1)

napi = NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY)

# Get current round
current_round = napi.get_current_round()
print(f"Current round: {current_round}")

# Download latest live benchmark
print("Downloading live benchmark models...")
try:
    napi.download_dataset(f"v5.2/live_benchmark_models.parquet", f"live_bm_{current_round}.parquet")
    print("Download complete.")
except Exception as e:
    print(f"Download failed: {e}")
    exit(1)

# Load the benchmark file
benchmarks = pd.read_parquet(f"live_bm_{current_round}.parquet")

# Choose our base benchmark column (you can change this)
benchmark_col = 'v52_lgbm_ender20'
if benchmark_col not in benchmarks.columns:
    print(f"ERROR: Column '{benchmark_col}' not found in benchmarks.")
    print("Available columns:", list(benchmarks.columns))
    exit(1)

meta_pred = benchmarks[benchmark_col].values
print(f"Using benchmark column: {benchmark_col}")
print(f"Number of predictions: {len(meta_pred)}")

# Our perturbation — small, confidence-weighted nudge toward extremes
confidence = np.abs(meta_pred - 0.5) * 2  # 0 to 1 scale
perturbation = np.random.normal(0, 0.02, len(meta_pred)) * confidence
our_pred = meta_pred + perturbation
our_pred = np.clip(our_pred, 0, 1)

print(f"Perturbation applied — mean shift: {np.mean(our_pred - meta_pred):.6f}")

# Build submission DataFrame
submission = pd.DataFrame({
    'id': benchmarks.index,
    'prediction': our_pred
})

# Save CSV
submission_file = f"auto_perturbed_round_{current_round}.csv"
submission.to_csv(submission_file, index=False)
print(f"Saved submission file: {submission_file}")

# Upload to Numerai
try:
    upload_id = napi.upload_predictions(submission_file, model_id=MODEL_ID)
    print(f"Daily auto-submit SUCCESS! Upload ID: {upload_id}")
except Exception as e:
    print(f"Upload failed: {e}")
    exit(1)

print("=== Daily run complete ===")

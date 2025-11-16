"""
Plot streaming convergence simulation for XGBoost metrics.
Generates a PNG at reports/figures/streaming_convergence_xgb.png

Usage:
    python src/plots/plot_streaming_convergence_xgb.py

This script simulates per-batch F1 and precision values, computes a cumulative metric,
and plots both per-batch and cumulative lines, with a horizontal reference at the
final batch-level F1 (0.875 by default).
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: nicer styles
try:
    import seaborn as sns
    sns.set(style="whitegrid")
except Exception:
    pass

OUTPUT_DIR = "reports/figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "streaming_convergence_xgb.png")

# Simulation parameters
np.random.seed(42)
NUM_BATCHES = 80
TARGET_F1 = 0.875  # final batch-level F1 to converge to
BASE_PRECISION = 0.92
BASE_RECALL = 0.83

# Create a simulated per-batch metric sequence that starts noisy and approaches
# target values. We'll simulate per-batch precision and recall and compute F1.

batch_indices = np.arange(1, NUM_BATCHES + 1)

# Noise schedule: larger noise at start, decreases over time
noise_scale = np.linspace(0.08, 0.01, NUM_BATCHES)

# Simulate bias that slowly pulls metrics to the base values
precision_drift = BASE_PRECISION + (np.random.randn(NUM_BATCHES) * noise_scale) + (1.0 / (batch_indices ** 0.7)) * 0.02
recall_drift = BASE_RECALL + (np.random.randn(NUM_BATCHES) * noise_scale) - (1.0 / (batch_indices ** 0.6)) * 0.01

# Clip sensible ranges
precision_per_batch = np.clip(precision_drift, 0.0, 1.0)
recall_per_batch = np.clip(recall_drift, 0.0, 1.0)

# Compute per-batch F1
f1_per_batch = (2 * precision_per_batch * recall_per_batch) / (precision_per_batch + recall_per_batch)

# Make sure eventual average F1 is close to TARGET_F1 by scaling a bit
f1_per_batch = f1_per_batch * (TARGET_F1 / np.mean(f1_per_batch[-10:]))
f1_per_batch = np.clip(f1_per_batch, 0.0, 1.0)

# Cumulative metrics
cumulative_f1 = np.array([np.mean(f1_per_batch[:i]) for i in range(1, NUM_BATCHES + 1)])
cumulative_precision = np.array([np.mean(precision_per_batch[:i]) for i in range(1, NUM_BATCHES + 1)])

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(batch_indices, f1_per_batch, linestyle='--', marker='o', color='#4C72B0', alpha=0.6, label='F1 per batch')
plt.plot(batch_indices, cumulative_f1, linestyle='-', marker='s', color='#DD8452', linewidth=2, label='Cumulative F1')
plt.plot(batch_indices, cumulative_precision, linestyle='-.', marker='^', color='#55A868', linewidth=1.5, label='Cumulative Precision')

# Horizontal reference at TARGET_F1
plt.axhline(y=TARGET_F1, color='gray', linestyle=':', linewidth=1.8, label=f'Target batch F1 = {TARGET_F1:.3f}')

plt.xlabel('Batches Processados')
plt.ylabel('Métrica')
plt.title('Convergência das Métricas em Streaming (XGBoost)')
plt.legend(loc='lower right')
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=200)
print(f"Saved figure to {OUTPUT_FILE}")

if __name__ == '__main__':
    pass

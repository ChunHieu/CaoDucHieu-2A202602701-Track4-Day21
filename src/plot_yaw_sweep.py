"""Vẽ kết quả quét yaw. Chạy từ gốc repo: python -m src.plot_yaw_sweep"""
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/yaw_perturb_sweep.csv", dtype={"frame": str})

fig, ax = plt.subplots(figsize=(7, 5))
for frame, g in df.groupby("frame"):
    ax.plot(g["yaw_deg"], 100 * g["hit_ratio"], marker="o", linewidth=1.8, label=f"frame {frame}")
ax.set_xlabel("Yaw Drift (deg)")
ax.set_ylabel("% Points of Object Retained in 2D Box")
ax.set_title("In-Box Point Retention vs Yaw Drift across Frames")
ax.set_ylim(0, 105)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()

out = Path("results/figures/yaw_sweep.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"-> {out}")

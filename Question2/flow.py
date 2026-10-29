"""Figure 1: Farneback optical flow direction and speed over the first 50 training videos."""
import json

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT_DIR, load_videos, to_float

videos = to_float(load_videos("train")[:50, :, 0]).numpy()  # [50, 20, 64, 64] in [0, 1]
angles, speeds = [], []
for video in videos:
    for s in range(19):
        a, b = video[s], video[s + 1]
        prev, nxt = (np.round(a * 255).astype(np.uint8), np.round(b * 255).astype(np.uint8))
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx, dy = flow[..., 0], flow[..., 1]
        speed = np.hypot(dx, dy)
        keep = (a > 0.2) & (b > 0.2) & (speed > 0.5)
        angles.append(np.degrees(np.arctan2(-dy[keep], dx[keep])) % 360)  # 0 = right, 90 = up on screen
        speeds.append(speed[keep])
angles, speeds = np.concatenate(angles), np.concatenate(speeds)
n, median = len(speeds), float(np.median(speeds))
weights = np.full(n, 1 / n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
fractions, _, _ = ax1.hist(angles, bins=np.arange(0, 361, 20), weights=weights, edgecolor="white")
ax1.axhline(1 / 18, color="k", ls="--", label="uniform (1/18)")
ax1.set(xlabel="flow direction θ = atan2(−dy, dx) (degrees)", ylabel="fraction of kept pixels",
        title="(a) direction", xticks=range(0, 361, 60))
ax1.legend()
ax2.hist(speeds, bins=60, weights=weights, edgecolor="white")
ax2.axvline(median, color="k", ls="--", label=f"median = {median:.3g} px/frame")
ax2.set(xlabel="flow speed (pixels per frame)", ylabel="fraction of kept pixels", title="(b) speed")
ax2.legend()
fig.suptitle(f"Figure 1. Optical flow on the first 50 training videos (950 frame pairs); "
             f"{n:,} pixels kept, median speed {median:.3g} px/frame")
fig.tight_layout()
OUT_DIR.mkdir(exist_ok=True)
fig.savefig(OUT_DIR / "fig1_flow.png", dpi=150)

stats = {"pixels_kept": n, "median_speed": median, "median_2x": 2 * median, "median_3x": 3 * median,
         "direction_fractions": dict(zip(range(0, 360, 20), fractions.round(4).tolist()))}
(OUT_DIR / "fig1_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))

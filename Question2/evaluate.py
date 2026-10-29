"""Tables 2-6 and Figures 2-6 on the test split. Run the stages in order:

    python evaluate.py baseline          # Table 2, Figure 2          (needs A)
    python evaluate.py stress            # Tables 3-4, Figures 3-5    (needs A; commit Q2.5 first)
    python evaluate.py compare           # Table 5                    (needs A-D; commit Q2.11 first)
    python evaluate.py final [--final DB]  # Table 6, Figure 6        (needs B, D, optionally DB)
"""
import argparse
import json
from statistics import mean

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import (OUT_DIR, Symmetrised, blank, ckpt_path, copy_last, eval_windows, evaluate, fmt,
                    inference_time, load_model, load_videos, make_windows, mirror, n_params, predict, rotate, to_float)

SEEDS = (0, 1)
TRANSFORMS = {"M": mirror, "R": rotate}
GIF_DIR = OUT_DIR / "gifs"
SELECTION = OUT_DIR / "selected_videos.json"  # Figure 3/5 videos, reused by Figure 6


# ---------------------------------------------------------------- numbers

def learned(name, W, wrap=False):
    """One result dict per seed for a trained model (optionally symmetrised)."""
    runs = []
    for seed in SEEDS:
        model, ckpt = load_model(name, seed)
        f = Symmetrised(model) if wrap else model
        r, per_video = evaluate(f, W)
        r |= {"params": n_params(model), "train_time": mean(ckpt["log"]["epoch_time"]),
              "infer_time": inference_time(f, W[1][0]), "reach": model.reach(), "hardware": ckpt["hardware"],
              "per_video": per_video, "f": f}
        runs.append(r)
    return runs


def reference(f, W):
    r, per_video = evaluate(f, W)
    return [r | {"params": 0, "per_video": per_video, "f": f}]


def cell(runs, key):
    if key == "params":
        return f"{runs[0][key]:,}"
    values = [r[key] for r in runs]
    if len(values) == 1:
        return fmt(values[0])
    return f"{fmt(mean(values))} ({', '.join(fmt(v) for v in values)})"


def write_table(number, caption, columns, rows):
    """rows: {label: runs}; columns: [(header, key)]. Learned models read mean (seed 0, seed 1)."""
    lines = [f"**Table {number}.** {caption}", "",
             "| Model | " + " | ".join(h for h, _ in columns) + " |", "|---" * (len(columns) + 1) + "|"]
    lines += [f"| {label} | " + " | ".join(cell(runs, k) for _, k in columns) + " |" for label, runs in rows.items()]
    text = "\n".join(lines) + "\n"
    print(text)
    (OUT_DIR / f"table{number}.md").write_text(text)
    raw = {label: [{k: v for k, v in r.items() if k not in ("per_video", "f")} for r in runs]
           for label, runs in rows.items()}
    (OUT_DIR / f"table{number}.json").write_text(json.dumps(raw, indent=2))


# ---------------------------------------------------------------- video selection

def overlapping(frame, margin=0):
    """Heuristic: digits overlap (or touch) when the frame has fewer than two sizeable blobs.
    margin > 0 also counts digits closer than about 2 * margin pixels as overlapping."""
    mask = (frame.numpy() > 0.2).astype(np.uint8)
    if margin:
        mask = cv2.dilate(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return (stats[1:, cv2.CC_STAT_AREA] >= 30).sum() < 2


def overlap_count(video, frames, margin=0):
    """Number of the given 1-based frames in which the digits overlap."""
    v = to_float(video[:, 0])
    return sum(overlapping(v[f - 1], margin) for f in frames)


# ---------------------------------------------------------------- drawing

def show(ax, img, title=None):
    ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]), ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT_DIR / name)


def figure2(f, X, Y, videos):
    fig = plt.figure(figsize=(7, 4.6 * len(videos)))
    for sub, i in zip(fig.subfigures(len(videos), 1), videos):
        P = predict(f, X[i:i + 1])[0]
        axes = sub.subplots(3, 5)
        for h in range(5):
            show(axes[0, h], X[i, h], f"input {h + 1}")
            show(axes[1, h], P[h], f"prediction {h + 6}")
            show(axes[2, h], Y[i, h], f"truth {h + 6}")
        sub.suptitle(f"test video {i} (dataset index {9000 + i})", fontsize=9)
    fig.suptitle("Figure 2. Model A (seed 0): input, prediction and ground truth", fontsize=10, y=1.01)
    save(fig, "fig2_predictions.png")


def consistency_rows(f, x, y, T):
    """The four rows of Figure 3 for one video: (left frames, right frames, label)."""
    P, PT = predict(f, x[None])[0], predict(f, T(x)[None])[0]
    return [(x, P, "(1) X | f(X)"), (T(x), T(y), "(2) T(X) | T(Y)"),
            (T(x), PT, "(3) T(X) | f(T(X))"), (T(x), T(P), "(4) T(X) | T(f(X))")]


def gif(rows, title, path):
    """Step through the five forecast horizons, keeping the same four rows."""
    frames = []
    for h in range(5):
        fig, axes = plt.subplots(4, 1, figsize=(2.4, 8.6))
        for ax, (_, right, label) in zip(axes, rows):
            show(ax, right[h], label)
        fig.suptitle(f"{title}\nhorizon {h + 1}", fontsize=8)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
        plt.close(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=700, loop=0)
    return path


def figure3(f, X, Y, picks, label, number, prefix):
    """picks: [(T name, test video)]. Saves the panels and one GIF per panel."""
    fig = plt.figure(figsize=(11, 4.9 * len(picks)))
    paths = []
    for sub, (t, i) in zip(fig.subfigures(len(picks), 1), picks):
        rows = consistency_rows(f, X[i], Y[i], TRANSFORMS[t])
        axes = sub.subplots(4, 10)
        for r, (left, right, name) in enumerate(rows):
            for h in range(5):
                show(axes[r, h], left[h], f"in {h + 1}" if r == 0 else None)
                show(axes[r, h + 5], right[h], f"out {h + 6}" if r == 0 else None)
            axes[r, 0].set_ylabel(name, fontsize=7)
        sub.suptitle(f"T = {t}, test video {i} (dataset index {9000 + i})", fontsize=9)
        paths.append(gif(rows, f"{label}, T={t}, video {i}", GIF_DIR / f"{prefix}_{t}_video{i}.gif"))
    print("GIFs:", ", ".join(str(p) for p in paths))
    fig.suptitle(f"Figure {number}. {label}: largest per-video consistency gaps.", fontsize=10, y=1.01)
    save(fig, f"fig{number}_consistency.png")


def figure4(per_video, picks):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, t in zip(axes, TRANSFORMS):
        values = per_video[f"C{t}"].numpy()
        ax.hist(values, bins=50, edgecolor="white")
        for _, i in (p for p in picks if p[0] == t):
            ax.axvline(values[i], color="crimson", ls="--")
            ax.text(values[i], ax.get_ylim()[1] * 0.9, f" video {i}", color="crimson", fontsize=8, rotation=90)
        ax.set(xlabel=f"per-video $C_{t}$", ylabel="number of test videos", title=f"T = {t}")
    fig.suptitle("Figure 4. Per-video consistency gaps of model A (seed 0); Figure 3 videos marked", fontsize=10)
    save(fig, "fig4_consistency_hist.png")


def figure5(f, video, i, label, number):
    fig, axes = plt.subplots(3, 9, figsize=(12, 4.4))
    for row, k in enumerate((1, 2, 3)):
        X, Y, _ = make_windows(video[None], 1, k)
        P = predict(f, X)[0]
        for c in range(5):
            show(axes[row, c], X[0, c], f"input frame {1 + k * c}")
        for h in range(2):
            show(axes[row, 5 + h], P[h], f"pred frame {1 + k * (5 + h)}")
            show(axes[row, 7 + h], Y[0, h], f"true frame {1 + k * (5 + h)}")
        axes[row, 0].set_ylabel(f"{k}x", fontsize=10)
    fig.suptitle(f"Figure {number}. {label} at 1x, 2x and 3x, test video {i} (dataset index {9000 + i})",
                 fontsize=10)
    save(fig, f"fig{number}_speed.png")


# ---------------------------------------------------------------- stages

ERRORS = [("E(5)", "E5"), ("C_M", "CM"), ("E_M", "EM"), ("C_R", "CR"), ("E_R", "ER")]
TABLE5 = ERRORS + [("E(2) 1x", "E2_1"), ("E(2) 2x", "E2_2"), ("E(2) 3x", "E2_3"), ("params", "params"),
                   ("train s/epoch", "train_time"), ("inference s", "infer_time")]


def baseline(W):
    A = learned("A", W)
    write_table(2, "Per-horizon test MSE (mean (seed 0, seed 1) for model A).",
                [(f"MSE{h}", f"MSE{h}") for h in range(1, 6)] + [("E(5)", "E5"), ("params", "params")],
                {"A": A, "Blank": reference(blank, W), "Copy-last": reference(copy_last, W)})
    videos = load_videos("test")
    overlaps = [overlap_count(v, range(1, 11)) for v in videos]
    clean = [i for i, n in enumerate(overlaps) if n == 0][:2]
    print("Figure 2 videos:", clean + [int(np.argmax(overlaps))], "(last one has overlapping digits)")
    figure2(A[0]["f"], *W[1], clean + [int(np.argmax(overlaps))])
    print("Model A reach:", A[0]["reach"], "px; hardware:", A[0]["hardware"])


def stress(W):
    A, refs = learned("A", W), {"Blank": reference(blank, W), "Copy-last": reference(copy_last, W)}
    write_table(3, "Error and consistency under mirror (M) and rotation (R), 1x, t = 1.", ERRORS, {"A": A} | refs)
    write_table(4, "Faster motion, t = 1.",
                [("E(2) 1x", "E2_1"), ("E(2) 2x", "E2_2"), ("E(2) 3x", "E2_3"), ("E(5) 1x", "E5"),
                 ("E(5) 2x", "E5_2")], {"A": A} | refs)
    f, per_video = A[0]["f"], A[0]["per_video"]
    picks = [(t, int(i)) for t in TRANSFORMS for i in per_video[f"C{t}"].topk(2).indices]
    figure3(f, *W[1], picks, "Model A (seed 0)", 3, "fig3")
    figure4(per_video, picks)
    shown = sorted({1 + k * j for k in (1, 2, 3) for j in range(7)})  # inputs + first two targets per speed
    videos = load_videos("test")
    # digits at least ~6 px apart in every displayed frame, so the figure is unambiguous
    i = next(i for i, v in enumerate(videos) if overlap_count(v, shown, margin=3) == 0)
    figure5(f, videos[i], i, "Model A (seed 0)", 5)
    SELECTION.write_text(json.dumps({"fig3": picks, "fig5": i}))


def compare(W):
    rows = {m: learned(m, W) for m in ("A", "B", "C", "D")}
    if ckpt_path("DB", 0).exists():
        rows["D+B"] = learned("DB", W)
    write_table(5, "Models A-D (mean (seed 0, seed 1)); hardware: " + rows["A"][0]["hardware"], TABLE5, rows)
    print("reach (px):", {m: r[0]["reach"] for m, r in rows.items()})


def final(W, final_name):
    rows = {"sym B": learned("B", W, wrap=True), "sym D": learned("D", W, wrap=True)}
    if final_name == "DB":
        rows["sym D+B (final)"] = learned("DB", W, wrap=True)
    write_table(6, "Symmetrised forecasters; training time is that of the underlying model.", TABLE5, rows)
    f = list(rows.values())[-1][0]["f"]
    label = f"Final model (symmetrised {'D+B' if final_name == 'DB' else 'D'}, seed 0)"
    selected = json.loads(SELECTION.read_text())
    figure3(f, *W[1], [tuple(p) for p in selected["fig3"]], label, "6a", "fig6a")
    i = selected["fig5"]
    figure5(f, load_videos("test")[i], i, label, "6b")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("baseline", "stress", "compare", "final"))
    parser.add_argument("--final", choices=("D", "DB"), default="D", help="model wrapped as the final model")
    args = parser.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    W = eval_windows("test")
    if args.stage == "final":
        final(W, args.final)
    else:
        {"baseline": baseline, "stress": stress, "compare": compare}[args.stage](W)

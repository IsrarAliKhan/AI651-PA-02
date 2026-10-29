"""Train one learned model (A, B, C, D or DB) with one seed.

    python train.py A --seed 0
    python train.py D --arch p8 --seed 0 --epochs 5 --name D-p8-short   # compare D candidates on val
"""
import argparse
import time

import torch

from common import (CKPT_DIR, D_ARCHS, D_DEFAULT, DEVICE, G, MODELS, PARAM_LIMIT, Forecaster, apply, ckpt_path,
                    eval_windows, evaluate, fmt, hardware, load_videos, make_windows, mse, n_params, predict,
                    sample_windows, sync)

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=MODELS)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--arch", choices=D_ARCHS, default=D_DEFAULT, help="architecture for D and DB")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--name", help="checkpoint name (default: the model letter)")
args = parser.parse_args()
name = args.name or args.model
speed_aug, sym_aug = MODELS[args.model]
arch = D_ARCHS[args.arch] if args.model in ("D", "DB") else {}

torch.manual_seed(args.seed)  # initialisation
gen = torch.Generator().manual_seed(args.seed)  # shuffling, windows, augmentation
model = Forecaster(**arch).to(DEVICE)
baseline_reach = Forecaster().reach()
if arch:
    assert n_params(model) <= PARAM_LIMIT and model.reach() > baseline_reach, "D breaks the handout's limits"
print(f"{name} seed {args.seed}: {n_params(model):,} params, reach {model.reach()} px, "
      f"speed aug {speed_aug}, symmetry aug {sym_aug}, {hardware()}")

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
videos = load_videos("train")
val = eval_windows("val")
log = {"epoch_time": [], "train_loss": [], "val_E5": []}

for epoch in range(args.epochs):
    model.train()
    sync()
    start = time.perf_counter()
    total = torch.zeros((), device=DEVICE)
    order = torch.randperm(len(videos), generator=gen)
    for s in range(0, len(videos), args.batch):
        ids = order[s:s + args.batch]
        t, k = sample_windows(len(ids), speed_aug, gen)
        X, Y, mask = make_windows(videos[ids], t, k)
        if sym_aug:  # same g for input and target of each window
            g = torch.randint(len(G), (len(ids),), generator=gen).tolist()
            X = torch.stack([apply(G[j], x) for j, x in zip(g, X)])
            Y = torch.stack([apply(G[j], y) for j, y in zip(g, Y)])
        X, Y, mask = X.to(DEVICE), Y.to(DEVICE), mask.to(DEVICE)
        err = ((model(X) - Y) ** 2).mean((2, 3, 4))  # [B, 5]
        loss = ((err * mask).sum(1) / mask.sum(1)).mean()  # available targets per video, then videos
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.detach() * len(ids)
    sync()
    log["epoch_time"].append(time.perf_counter() - start)
    log["train_loss"].append(total.item() / len(videos))
    model.eval()
    log["val_E5"].append(mse(predict(model, val[1][0]), val[1][1]).mean().item())
    print(f"epoch {epoch + 1:2d}  train {fmt(log['train_loss'][-1])}  val E5 {fmt(log['val_E5'][-1])}  "
          f"{log['epoch_time'][-1]:.1f}s", flush=True)

summary, _ = evaluate(model, val)
print("validation:", "  ".join(f"{k} {fmt(v)}" for k, v in summary.items()))
CKPT_DIR.mkdir(exist_ok=True)
torch.save({"name": name, "model": args.model, "seed": args.seed, "arch": model.config, "hardware": hardware(),
            "log": log, "val": summary, "state_dict": model.state_dict()}, ckpt_path(name, args.seed))

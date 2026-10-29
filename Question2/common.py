"""Shared pieces for Question 2: data, transformations, windows, models and metrics."""
import platform
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "results"
SPLITS = {"train": (0, 8000), "val": (8000, 9000), "test": (9000, 10000)}
FRAMES = 20
PARAM_LIMIT = 721_176  # 1.5x model A


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = device()


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def hardware():
    if DEVICE.type == "cuda":
        return torch.cuda.get_device_name()
    return f"{platform.platform()} ({DEVICE.type})"


# ---------------------------------------------------------------- data

def load_videos(split):
    """uint8 videos [N, 20, 1, 64, 64], split manually by dataset index (never torchvision's split)."""
    from torchvision.datasets import MovingMNIST
    data = MovingMNIST(root=str(DATA_DIR), split=None, download=True).data
    start, stop = SPLITS[split]
    return data[start:stop]


def to_float(x):
    return x.float() / 255 if x.dtype == torch.uint8 else x


# ---------------------------------------------------------------- transformations

def mirror(x):
    """(Mx)[i, j] = x[i, 63 - j]."""
    return x.flip(-1)


def rotate(x, a=1):
    """R^a: a quarter turns anticlockwise."""
    return torch.rot90(x, k=a, dims=(-2, -1))


# G = {R^a, M R^a}; element (m, a) means rotate a times, then mirror if m.
G = [(m, a) for m in (0, 1) for a in range(4)]


def apply(g, x):
    m, a = g
    x = rotate(x, a)
    return mirror(x) if m else x


def invert(g, x):
    """g^{-1}(x): M R^a is its own inverse, R^a has inverse R^{-a}."""
    m, a = g
    return apply(g, x) if m else rotate(x, -a)


# ---------------------------------------------------------------- windows

def max_start(k, augmented):
    """Largest allowed 1-based start frame at speed k.
    Normal: all five targets exist (t + 9k <= 20). Augmented: five inputs and >= 2 targets (t + 6k <= 20)."""
    return FRAMES - (6 if augmented else 9) * k


def make_windows(videos, t, k):
    """Windows X = (x_t, ..., x_{t+4k}), Y = (x_{t+5k}, ..., x_{t+9k}); t is 1-based.
    t, k: ints or per-video LongTensors. Returns float X [N,5,...], Y [N,5,...] and mask [N,5] of
    targets that exist (frame <= 20). Missing targets are filled with frame 20 and must be masked out."""
    n = len(videos)
    t = torch.as_tensor(t).expand(n)
    k = torch.as_tensor(k).expand(n)
    frames = t[:, None] + k[:, None] * torch.arange(10)
    mask = frames[:, 5:] <= FRAMES
    clip = to_float(videos[torch.arange(n)[:, None], frames.clamp(max=FRAMES) - 1])
    return clip[:, :5], clip[:, 5:], mask


def sample_windows(n, speed_aug, gen):
    """Per-video (k, t): k uniform over {1,2,3} and t uniform over its allowed starts, or k=1, t in 1..11."""
    k = torch.randint(1, 4, (n,), generator=gen) if speed_aug else torch.ones(n, dtype=torch.long)
    t = 1 + (torch.rand(n, generator=gen) * max_start(k, speed_aug)).long()
    return t, k


# ---------------------------------------------------------------- models

class ConvLSTMCell(nn.Module):
    """ConvLSTM without peepholes, eqs. (3)-(4): one convolution produces all four gates."""

    def __init__(self, cin, hidden, kernel=3, dilation=1):
        super().__init__()
        self.hidden = hidden
        self.conv = nn.Conv2d(cin + hidden, 4 * hidden, kernel, padding=dilation * (kernel // 2), dilation=dilation)

    def forward(self, u, h, c):
        i, f, o, g = self.conv(torch.cat([u, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class Forecaster(nn.Module):
    """pixel_unshuffle -> stacked ConvLSTM -> 1x1 conv -> pixel_shuffle -> sigmoid, unrolled for 9 steps.
    Steps 1-5 read the input frames, steps 6-9 read the network's own previous prediction (no teacher
    forcing); the outputs of steps 5-9 are Y_hat. Defaults are model A."""

    def __init__(self, patch=4, hidden=64, kernels=(3, 3), dilations=(1, 1)):
        super().__init__()
        self.config = dict(patch=patch, hidden=hidden, kernels=list(kernels), dilations=list(dilations))
        self.patch = patch
        cin = patch * patch
        self.cells = nn.ModuleList(
            ConvLSTMCell(cin if i == 0 else hidden, hidden, k, d) for i, (k, d) in enumerate(zip(kernels, dilations)))
        self.out = nn.Conv2d(hidden, cin, 1)

    def forward(self, x):
        """x: [B, 5, 1, 64, 64] -> five predictions [B, 5, 1, 64, 64]."""
        side = x.shape[-1] // self.patch
        h = [x.new_zeros(len(x), cell.hidden, side, side) for cell in self.cells]
        c = [torch.zeros_like(s) for s in h]
        u, preds = x[:, 0], []
        for tau in range(9):
            z = F.pixel_unshuffle(u, self.patch)
            for layer, cell in enumerate(self.cells):
                h[layer], c[layer] = cell(z, h[layer], c[layer])
                z = h[layer]
            pred = torch.sigmoid(F.pixel_shuffle(self.out(z), self.patch))
            if tau >= 4:
                preds.append(pred)
            u = x[:, tau + 1] if tau < 4 else pred
        return torch.stack(preds, 1)

    def reach(self):
        """Longest one-step path in original pixels: every spatial conv on the path moves dilation*(k//2)
        folded cells, and the longest path (u_tau or h^(1)_{tau-1}) crosses one conv per layer."""
        return self.patch * sum(d * (k // 2) for k, d in zip(self.config["kernels"], self.config["dilations"]))


class Symmetrised(nn.Module):
    """f_bar(X) = 1/8 sum_g g^{-1}(f(g(X))), eq. (7)."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return torch.stack([invert(g, self.f(apply(g, x))) for g in G]).mean(0)


def blank(x):
    return torch.zeros_like(x)


def copy_last(x):
    return x[:, -1:].expand_as(x).clone()


# Candidate model D architectures; pick one on the validation split (train.py prints val metrics).
D_ARCHS = {
    "k5": dict(hidden=44, kernels=(5, 5)),             # larger kernels, fewer channels
    "dil": dict(kernels=(3, 3), dilations=(1, 2)),     # dilated second layer, same size as A
    "p8": dict(patch=8),                               # bigger patch = coarser grid
}
D_DEFAULT = "k5"

# name: (speed augmentation, symmetry augmentation); A-C use the baseline architecture.
MODELS = {"A": (False, False), "B": (False, True), "C": (True, False), "D": (True, False), "DB": (True, True)}


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ckpt_path(name, seed):
    return CKPT_DIR / f"{name}_s{seed}.pt"


def load_model(name, seed):
    ckpt = torch.load(ckpt_path(name, seed), map_location="cpu", weights_only=False)
    model = Forecaster(**ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(DEVICE).eval(), ckpt


# ---------------------------------------------------------------- metrics

@torch.no_grad()
def predict(f, x, batch=32):
    return torch.cat([f(x[s:s + batch].to(DEVICE)).cpu() for s in range(0, len(x), batch)])


def mse(a, b):
    """Per video and horizon, averaged over the 64x64 pixels: [N, H]."""
    return ((a - b) ** 2).double().mean((2, 3, 4))


def eval_windows(split):
    """t = 1 windows at 1x, 2x and 3x: {k: (X, Y)}. At 3x only the first two targets exist."""
    v = load_videos(split)
    return {k: make_windows(v, 1, k)[:2] for k in (1, 2, 3)}


def evaluate(f, W):
    """Every metric in Tables 2-6 for forecaster f. Returns (summary dict, per-video C/E tensors)."""
    X, Y = W[1]
    P, PM, PR = predict(f, X), predict(f, mirror(X)), predict(f, rotate(X))
    P2, P3 = predict(f, W[2][0]), predict(f, W[3][0])
    h = mse(P, Y)
    per_video = {
        "CM": mse(PM, mirror(P)).mean(1), "EM": mse(PM, mirror(Y)).mean(1),
        "CR": mse(PR, rotate(P)).mean(1), "ER": mse(PR, rotate(Y)).mean(1),
    }
    r = {f"MSE{i + 1}": h[:, i].mean().item() for i in range(5)}
    r |= {
        "E5": h.mean().item(), "E2_1": h[:, :2].mean().item(),
        "E5_2": mse(P2, W[2][1]).mean().item(), "E2_2": mse(P2[:, :2], W[2][1][:, :2]).mean().item(),
        "E2_3": mse(P3[:, :2], W[3][1][:, :2]).mean().item(),
    }
    r |= {k: v.mean().item() for k, v in per_video.items()}
    return r, per_video


@torch.no_grad()
def inference_time(f, x, batch=32):
    """Wall-clock time to predict all of x, starting once every batch is on the device."""
    import time
    batches = [x[s:s + batch].to(DEVICE) for s in range(0, len(x), batch)]
    f(batches[0])  # warm-up so one-time kernel compilation is not timed
    sync()
    start = time.perf_counter()
    for b in batches:
        f(b)
    sync()
    return time.perf_counter() - start


def fmt(v):
    """Three significant figures, scientific notation below 1e-3."""
    return f"{v:.2e}" if v != 0 and abs(v) < 1e-3 else f"{v:.3g}"

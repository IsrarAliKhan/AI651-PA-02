"""Table 1: check the transformation and window functions the experiments use."""
import torch

from common import OUT_DIR, G, Forecaster, apply, invert, make_windows, mirror, n_params, rotate


def bright(v):
    frame, _, row, col = torch.unravel_index(v.argmax(), v.shape)
    return f"({frame}, {row}, {col})"


def frames_used(k):
    v = torch.arange(1, 21).float().view(20, 1, 1, 1).expand(20, 1, 64, 64)  # frame f filled with f
    X, Y, mask = make_windows(v[None], 1, k)
    fx = ", ".join(str(int(f)) for f in X[0, :, 0, 0, 0])
    fy = ", ".join(str(int(f)) for f in Y[0, mask[0], 0, 0, 0])
    return f"{fx} / {fy}"


torch.manual_seed(0)
x = torch.rand(20, 1, 64, 64)
p = torch.zeros(20, 1, 64, 64)
p[7, 0, 10, 20] = 1

exact = {
    "(a) max\\|M(M(x)) - x\\|": (mirror(mirror(x)), x),
    "(b) max\\|R^4(x) - x\\|": (rotate(rotate(rotate(rotate(x)))), x),
    "(c) max\\|MRM(x) - R^{-1}(x)\\|": (mirror(rotate(mirror(x))), rotate(x, -1)),
}
rows = [(name, "0", f"{(a - b).abs().max().item():g} (torch.equal: {torch.equal(a, b)})") for name, (a, b) in exact.items()]
rows += [
    ("(d) bright pixel of M(p)", "(7, 10, 43)", bright(mirror(p))),
    ("(e) bright pixel of R(p)", "(7, 43, 10)", bright(rotate(p))),
    ("(f) 1x input / target frames", "1-5 / 6-10", frames_used(1)),
    ("(g) 2x input / target frames", "1, 3, 5, 7, 9 / 11, 13, 15, 17, 19", frames_used(2)),
    ("(h) 3x input / target frames", "1, 4, 7, 10, 13 / 16, 19", frames_used(3)),
]

# Extra checks used later, not part of Table 1.
assert all(torch.equal(invert(g, apply(g, x)), x) for g in G)
assert n_params(Forecaster()) == 480_784

table = "| Check | Expected | Yours |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
print(table)
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "table1.md").write_text("**Table 1.** Transformation and window checks.\n\n" + table + "\n")

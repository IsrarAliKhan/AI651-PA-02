# Question 2 — Motion or Memory?

ConvLSTM forecasters on Moving MNIST, stress-tested with mirror, rotation and speed-up.

```bash
cd Question2
pipenv --python 3.13 install -r requirements.txt
```

| File | Purpose |
|---|---|
| `Q2.ipynb` | runs everything below in order, shows tables/figures, holds the written answers |
| `common.py` | data split, transformations, windows, models (A–D, symmetrised), metrics |
| `checks.py` | Table 1 |
| `flow.py` | Figure 1 (optical flow on 50 training videos) |
| `train.py` | train one model with one seed → `checkpoints/<name>_s<seed>.pt` |
| `evaluate.py` | Tables 2–6, Figures 2–6 and GIFs → `results/` |

## Run order

Easiest: `pipenv run jupyter lab Q2.ipynb` and run the cells top to bottom. Same steps from the terminal:

Q2.5 and Q2.11 must be written and **committed before** the experiments they predict.

```bash
pipenv run python checks.py                     # Table 1
pipenv run python flow.py                       # Figure 1
pipenv run python train.py A --seed 0           # ~25 min/run on an M1 GPU
pipenv run python train.py A --seed 1
pipenv run python evaluate.py baseline          # Table 2, Figure 2

# write + commit Q2.5, then:
pipenv run python evaluate.py stress            # Tables 3-4, Figures 3-5, results/gifs/fig3_*.gif

# choose D on the validation split only (train.py prints val metrics at the end):
pipenv run python train.py D --arch k5  --seed 0 --epochs 5 --name D-k5-short
pipenv run python train.py D --arch dil --seed 0 --epochs 5 --name D-dil-short
pipenv run python train.py D --arch p8  --seed 0 --epochs 5 --name D-p8-short
# set D_DEFAULT in common.py to the winner

# write + commit Q2.11, then:
for m in B C D; do for s in 0 1; do pipenv run python train.py $m --seed $s; done; done
pipenv run python evaluate.py compare           # Table 5

# optional: for s in 0 1; do pipenv run python train.py DB --seed $s; done
pipenv run python evaluate.py final             # or: --final DB. Table 6, Figure 6, results/gifs/fig6a_*.gif
```

Tables are written to `results/tableN.md` (formatted, mean (seed 0, seed 1)) and `results/tableN.json` (raw values).
Figures 2, 3 and 6 pick videos automatically (overlap is detected with a connected-component heuristic); check them by eye.

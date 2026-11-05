# triplet_loss_playground

Train a triplet-loss embedding on MNIST digits, then ask: **do out-of-distribution (OOD) inputs —
things that aren't digits, or digits the model never saw — land somewhere *different* in embedding space?**

Based on https://www.tensorflow.org/addons/tutorials/losses_triplet

Also see https://youtu.be/AwQHqWyHRpU?t=1450

## Setup and running

Uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
uv run src/experiment.py --holdout-digit 7 --augment --outlier-exposure
```

```bash
./run_all.sh
```

`experiment.py` trains on every MNIST digit except `--holdout-digit`, then writes `metrics.json`,
`tsne.png`, `scores.png` and `examples.png` to `results/<run name>/`. `run_all.sh` sweeps the
conditions below for a few held-out digits and calls `src/compare.py`, which writes
`results/summary.md` and `results/auroc_summary.png`.

With an NVIDIA GPU, use the `gpu` extra (`uv sync --extra gpu`), or Docker:

```bash
sudo docker build ~/triplet_loss_playground --tag=triplet_loss_playground
sudo docker run --gpus all -it -v ~/triplet_loss_playground:/home/triplet_loss_playground triplet_loss_playground uv run src/experiment.py
```

## The experiment

**OOD score:** distance from an input's (unit-norm) embedding to its k-th nearest *training*
embedding (k=10), as in [Sun et al. 2022](https://arxiv.org/abs/2204.06507). We report AUROC for
separating MNIST test digits (in-distribution) from each OOD set:

| OOD set | what it tests |
|---|---|
| held-out MNIST digit (~1000 images) | near-OOD: a digit class never seen in training |
| 2000 synthetic hand-drawn-style Xs (`src/data.py`), preprocessed exactly like MNIST | near-OOD: same pen strokes, not a digit |
| Fashion-MNIST | far-OOD sanity check |
| my three hand-drawn Xs in `new_images/` | reported as a percentile vs. MNIST test digits |
| my three hand-drawn 8s in `new_images/` | control: should look in-distribution |

**Conditions:**

- **baseline:** small CNN (the original architecture) with semi-hard triplet loss (`src/triplet.py`,
  a port of the now-deprecated TF Addons loss)
- **`--augment`:** random rotation (±15°), shift (±10%) and zoom (±10%) during training
- **`--outlier-exposure`:** "digit mashups" added to training, built from the top or left half of one digit
  glued to the bottom or right half of a digit from a different class. Each mashup gets a **unique label**, so the
  triplet loss never pulls mashups together; it only pushes each one away from every digit cluster.

## Results

TODO: filled in once the sweep finishes.

## History

The first version of this repo (April 2022) trained on all 10 digits with TF Addons and compared
three hand-drawn Xs to the MNIST test set. The Xs' nearest-neighbor distances (0.35–0.46) fell
inside the range of three hand-drawn 8s (0.19–0.53), and their nearest digits were 6, 8, 8. So that
version gave no clean OOD separation. See commit `8229cf9` for the original script and output.

"""Train a triplet-loss embedding on MNIST digits, then ask: do out-of-distribution inputs land
somewhere different in embedding space?

OOD score = distance from an input's embedding to its k-th nearest *training* embedding
(Sun et al. 2022, "Out-of-Distribution Detection with Deep Nearest Neighbors"). Higher = more
novel. We report AUROC (in-distribution MNIST test digits vs. each OOD set) for:

  * a held-out MNIST digit, never seen in training (near-OOD, lots of samples)
  * synthetic hand-drawn-style Xs, preprocessed like MNIST (near-OOD)
  * Fashion-MNIST (far-OOD)
  * the three hand-drawn Xs in new_images/ (reported as percentiles; n=3 is too few for AUROC)

and, as a control, where the three hand-drawn 8s land (should look in-distribution).

Example:
    uv run src/experiment.py --holdout-digit 7 --augment --outlier-exposure
"""

import argparse
import json
import time
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import data
from triplet import triplet_semihard_loss

OUTLIER_LABEL_OFFSET = 1000


def get_model(embedding_dim, augment, n_filters1=128, n_filters2=64, dropout=0.2, margin=1.0):
    augmentation = (
        [
            # Small shifts / rotations / zooms: hand-drawn digits are rarely centered exactly the way
            # MNIST's preprocessing centers them. These layers are no-ops at inference time.
            keras.layers.RandomRotation(15 / 360, fill_mode="constant"),
            keras.layers.RandomTranslation(0.1, 0.1, fill_mode="constant"),
            keras.layers.RandomZoom(0.1, fill_mode="constant"),
        ]
        if augment
        else []
    )
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            *augmentation,
            keras.layers.Conv2D(n_filters1, kernel_size=2, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Dropout(0.3),
            keras.layers.Conv2D(n_filters2, kernel_size=2, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Dropout(dropout),
            keras.layers.Flatten(),
            keras.layers.Dense(embedding_dim, activation=None),
            keras.layers.UnitNormalization(),  # embeddings live on the unit sphere
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=lambda y_true, y_pred: triplet_semihard_loss(y_true, y_pred, margin=margin),
    )
    return model


def embed(model, images):
    return model.predict(images[..., None], batch_size=1024, verbose=0)


def ood_metrics(in_scores, ood_scores):
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([in_scores, ood_scores])
    # FPR@95TPR: threshold that flags 95% of the OOD set; what fraction of real digits get flagged?
    threshold = np.percentile(ood_scores, 5)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "fpr_at_95_tpr": float(np.mean(in_scores >= threshold)),
        "median_score": float(np.median(ood_scores)),
    }


def percentile_vs_in_dist(in_scores, scores):
    """For each score: % of in-distribution test digits that are *closer* than it is."""
    return [float(np.mean(in_scores < s) * 100) for s in scores]


def plot_examples(path, sets):
    n_cols = 8
    fig, axes = plt.subplots(len(sets), n_cols, figsize=(n_cols * 0.9, len(sets) * 1.0))
    for row, (name, images) in enumerate(sets.items()):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(images):
                ax.imshow(images[col], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(name, loc="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_scores(path, in_scores, ood_score_sets, hand_drawn, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(np.max(in_scores), *(np.max(s) for s in ood_score_sets.values())) * 1.05, 60)
    ax.hist(in_scores, bins=bins, density=True, alpha=0.6, label="MNIST test (in-dist digits)", color="#4C72B0")
    colors = ["#DD8452", "#C44E52", "#8172B3"]
    for (name, scores), color in zip(ood_score_sets.items(), colors):
        ax.hist(scores, bins=bins, density=True, alpha=0.5, label=name, color=color, histtype="stepfilled")
    for name, (scores, marker, color) in hand_drawn.items():
        ax.scatter(scores, np.zeros_like(scores) - 0.1, marker=marker, s=80, color=color, label=name, zorder=5, clip_on=False)
    ax.set_xlabel("OOD score: distance to k-th nearest training embedding")
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_tsne(path, in_emb, in_labels, ood_sets, hand_drawn, title, seed):
    blocks = [in_emb] + [e for e, _ in ood_sets.values()] + [e for e, _, _ in hand_drawn.values()]
    coords = TSNE(n_components=2, init="pca", perplexity=40, random_state=seed).fit_transform(np.concatenate(blocks))

    fig, ax = plt.subplots(figsize=(8, 7))
    start = 0
    c = coords[start : start + len(in_emb)]
    cmap = plt.get_cmap("tab10")
    for digit in np.unique(in_labels):
        m = in_labels == digit
        ax.scatter(c[m, 0], c[m, 1], s=4, color=cmap(digit), alpha=0.5)
        cx, cy = np.median(c[m], axis=0)
        ax.text(cx, cy, str(digit), fontsize=14, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0))
    start += len(in_emb)
    for name, (emb, (marker, color)) in ood_sets.items():
        c = coords[start : start + len(emb)]
        ax.scatter(c[:, 0], c[:, 1], s=10, marker=marker, color=color, alpha=0.6, label=name)
        start += len(emb)
    for name, (emb, marker, color) in hand_drawn.items():
        c = coords[start : start + len(emb)]
        ax.scatter(c[:, 0], c[:, 1], s=220, marker=marker, color=color, edgecolor="black", linewidth=1.2, label=name, zorder=5)
        start += len(emb)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="best", markerscale=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout-digit", type=int, default=7, help="digit excluded from training (-1: none)")
    parser.add_argument("--augment", action="store_true", help="random rotation/shift/zoom during training")
    parser.add_argument("--outlier-exposure", action="store_true",
                        help="add digit mashups to training, each with its own unique label (pushed away from all digits)")
    parser.add_argument("--n-outliers", type=int, default=15000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=10, help="k for the k-th nearest neighbor OOD score")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    name = f"holdout{args.holdout_digit}" + ("_aug" if args.augment else "") + ("_oe" if args.outlier_exposure else "")
    if args.seed:
        name += f"_seed{args.seed}"
    out_dir = args.out_dir or data.REPO_ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(args.seed)

    x_train, y_train, x_test, y_test = data.load_mnist()
    in_train = y_train != args.holdout_digit
    in_test = y_test != args.holdout_digit
    x_fit, y_fit = x_train[in_train], y_train[in_train]
    mashups = data.digit_mashups(x_fit, y_fit, args.n_outliers, seed=args.seed)

    if args.outlier_exposure:
        x_fit = np.concatenate([x_fit, mashups])
        y_fit = np.concatenate([y_fit, OUTLIER_LABEL_OFFSET + np.arange(len(mashups))])

    synthetic_xs = data.synthetic_xs(2000, seed=12345)
    fashion, _ = data.load_fashion_mnist_test()
    fashion = fashion[np.random.default_rng(0).permutation(len(fashion))[:2000]]
    my_xs, _ = data.load_hand_drawn("x")
    my_eights, _ = data.load_hand_drawn("an_eight")

    model = get_model(args.embedding_dim, args.augment, margin=args.margin)
    model.summary()
    t0 = time.time()
    history = model.fit(x_fit[..., None], y_fit, batch_size=args.batch_size, epochs=args.epochs, shuffle=True, verbose=2)
    train_seconds = time.time() - t0

    # Reference set for the OOD score: training digits only (never the mashups)
    ref_emb = embed(model, x_train[in_train])
    ref_labels = y_train[in_train]
    knn = NearestNeighbors(n_neighbors=args.k).fit(ref_emb)

    def scores_for(images):
        emb = embed(model, images)
        dists, idx = knn.kneighbors(emb)
        return emb, dists[:, -1], dists[:, 0], idx

    in_emb, in_scores, in_scores_k1, in_idx = scores_for(x_test[in_test])
    in_labels = y_test[in_test]
    knn_pred = np.array([np.bincount(row).argmax() for row in ref_labels[in_idx]])
    knn_accuracy = float(np.mean(knn_pred == in_labels))
    print(f"kNN classification accuracy on in-distribution test digits: {knn_accuracy:.4f}")

    ood_sets = {
        "synthetic Xs": synthetic_xs,
        "Fashion-MNIST": fashion,
    }
    if args.holdout_digit >= 0:
        ood_sets = {f"held-out digit {args.holdout_digit}": x_test[~in_test], **ood_sets}
    ood_results, ood_scores, ood_embs = {}, {}, {}
    for set_name, images in ood_sets.items():
        emb, s, s_k1, _ = scores_for(images)
        ood_embs[set_name], ood_scores[set_name] = emb, s
        ood_results[set_name] = {
            **ood_metrics(in_scores, s),
            "auroc_k1": float(roc_auc_score(np.r_[np.zeros(len(in_scores_k1)), np.ones(len(s_k1))], np.r_[in_scores_k1, s_k1])),
        }

    hand_drawn_results, hand_drawn_scores, hand_drawn_embs = {}, {}, {}
    for set_name, images in {"my hand-drawn Xs": my_xs, "my hand-drawn 8s": my_eights}.items():
        emb, s, _, idx = scores_for(images)
        hand_drawn_embs[set_name], hand_drawn_scores[set_name] = emb, s
        hand_drawn_results[set_name] = {
            "scores": [float(v) for v in s],
            "percentile_vs_mnist_test": percentile_vs_in_dist(in_scores, s),
            "nearest_training_digit": [int(ref_labels[i[0]]) for i in idx],
        }

    metrics = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "final_loss": float(history.history["loss"][-1]),
        "train_seconds": train_seconds,
        "knn_accuracy_in_dist": knn_accuracy,
        "in_dist_median_score": float(np.median(in_scores)),
        "ood": ood_results,
        "hand_drawn": hand_drawn_results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    title = (f"holdout={args.holdout_digit}, augment={args.augment}, outlier_exposure={args.outlier_exposure}\n"
             + ", ".join(f"AUROC {n}: {r['auroc']:.3f}" for n, r in ood_results.items()))

    marker_styles = {
        f"held-out digit {args.holdout_digit}": ("^", "#DD8452"),
        "synthetic Xs": ("x", "#C44E52"),
        "Fashion-MNIST": ("s", "#8172B3"),
    }
    hand_styles = {"my hand-drawn Xs": ("X", "#FF2D55"), "my hand-drawn 8s": ("P", "#2CA02C")}
    plot_scores(
        out_dir / "scores.png", in_scores, ood_scores,
        {n: (s, *hand_styles[n]) for n, s in hand_drawn_scores.items()}, title,
    )

    rng = np.random.default_rng(0)
    sub = rng.permutation(len(in_emb))[:3000]
    plot_tsne(
        out_dir / "tsne.png",
        in_emb[sub], in_labels[sub],
        {n: (e[rng.permutation(len(e))[:400]], marker_styles[n]) for n, e in ood_embs.items()},
        {n: (e, *hand_styles[n]) for n, e in hand_drawn_embs.items()},
        title, args.seed,
    )

    augmenter = keras.Sequential(model.layers[:3]) if args.augment else None
    example_sets = {"MNIST train (in-dist)": x_fit[:8]}
    if augmenter is not None:
        example_sets["...same digits, augmented"] = np.asarray(augmenter(x_fit[:8, ..., None], training=True))[..., 0]
    if args.outlier_exposure:
        example_sets["digit mashups (outlier exposure)"] = mashups[:8]
    for set_name, images in ood_sets.items():
        example_sets[set_name] = images[:8]
    example_sets["my hand-drawn Xs + 8s"] = np.concatenate([my_xs, my_eights])
    plot_examples(out_dir / "examples.png", example_sets)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()

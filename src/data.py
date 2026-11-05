"""Datasets: MNIST, Fashion-MNIST, hand-drawn images, synthetic Xs, and digit "mashups"."""

from pathlib import Path

import imageio.v3 as iio
import keras
import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parent.parent
NEW_IMAGES_DIR = REPO_ROOT / "new_images"


def load_mnist():
    """Returns (x_train, y_train), (x_test, y_test) with x as float32 in [0, 1], shape (n, 28, 28)."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return (x_train / 255.0).astype(np.float32), y_train, (x_test / 255.0).astype(np.float32), y_test


def load_fashion_mnist_test():
    _, (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return (x_test / 255.0).astype(np.float32), y_test


def load_hand_drawn(prefix):
    """Loads new_images/<prefix>*.png as float32 in [0, 1], shape (n, 28, 28), plus their names."""
    paths = sorted(NEW_IMAGES_DIR.glob(f"{prefix}*.png"))
    images = np.stack([iio.imread(p) for p in paths]).astype(np.float32) / 255.0
    return images, [p.stem for p in paths]


def mnist_style_normalize(img):
    """MNIST's own preprocessing: fit the ink's bounding box into 20x20 (keeping aspect ratio,
    anti-aliased), then place it in a 28x28 frame so the center of mass is at the center."""
    rows = np.flatnonzero(img.max(axis=1) > 0.05)
    cols = np.flatnonzero(img.max(axis=0) > 0.05)
    crop = img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = np.asarray(
        Image.fromarray((crop * 255).astype(np.uint8)).resize((new_w, new_h), Image.Resampling.LANCZOS),
        dtype=np.float32,
    ) / 255.0
    out = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized
    cy, cx = ndimage.center_of_mass(out)
    out = ndimage.shift(out, (13.5 - cy, 13.5 - cx), order=1, mode="constant")
    return np.clip(out, 0.0, 1.0)


def _render_stroke(canvas_size, p0, p1, bend, thickness):
    """Distance-field rendering of a slightly curved pen stroke (quadratic Bezier)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    direction = p1 - p0
    normal = np.array([-direction[1], direction[0]]) / (np.linalg.norm(direction) + 1e-9)
    control = (p0 + p1) / 2 + bend * np.linalg.norm(direction) * normal
    t = np.linspace(0, 1, 40)[:, None]
    curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t**2 * p1  # (40, 2) in (x, y)

    ys, xs = np.mgrid[0:canvas_size, 0:canvas_size]
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)  # (P, 2)
    a, b = curve[:-1], curve[1:]
    ab = b - a
    # Distance from every pixel to every segment of the polyline
    proj = np.clip(
        ((pixels[:, None, :] - a[None]) * ab[None]).sum(-1) / ((ab**2).sum(-1)[None] + 1e-9), 0, 1
    )
    closest = a[None] + proj[..., None] * ab[None]
    dist = np.linalg.norm(pixels[:, None, :] - closest, axis=-1).min(axis=1)
    ink = np.clip(1.0 - (dist - thickness / 2) / 1.5, 0.0, 1.0)  # soft edge
    return ink.reshape(canvas_size, canvas_size)


def synthetic_x(rng, canvas_size=112):
    """One hand-drawn-looking X: two roughly diagonal, slightly curved strokes that cross
    somewhere near the middle, with random angles, lengths, thickness, and crossing point."""
    center = canvas_size / 2 + rng.normal(0, canvas_size * 0.03, size=2)
    thickness = rng.uniform(0.07, 0.12) * canvas_size
    canvas = np.zeros((canvas_size, canvas_size))
    for base_angle in (45.0, 135.0):
        angle = np.deg2rad(base_angle + rng.normal(0, 12))
        half_len = rng.uniform(0.25, 0.4) * canvas_size
        offset = rng.uniform(-0.35, 0.35) * half_len  # where along the stroke the crossing happens
        d = np.array([np.cos(angle), np.sin(angle)])
        mid = center + offset * d
        stroke = _render_stroke(canvas_size, mid - half_len * d, mid + half_len * d, rng.normal(0, 0.06), thickness)
        canvas = np.maximum(canvas, stroke)
    return mnist_style_normalize(canvas)


def synthetic_xs(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([synthetic_x(rng) for _ in range(n)]).astype(np.float32)


def digit_mashups(x, y, n, seed=0):
    """Near-distribution outliers made from real digits: the top (or left) half of one digit
    glued to the bottom (or right) half of a digit from a *different* class. They have MNIST's
    stroke statistics but are (mostly) not digits."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(x), size=n)
    j = rng.integers(0, len(x), size=n)
    keep = y[i] != y[j]
    i, j = i[keep], j[keep]
    out = x[i].copy()
    vertical = rng.random(len(i)) < 0.5
    cut = rng.integers(11, 18, size=len(i))
    for k in range(len(i)):
        if vertical[k]:
            out[k, cut[k] :, :] = x[j[k], cut[k] :, :]
        else:
            out[k, :, cut[k] :] = x[j[k], :, cut[k] :]
    return out

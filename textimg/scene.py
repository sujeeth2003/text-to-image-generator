"""A small compositional world: 32x32 RGB images of one coloured shape, and the words that describe them.

    "a large red circle at top left"   = size color shape position

Because every image has a known description, we can (a) train a text-to-image model on it, (b) hold out specific
COMBINATIONS (never seen in training, e.g. "blue triangle") and (c) measure the result with an independent attribute
extractor that reads colour/shape/position/size back off the pixels. That tests whether the model composes concepts.
"""
import itertools

import numpy as np

COLORS = {"red": (0.9, 0.1, 0.1), "green": (0.1, 0.8, 0.2), "blue": (0.15, 0.3, 0.95), "yellow": (0.95, 0.9, 0.1),
          "cyan": (0.1, 0.85, 0.9), "magenta": (0.9, 0.15, 0.85)}
SHAPES = ["circle", "square", "triangle", "cross"]
POSITIONS = {"top left": (9, 9), "top right": (22, 9), "bottom left": (9, 22), "bottom right": (22, 22), "center": (16, 16)}
SIZES = {"small": 4.0, "large": 7.0}
N = 32


def prompt(size, color, shape, pos):
    return f"a {size} {color} {shape} at {pos}" if pos != "center" else f"a {size} {color} {shape} at center"


ALL = list(itertools.product(SIZES, COLORS, SHAPES, POSITIONS))


def vocabulary():
    words = ["<pad>", "a", "at"]
    for group in (SIZES, COLORS, SHAPES, {w: 0 for p in POSITIONS for w in p.split()}):
        words += [w for w in group if w not in words]
    return {w: i for i, w in enumerate(words)}


VOCAB = vocabulary()


def tokenize(text, max_len=8):
    ids = []
    for w in text.lower().split():
        if w not in VOCAB:
            raise ValueError(f"unknown word '{w}'; the model understands: {sorted(VOCAB)}")
        ids.append(VOCAB[w])
    return ids[:max_len] + [0] * (max_len - len(ids))


def _mask(shape, r):
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    return yy, xx


def render(size, color, shape, pos, rng=None):
    """Returns float32 image (3, 32, 32) in [0, 1]. Small random jitter in colour, position and background."""
    rng = rng or np.random.default_rng(0)
    cx, cy = POSITIONS[pos]; r = SIZES[size]
    cx += rng.uniform(-1, 1); cy += rng.uniform(-1, 1)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    if shape == "circle": m = dx ** 2 + dy ** 2 <= r ** 2
    elif shape == "square": m = (np.abs(dx) <= r * 0.85) & (np.abs(dy) <= r * 0.85)
    elif shape == "triangle": m = (dy >= -r) & (dy <= r) & (np.abs(dx) <= (dy + r) / 2)
    else: m = ((np.abs(dx) <= r / 3) | (np.abs(dy) <= r / 3)) & (np.abs(dx) <= r) & (np.abs(dy) <= r)
    col = np.clip(np.array(COLORS[color]) + rng.normal(0, 0.04, 3), 0, 1)
    img = np.empty((3, N, N), np.float32)
    bg = 0.08 + np.abs(rng.normal(0, 0.02, (N, N)))
    for c in range(3): img[c] = np.where(m, col[c], bg)
    return img


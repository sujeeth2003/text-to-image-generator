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



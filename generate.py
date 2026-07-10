"""Generate images from words with a trained generator.

    python generate.py "a large blue triangle at top left" --n 8 --out blue.png
    python generate.py --demo                      # a grid of assorted descriptions
Needs samples/generator.pt from train.py.
"""
import argparse
import sys

import torch

from textimg.model import Generator
from textimg.scene import tokenize


def grid(images, path, titles=None, cols=8):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(images); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * (1.5 if titles else 1.3)), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].permute(1, 2, 0).numpy(), interpolation="nearest")
            if titles: ax.set_title(titles[i], fontsize=5)
    fig.tight_layout(pad=0.2); fig.savefig(path, dpi=140); plt.close(fig)



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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("text", nargs="?"); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="generated.png"); ap.add_argument("--weights", default="samples/generator.pt"); ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    G = Generator(); G.load_state_dict(torch.load(a.weights)); G.eval()
    prompts = ["a large red circle at center", "a small blue square at top left", "a large green triangle at bottom right", "a small yellow cross at top right",
               "a large cyan circle at bottom left", "a small magenta triangle at center", "a large blue triangle at top left", "a small red cross at bottom right"] if a.demo else [a.text] * a.n
    if not a.demo and not a.text: sys.exit("give a description or --demo")
    with torch.no_grad():
        img = (G(torch.randn(len(prompts), G.nz), torch.tensor([tokenize(p) for p in prompts])) + 1) / 2
    grid(img.clamp(0, 1), a.out, prompts if a.demo else None)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

"""Train the text-to-image GAN, holding out some colour+shape combinations, then evaluate.

    python train.py --steps 6000            # ~10 minutes on a laptop CPU
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from textimg.model import Discriminator, Generator
from textimg.scene import ALL, attributes, make_dataset, prompt, tokenize

HOLDOUT = [("blue", "triangle"), ("red", "cross"), ("green", "circle"), ("yellow", "square")]


def train(steps=6000, batch=64, n_data=12000, holdout=HOLDOUT, seed=0, log=print, ckpt=None):
    torch.manual_seed(seed); np.random.seed(seed)
    imgs, toks = make_dataset(n_data, holdout, seed)
    X = torch.from_numpy(imgs) * 2 - 1; T = torch.from_numpy(toks)
    G, D = Generator(), Discriminator()
    oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_data, (batch,)); real, tok = X[idx], T[idx]
        wrong = T[torch.randint(0, n_data, (batch,))]                     # mismatched text (mostly a different description)
        z = torch.randn(batch, G.nz); fake = G(z, tok)
        oD.zero_grad()
        ones, zeros = torch.full((batch,), 0.9), torch.zeros(batch)
        ld = (F.binary_cross_entropy_with_logits(D(real, tok), ones)
              + 0.5 * F.binary_cross_entropy_with_logits(D(fake.detach(), tok), zeros)
              + 0.5 * F.binary_cross_entropy_with_logits(D(real, wrong), zeros))
        ld.backward(); oD.step()
        oG.zero_grad()
        lg = F.binary_cross_entropy_with_logits(D(fake, tok), torch.ones(batch)); lg.backward(); oG.step()
        if step % 500 == 0 or step == 1:
            log(f"step {step:5d}  D {ld.item():.3f}  G {lg.item():.3f}  ({time.time() - t0:.0f}s)")
    G.eval()
    if ckpt: torch.save(G.state_dict(), ckpt)
    return G


@torch.no_grad()
def evaluate(G, holdout=HOLDOUT, per_prompt=4, seed=1):
    """Generate images for every description and read the attributes back off the pixels."""
    torch.manual_seed(seed)
    hold = set(holdout)
    res = {"seen": [], "held-out": []}
    for size, color, shape, pos in ALL:
        text = prompt(size, color, shape, pos)
        tok = torch.tensor([tokenize(text)] * per_prompt)
        img = ((G(torch.randn(per_prompt, G.nz), tok) + 1) / 2).clamp(0, 1).numpy()
        for im in img:
            a = attributes(im)
            want = (size, color, shape, pos)
            res["held-out" if (color, shape) in hold else "seen"].append(tuple(bool(a) and a[i] == want[i] for i in range(4)) if a else (False,) * 4)
    return {k: np.array(v).mean(0).tolist() + [float(np.array(v).all(1).mean())] for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--out", default="samples")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    G = train(a.steps, ckpt=os.path.join(a.out, "generator.pt"))
    r = evaluate(G)
    print(f"\nattribute accuracy read back from generated images ({len(ALL)} descriptions x 4 samples)")
    print(f"{'':<10}{'size':>7}{'color':>7}{'shape':>7}{'position':>10}{'ALL 4':>8}")
    for k, v in r.items():
        print(f"{k:<10}{v[0]:>7.0%}{v[1]:>7.0%}{v[2]:>7.0%}{v[3]:>10.0%}{v[4]:>8.0%}")
    print(f"(held-out = combinations never seen in training: {HOLDOUT})")


if __name__ == "__main__":
    main()

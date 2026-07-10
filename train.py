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


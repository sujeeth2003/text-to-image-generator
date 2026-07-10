"""Text-conditioned GAN (in the spirit of Reed et al. 2016, 'Generative Adversarial Text to Image Synthesis').

Text encoder : each word has a learned embedding; the sentence is the SUM of its word embeddings (order-free, so every
               word contributes independently, which is what lets 'blue' and 'triangle' combine in ways never seen together).
Generator    : [noise z ; text embedding] -> 4x4 -> 8x8 -> 16x16 -> 32x32 RGB (transposed convolutions, BatchNorm, ReLU, tanh).
Discriminator: image -> features; the text embedding is injected at 4x4 resolution; outputs real/fake.
               Trained on three kinds of pair (matching-aware): (real image, right text) = real,
               (fake image, text) = fake, and (real image, WRONG text) = fake. The third forces the discriminator (and hence
               the generator) to care whether the image actually matches the words, not just whether it looks realistic.
"""
import torch
from torch import nn

from .scene import VOCAB


class TextEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), dim, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(0.2))

    def forward(self, tok):                                   # (B, L) -> (B, dim)
        return self.proj(self.emb(tok).sum(1))


class Generator(nn.Module):
    def __init__(self, nz=64, td=64, ngf=96):
        super().__init__()
        self.enc, self.nz = TextEncoder(td), nz
        self.fc = nn.Sequential(nn.Linear(nz + td, ngf * 4 * 4 * 4), nn.BatchNorm1d(ngf * 4 * 4 * 4), nn.ReLU(True))
        self.ngf = ngf
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),   # 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),           # 16
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1), nn.Tanh())                                                       # 32

    def forward(self, z, tok):
        h = self.fc(torch.cat([z, self.enc(tok)], 1)).view(-1, self.ngf * 4, 4, 4)
        return self.net(h)                                    # [-1, 1]


class Discriminator(nn.Module):
    def __init__(self, td=64, ndf=64):
        super().__init__()
        self.enc = TextEncoder(td)
        self.conv = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),                                                  # 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),      # 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True))  # 4
        self.head = nn.Sequential(nn.Conv2d(ndf * 4 + td, ndf * 4, 3, 1, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 4, 1, 4, 1, 0))

    def forward(self, img, tok):
        f = self.conv(img)
        t = self.enc(tok)[:, :, None, None].expand(-1, -1, f.shape[2], f.shape[3])
        return self.head(torch.cat([f, t], 1)).view(-1)

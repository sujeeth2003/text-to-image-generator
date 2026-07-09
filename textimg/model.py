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


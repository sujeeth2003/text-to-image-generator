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


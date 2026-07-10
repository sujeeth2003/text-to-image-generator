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


import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from textimg.model import Discriminator, Generator  # noqa: E402
from textimg.scene import ALL, VOCAB, attributes, make_dataset, prompt, render, tokenize  # noqa: E402
from train import train  # noqa: E402



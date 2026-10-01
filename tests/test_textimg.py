import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from textimg.model import Discriminator, Generator  # noqa: E402
from textimg.scene import ALL, VOCAB, attributes, make_dataset, prompt, render, tokenize  # noqa: E402
from train import train  # noqa: E402


class SceneTests(unittest.TestCase):
    def test_attribute_extractor_reads_back_every_rendered_description(self):
        rng = np.random.default_rng(0)
        bad = []
        for size, color, shape, pos in ALL:
            for _ in range(3):
                got = attributes(render(size, color, shape, pos, rng))
                if got != (size, color, shape, pos): bad.append(((size, color, shape, pos), got))
        self.assertEqual(bad, [], f"{len(bad)} of {len(ALL) * 3} mismatches, e.g. {bad[:3]}")

    def test_tokenizer_and_unknown_words(self):
        ids = tokenize("a large red circle at top left")
        self.assertEqual(len(ids), 8); self.assertTrue(all(0 <= i < len(VOCAB) for i in ids))
        with self.assertRaises(ValueError) as e: tokenize("a purple dragon")
        self.assertIn("understands", str(e.exception))


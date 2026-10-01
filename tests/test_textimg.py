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

    def test_holdout_combinations_are_really_absent(self):
        _, toks = make_dataset(3000, holdout=[("blue", "triangle")], seed=1)
        blue, tri = VOCAB["blue"], VOCAB["triangle"]
        self.assertFalse(any(blue in t and tri in t for t in toks))
        self.assertTrue(any(blue in t for t in toks) and any(tri in t for t in toks))    # both concepts still seen separately

    def test_empty_image_has_no_attributes(self):
        self.assertIsNone(attributes(np.full((3, 32, 32), 0.05, np.float32)))



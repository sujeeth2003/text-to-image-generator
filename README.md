# Text-to-Image Generator (from words to pixels)

A small **text-conditioned GAN** that turns a sentence such as *"a large blue triangle at top left"* into a 32x32 image. Built from scratch in PyTorch to understand how words become pixels, and to test the question that matters for any text-to-image model: **does it compose concepts it has never seen together?**

![demo](samples/demo.png)

## The world it lives in
Every image shows one shape: **size** (small/large) x **colour** (6) x **shape** (circle, square, triangle, cross) x **position** (5), rendered procedurally with jitter and noise (`textimg/scene.py`), so the exact description of every training image is known. That gives an honest test.

## Model (`textimg/model.py`)
- **Text encoder:** each word has a learned embedding and the sentence is the **sum** of its word embeddings. Order-free by design, so "blue" and "triangle" each contribute independently, which is what makes new combinations possible.
- **Generator:** noise + text embedding -> 4x4 -> 8x8 -> 16x16 -> 32x32 RGB (transposed convolutions, BatchNorm, tanh).
- **Discriminator:** image features with the text injected at 4x4 resolution, trained **matching-aware** (Reed et al. 2016): (real image, right text) = real; (generated image, text) = fake; **(real image, WRONG text) = fake**. The third case is what forces the model to care whether the image *matches the words*, not merely whether it looks like a plausible image.

## Evaluation that does not trust my eyes (`train.py: evaluate`)
Generate an image for every one of the 240 descriptions (4 samples each), then **read the attributes back off the pixels** with an independent extractor (colour by nearest palette colour, position by centroid, size by extent, shape by fill ratio and width profile; verified to read back 100% of freshly rendered images, see tests). Four colour+shape pairs were **held out of training entirely**: blue triangle, red cross, green circle, yellow square.

```
             size  color  shape  position   ALL 4
seen         100%   100%    83%      100%     83%
held-out     100%   100%    67%      100%     67%
```
(6,000 training steps, batch 64, about 60 minutes on a laptop CPU; single run, so treat the last digit as noisy.)

What it shows:
- **Compositional generalisation works for colour, size and position:** 100% even for combinations never seen in training. In the demo, *"a large blue triangle at top left"* (a held-out pair) is a clean blue triangle.
- **Shape is the weak attribute**, and it degrades on held-out pairs (83% -> 67%). Shapes are harder to render than colours, especially the **small cross**, which comes out as a fuzzy blob; a 32x32 cross with 2-3 pixel arms is near the model's resolution limit.
- A GAN needed ~6,000 steps to get here; diffusion models typically train more stably, and would be the natural next step.

## Use
```bash
pip install torch numpy matplotlib
python -m unittest discover -s tests              # 7 tests (attribute extractor, tokenizer, held-out really absent, model behaviour, training smoke test)
python train.py --steps 6000                      # trains, evaluates, writes samples/generator.pt
python generate.py "a large blue triangle at top left" --n 8 --out blue.png
python generate.py --demo
```
Only the words in its vocabulary work (`a`, `at`, sizes, six colours, four shapes, position words); anything else raises an error that lists what it understands. The 9 MB weights are not committed (retrain, or attach them to a GitHub Release).

## Limits
32x32 images of one object; a closed vocabulary, not natural language; the GAN sometimes blurs small shapes. It demonstrates the mechanism (embed words, condition a generator, train it to respect the text, evaluate composition), not a general text-to-image system.

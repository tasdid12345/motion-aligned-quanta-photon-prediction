# Motion-aligned photon prediction on simulated 1-bit quanta videos

## Motivation
The paper "bit2bit: 1-bit quanta video reconstruction via self-supervised photon prediction" shows that videos can be reconstructed from single-bit quanta sensor frames by learning to predict photons from neighbouring bits, without clean targets. This small experiment asks whether giving such a predictor motion-aligned temporal information helps compared with a plain fixed spatiotemporal window.

## Question
Can a self-supervised photon predictor with motion-aligned temporal features reconstruct simulated 1-bit quanta videos better than the same predictor using a plain fixed spatiotemporal window, especially under fast motion (2-4 px/frame) and dim light?

## Dataset
Simulated, not the paper's data. Photographs bundled with scikit-learn (`load_sample_images`): translating grayscale crops (0.5-4 px/frame) with bits drawn as Bernoulli(1-exp(-flux)) at several dim-to-moderate flux levels. Training videos come from `china.jpg`, held-out test videos from `flower.jpg`.

## Baseline vs improved
- **Baseline:** LightGBM predicts a pixel's bit from a fixed 5x5x5 window of bits with the centre hidden (a stand-in for photon prediction, not the paper's network); the probability is inverted to an intensity estimate.
- **Improved:** the same learner, data, grid and budget, plus features from neighbouring frames shifted by a global motion estimated from binned-frame cross-correlation, summed over several temporal scales.

Main metric: PSNR against noise-free test frames; secondary: PSNR on fast-motion and lowest-light test subsets. Hyper-parameters are chosen by video-grouped cross-validation on training data only; the test set is used once at the end; three LightGBM seeds (42-44).

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> Y Liu, A Krull, H Basevi, A Leonardis (2024). *bit2bit: 1-bit quanta video reconstruction via self-supervised photon prediction*. Advances in Neural. https://proceedings.neurips.cc/paper_files/paper/2024/hash/a128e43ed4708862aec51549b0050fc6-Abstract-Conference.html

Paper author: Prof. Ales Leonardis, University of Birmingham (a.leonardis@bham.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

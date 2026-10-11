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

## Results (latest run)

On simulated 1-bit quanta videos from scikit-learn sample photos, a fixed-window LightGBM photon predictor reached 16.16 dB PSNR, versus 16.78 dB with motion-aligned features: a modest 0.62 dB overall gain, rising to 1.47 dB in dim light.

| Metric | Value |
|---|---|
| baseline_score | 16.16 |
| baseline_score_std | 0.02173 |
| baseline_psnr_fast | 15.27 |
| baseline_psnr_dim | 10.29 |
| baseline_psnr_fast_dim | 9.8 |
| improved_score | 16.78 |
| improved_score_std | 0.02669 |
| improved_psnr_fast | 15.81 |
| improved_psnr_dim | 11.77 |
| improved_psnr_fast_dim | 11.59 |
| improvement | 0.6214 |
| improvement_fast | 0.5452 |
| improvement_dim | 1.472 |
| motion_mae_px_per_frame_test | 1.4 |
| motion_mae_px_per_frame_test_dim | 1.737 |

Run on 2026-10-11. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Test set is a different photograph (all test videos unseen scenes), not a random row split; rows within videos are correlated. Sensor flux scale alpha is assumed known when inverting probabilities. Motion estimated from the bits of each video (including the target frames; effect on a global estimate is negligible). Global translation only; the bundled photos are low-resolution stand-ins for real scenes; num_leaves chosen by 3-fold video-grouped CV on training only; PSNR averaged over 24 test videos and 3 LightGBM seeds (data fixed at seed 42). Single simulated dataset: results may not generalise to real SPAD data.

## Reference

This project is motivated by the following paper:

> Y Liu, A Krull, H Basevi, A Leonardis (2024). *bit2bit: 1-bit quanta video reconstruction via self-supervised photon prediction*. Advances in Neural. https://proceedings.neurips.cc/paper_files/paper/2024/hash/a128e43ed4708862aec51549b0050fc6-Abstract-Conference.html

Paper author: Prof. Ales Leonardis, University of Birmingham (a.leonardis@bham.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

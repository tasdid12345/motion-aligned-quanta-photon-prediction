"""bit2bit-style test: fixed 5x5x5 bit window vs. + motion-aligned temporal features (LightGBM)."""
import json, os, sys, time
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.datasets import load_sample_images
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

SEED = 42
T, H, W, M, K, CAN = 16, 96, 96, 18, 4, 240   # frames, frame size, spatial margin, max temporal reach
T0, T1 = K, T - K                              # frames used as prediction targets
SPEEDS, ALPHAS = [0.5, 1.0, 2.0, 4.0], [0.05, 0.2, 0.8]   # px/frame; mean photons/pixel/frame scale
N_BASE = 124                                   # 5x5x5 window minus hidden centre
ROWS_PER_VIDEO = 6000
t_start = time.time()

# ---------- simulation ----------
def make_video(img, alpha, speed, rng):
    """Translate a crop of `img`; return 1-bit frames, noise-free frames (in [0,1]), true per-frame motion."""
    y0, x0 = rng.integers(0, img.shape[0] - CAN + 1), rng.integers(0, img.shape[1] - CAN + 1)
    canvas = img[y0:y0 + CAN, x0:x0 + CAN]
    th = rng.uniform(0, 2 * np.pi)
    d = speed * np.array([np.sin(th), np.cos(th)])
    c = (CAN - H) // 2
    clean = np.stack([ndimage.shift(canvas, (-d[0] * t, -d[1] * t), order=1)[c:c + H, c:c + W] for t in range(T)])
    bits = (rng.random(clean.shape) < 1 - np.exp(-alpha * clean)).astype(np.uint8)
    return bits, clean, -d

def make_set(img, n_crops, rng):
    out = []
    for s in SPEEDS:
        for a in ALPHAS:
            for _ in range(n_crops):
                out.append((*make_video(img, a, s, rng), s, a))
    return out

# ---------- motion estimation (bits only): cross-correlate temporally binned frames ----------
def estimate_motion(bits, B=4):
    bins = bits.reshape(T // B, B, H, W).sum(1).astype(float)
    bins -= bins.mean(axis=(1, 2), keepdims=True)
    F = np.fft.fft2(bins)
    corr = sum(np.fft.ifft2(np.conj(F[g]) * F[g + 1]).real for g in range(len(F) - 1))
    p = np.array(np.unravel_index(np.argmax(corr), corr.shape), float)
    p = np.where(p > np.array([H, W]) / 2, p - np.array([H, W]), p)
    return np.clip(p / B, -4, 4)

# ---------- features: first N_BASE columns = baseline, all columns = improved ----------
def features(bits, v):
    sl = lambda k, sy, sx: bits[T0 + k:T1 + k, M + sy:H - M + sy, M + sx:W - M + sx]
    base = [sl(dt, dy, dx) for dt in range(-2, 3) for dy in range(-2, 3) for dx in range(-2, 3)
            if (dt, dy, dx) != (0, 0, 0)]
    ks = [k for k in range(-K, K + 1) if k]
    sh = {k: (int(np.round(k * v[0])), int(np.round(k * v[1]))) for k in ks}   # motion-aligned shift
    al = lambda k, oy, ox: sl(k, sh[k][0] + oy, sh[k][1] + ox)
    # 3x3 aligned neighbourhood summed over several temporal scales (|k|<=1,2,4)
    sums = [sum(al(k, oy, ox) for k in ks if abs(k) <= r) for oy in (-1, 0, 1) for ox in (-1, 0, 1) for r in (1, 2, 4)]
    cent = [al(k, 0, 0) for k in ks]            # aligned centre-pixel bit at each offset
    X = np.stack(base + sums + cent, -1).astype(np.uint8)
    y = bits[T0:T1, M:H - M, M:W - M]
    return X.reshape(-1, X.shape[-1]), y.reshape(-1)

def prep(videos):
    rows = []
    for bits, clean, v, s, a in videos:
        X, y = features(bits, estimate_motion(bits))
        rows.append(dict(X=X, y=y, truth=clean[T0:T1, M:H - M, M:W - M].reshape(-1), s=s, a=a,
                         err=float(np.abs(estimate_motion(bits) - v).mean())))
    return rows

def new_model(nl, seed):
    return lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=nl, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)

# ---------- data ----------
ds = load_sample_images()
imgs = {os.path.basename(f): im.astype(float).mean(-1) / 255 for f, im in zip(ds.filenames, ds.images)}
rng = np.random.default_rng(SEED)
print("simulating videos")
train = prep(make_set(imgs["china.jpg"], 2, rng))     # scene 1: training
test = prep(make_set(imgs["flower.jpg"], 2, rng))     # scene 2: held-out test
srng = np.random.default_rng(SEED)
Xtr, ytr, gtr = [], [], []
for i, r in enumerate(train):
    idx = srng.choice(len(r["y"]), ROWS_PER_VIDEO, replace=False)
    Xtr.append(r["X"][idx]); ytr.append(r["y"][idx]); gtr.append(np.full(ROWS_PER_VIDEO, i))
Xtr, ytr, gtr = np.concatenate(Xtr).astype(np.float32), np.concatenate(ytr), np.concatenate(gtr)
n_test_rows = sum(len(r["y"]) for r in test)
print(f"train rows {len(ytr)}, test rows {n_test_rows}, features {Xtr.shape[1]} ({time.time() - t_start:.0f}s)")

# ---------- CV (grouped by training video) to select num_leaves, same grid for both ----------
def select(cols):
    best = None
    for nl in (15, 63):
        losses = []
        for a, b in GroupKFold(3).split(Xtr, ytr, gtr):
            m = new_model(nl, SEED).fit(Xtr[a][:, cols], ytr[a])
            losses.append(log_loss(ytr[b], m.predict_proba(Xtr[b][:, cols])[:, 1]))
        if best is None or np.mean(losses) < best[0]:
            best = (np.mean(losses), nl)
    return best[1]

def evaluate(model, cols):
    """Per-video PSNR of the flux->intensity estimate against noise-free frames."""
    out = []
    for r in test:
        p = np.clip(model.predict_proba(r["X"][:, cols].astype(np.float32))[:, 1], 1e-4, 1 - 1e-4)
        est = np.clip(-np.log1p(-p) / r["a"], 0, 1)     # invert p = 1-exp(-alpha*I)
        out.append(10 * np.log10(1 / np.mean((est - r["truth"]) ** 2)))
    return np.array(out)

fast = np.array([r["s"] >= 2 for r in test]); dim = np.array([r["a"] == min(ALPHAS) for r in test])
res, leaves = {}, {}
for name, cols in [("baseline", np.arange(N_BASE)), ("improved", np.arange(Xtr.shape[1]))]:
    leaves[name] = select(cols)
    print(f"{name}: CV-selected num_leaves={leaves[name]} ({time.time() - t_start:.0f}s)")
    res[name] = np.array([evaluate(new_model(leaves[name], sd).fit(Xtr[:, cols], ytr), cols) for sd in (42, 43, 44)])
    print(f"{name}: mean PSNR {res[name].mean():.3f}")

# ---------- results ----------
metrics = {}
for name, P in res.items():
    for key, mask in [("score", slice(None)), ("psnr_fast", fast), ("psnr_dim", dim), ("psnr_fast_dim", fast & dim)]:
        per_seed = P[:, mask].mean(1)
        metrics[f"{name}_{key}"] = float(per_seed.mean())
        if key == "score":
            metrics[f"{name}_score_std"] = float(per_seed.std())
metrics["improvement"] = metrics["improved_score"] - metrics["baseline_score"]
metrics["improvement_fast"] = metrics["improved_psnr_fast"] - metrics["baseline_psnr_fast"]
metrics["improvement_dim"] = metrics["improved_psnr_dim"] - metrics["baseline_psnr_dim"]
metrics["motion_mae_px_per_frame_test"] = float(np.mean([r["err"] for r in test]))
metrics["motion_mae_px_per_frame_test_dim"] = float(np.mean([r["err"] for r, d in zip(test, dim) if d]))
if not all(np.isfinite(v) for v in metrics.values()):
    sys.exit("non-finite metrics")

os.makedirs("figures", exist_ok=True)
labels = ["all", "fast (>=2 px/f)", "dimmest", "fast & dimmest"]
keys = ["score", "psnr_fast", "psnr_dim", "psnr_fast_dim"]
x = np.arange(4)
plt.bar(x - .2, [metrics[f"baseline_{k}"] for k in keys], .4, label="baseline 5x5x5")
plt.bar(x + .2, [metrics[f"improved_{k}"] for k in keys], .4, label="+ motion-aligned")
plt.xticks(x, labels); plt.ylabel("test PSNR (dB)"); plt.legend(); plt.tight_layout()
plt.savefig("figures/psnr.png", dpi=120)

json.dump({
    "question": "Can a self-supervised photon predictor with motion-aligned temporal features reconstruct simulated 1-bit quanta videos better than the same predictor with a plain fixed spatiotemporal window, especially under fast motion and dim light?",
    "method": "Simulated 1-bit SPAD videos (translating crops of sklearn sample photos, Bernoulli(1-exp(-flux))); LightGBM predicts each pixel's bit from neighbouring bits with the centre hidden; predicted probability inverted to intensity; PSNR vs noise-free frames on held-out scene.",
    "dataset": {"name": "Simulated quanta videos from scikit-learn sample images (china.jpg train, flower.jpg test)",
                "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_images.html",
                "n_rows": int(len(ytr) + n_test_rows), "n_train": int(len(ytr)), "n_test": int(n_test_rows),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "PSNR (dB) of reconstructed intensity vs noise-free test frames",
    "higher_is_better": True,
    "baseline": "LightGBM self-supervised photon predictor on a fixed 5x5x5 bit window (centre hidden); stand-in, not the paper's network.",
    "improved": "Same learner/budget/hyper-parameter grid, plus features from frames aligned with a global motion estimated by binned-frame cross-correlation, summed over several temporal scales.",
    "metrics": metrics,
    "notes": "Test set is a different photograph (all test videos unseen scenes), not a random row split; rows within videos are correlated. Sensor flux scale alpha is assumed known when inverting probabilities. Motion estimated from the bits of each video (including the target frames; effect on a global estimate is negligible). Global translation only; the bundled photos are low-resolution stand-ins for real scenes; num_leaves chosen by 3-fold video-grouped CV on training only; PSNR averaged over 24 test videos and 3 LightGBM seeds (data fixed at seed 42). Single simulated dataset: results may not generalise to real SPAD data.",
}, open("results.json", "w", encoding="utf-8"), indent=2)
print(f"done in {time.time() - t_start:.0f}s")

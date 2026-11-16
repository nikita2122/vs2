"""
Numerical study for a computer-science thesis:

  operating-status identification from a designed sensor placement,
  and a deep network that maps those features to commands which raise
  grain-recovery efficiency while protecting mechanical stability.

The plant is a reduced-order axial thresher used only as a simulator.
Commands are four normalized setpoints. Rotor hardware details are not
part of the learning method.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SEED = 7
RNG = np.random.default_rng(SEED)

N_STATION = 8
XI = np.linspace(0.0, 1.0, N_STATION)
F_BLOCK = 1680.0

# crop: phi, difficulty, load scale
CROPS = {
    "wheat": np.array([0.46, 1.00, 1.00]),
    "rice": np.array([0.41, 1.28, 1.12]),
    "soybean": np.array([0.55, 0.86, 0.92]),
}
CROP_NAMES = ("wheat", "rice", "soybean")
CROP_PAR = np.stack([CROPS[n] for n in CROP_NAMES], axis=0)

STATUS_NAMES = (
    "stable",
    "front-loaded",
    "under-filled",
    "blockage-risk",
    "high-loss",
)

# Seasonal mix. Weights sum to 1.
POINTS = [
    ("Wheat, nominal", "wheat", 6.0, 0.145, 0.22),
    ("Wheat, wet", "wheat", 6.0, 0.220, 0.14),
    ("Wheat, dry heavy", "wheat", 8.0, 0.120, 0.12),
    ("Wheat, light", "wheat", 3.5, 0.160, 0.10),
    ("Rice, nominal", "rice", 5.0, 0.180, 0.14),
    ("Rice, wet", "rice", 6.4, 0.250, 0.12),
    ("Soybean, nominal", "soybean", 4.5, 0.130, 0.10),
    ("Soybean, wet", "soybean", 4.2, 0.180, 0.06),
]


def plant(u: np.ndarray, q: np.ndarray, m: np.ndarray, crop: np.ndarray) -> dict:
    """Vectorized reduced-order plant.

    u: (n, 4) intensity, residence, cleaning, feed admission, each in [0, 1].
    q, m: (n,) fresh biomass feed and moisture.
    crop: (n,) integer crop index.
    """
    u = np.asarray(u, dtype=float)
    q = np.asarray(q, dtype=float)
    m = np.asarray(m, dtype=float)
    crop = np.asarray(crop, dtype=int)
    ui, ur, uc, uf = u.T
    phi = CROP_PAR[crop, 0]
    diff = CROP_PAR[crop, 1]
    load_s = CROP_PAR[crop, 2]

    # Feed admission is a trim, not a shutdown. qe/q stays in [0.82, 1].
    qe = q * (0.82 + 0.18 * uf)
    wet = m - 0.15
    qn = np.clip(qe / 6.0, 0.35, 1.8)
    # Packing is an interaction: it appears only when feed and moisture are both high,
    # and a larger residence command relieves it. A linear function of the two
    # scalars cannot represent that product.
    pack = np.maximum(qn - 0.75, 0.0) * np.maximum(m - 0.16, 0.0) / 0.08

    lu = (
        0.058
        * diff
        * np.exp(-3.40 * ui)
        * np.exp(-0.85 * ur)
        * np.exp(6.4 * wet)
        * qn ** 0.58
        * np.exp(0.70 * pack * np.maximum(0.80 - ur, 0.0))
    )
    lg = (
        0.026
        * diff
        * np.exp(-2.05 * ur)
        * np.exp(-0.30 * ui)
        * np.exp(4.0 * wet)
        * qn ** 0.46
        * np.exp(0.50 * pack * np.maximum(0.78 - ur, 0.0))
    )
    ls = (
        0.0125
        * np.exp(-2.05 * (uc - 0.18))
        * np.exp(2.8 * np.maximum(m - 0.18, 0.0))
        * qn ** 0.58
        * (1.08 + 0.28 * (1.0 - ui))
    )
    iota = (
        0.0108
        * np.exp(1.70 * (uc - 0.40))
        * np.exp(-0.45 * (ui - 0.50))
        * (0.80 + 1.05 * m)
    )
    loss = lu + lg + ls
    # Breakage rises with residence and intensity, and it rises faster once the
    # grain is wet. It is the price of holding a difficult crop in the machine.
    breakage = (
        0.0034
        * diff
        * np.exp(1.90 * ur)
        * np.exp(0.38 * ui)
        * np.exp(2.7 * np.maximum(m - 0.18, 0.0))
    )
    # Recovery is grain saved over grain offered. Cutting the feed lowers it.
    eta = (qe / np.maximum(q, 1e-6)) * (1.0 - loss)

    e = (
        (0.58 + 0.72 * ui + 0.62 * ur)
        * (1.0 + 2.15 * (m - 0.14))
        * (0.90 + 0.16 * diff)
        * (0.94 + 0.08 * qn)
    )

    sf = 1.55 - 0.58 * ur + 0.42 * pack * np.maximum(1.0 - ur, 0.0)
    sr = 0.70 + 0.34 * ur
    shape = sf[:, None] + (sr - sf)[:, None] * XI[None, :]
    curve = 1.0 + (0.16 * pack * np.maximum(1.0 - ur, 0.0))[:, None] * (XI[None, :] - 0.55) ** 2
    F = (
        720.0
        * (qe / 6.0)[:, None]
        * (1.0 + 1.85 * (m - 0.14))[:, None]
        * (0.86 + 0.28 * ui)[:, None]
        * shape
        * curve
        * load_s[:, None]
    )
    fmax = F.max(axis=1)
    fmean = F.mean(axis=1)
    fstd = F.std(axis=1)
    cv = fstd / np.maximum(fmean, 1.0)
    # Housing vibration sees intensity, moisture, and mean throughput.
    # It does not see which station is hot. That spatial fact is left to the load cells.
    vib = (
        (0.20 + 0.88 * np.maximum(ui, 0.0) ** 1.30)
        * (1.0 + 3.4 * np.maximum(m - 0.20, 0.0))
        * (0.85 + 0.20 * qn)
    )
    cv_n = np.clip(cv / 0.42, 0.0, 1.35)
    v_n = np.clip(vib / 1.55, 0.0, 1.35)
    m_n = np.clip((fmax / F_BLOCK - 0.48) / 0.62, 0.0, 1.35)
    stab = np.clip(1.0 - 0.40 * cv_n - 0.32 * v_n - 0.28 * m_n, 0.0, 1.0)

    torque = 0.0042 * F.sum(axis=1)  # shaft torque, a scalar; it does not locate the peak
    # kW = (kWh/t) * (kg/s) * (t/1000 kg) * (3600 s/h) = e * q_grain * 3.6
    power = e * (qe * phi) * 3.6
    ret = (0.045 + 0.85 * ls + 0.15 * lg) * (qe * phi)

    return {
        "lu": lu,
        "lg": lg,
        "ls": ls,
        "loss": loss,
        "eta": eta,
        "iota": iota,
        "e": e,
        "F": F,
        "cv": cv,
        "fmax": fmax,
        "fmean": fmean,
        "vib": vib,
        "stab": stab,
        "torque": torque,
        "power": power,
        "return": ret,
        "qe": qe,
        "phi": phi,
        "breakage": breakage,
    }


def objective_from_metrics(met: dict) -> np.ndarray:
    """Scalar to maximize.

    Recovery is grain saved over grain offered. Energy, impurity, load
    nonuniformity, and a blockage hinge are prices, so the optimizer cannot
    buy a higher recovery fraction by refusing crop or by idling the rotor.
    """
    return (
        met["eta"]
        - 0.012 * met["e"]
        - 0.48 * met["iota"]
        - 1.15 * met["breakage"]
        - 0.025 * met["cv"]
        - 0.26 * np.maximum(0.0, 0.70 - met["stab"])
        - 0.40 * np.maximum(0.0, met["fmax"] / F_BLOCK - 0.82) ** 2
    )


def status_from_metrics(met: dict) -> np.ndarray:
    """Priority labeling of operating status. Spatial classes need the load shape."""
    F = met["F"]
    contrast = (F[:, 0] - F[:, -1]) / np.maximum(met["fmean"], 1.0)
    y = np.full(F.shape[0], 0, dtype=int)  # stable
    y[met["fmean"] <= 430.0] = 2  # under-filled
    y[(contrast >= 0.38) & (met["fmax"] >= 760.0)] = 1  # front-loaded
    y[met["loss"] >= 0.028] = 4  # high-loss
    y[met["fmax"] >= 0.84 * F_BLOCK] = 3  # blockage-risk
    return y


def clip_u(u: np.ndarray) -> np.ndarray:
    return np.clip(u, 0.0, 1.0)


def schedule_u(m: np.ndarray) -> np.ndarray:
    """Moisture card used as the non-learning baseline.

    It raises intensity, residence, and cleaning when the grain is wetter.
    It does not read feed rate or the axial load shape, and it leaves the
    feed trim fully open.
    """
    m = np.asarray(m, dtype=float)
    ui = np.clip(0.50 + 1.25 * (m - 0.15), 0.38, 0.82)
    ur = np.clip(0.46 + 1.15 * (m - 0.15), 0.32, 0.80)
    uc = np.clip(0.47 + 0.85 * (m - 0.16), 0.36, 0.70)
    uf = np.full(np.shape(m), 1.0)
    return np.stack([ui, ur, uc, uf], axis=1)


def fixed_u(n: int) -> np.ndarray:
    u = np.array([0.50, 0.44, 0.46, 1.00])
    return np.repeat(u[None, :], n, axis=0)


def oracle_one(q: float, m: float, crop: int, n_starts: int = 5) -> np.ndarray:
    def fun(u):
        met = plant(u[None, :], np.array([q]), np.array([m]), np.array([crop]))
        return -float(objective_from_metrics(met)[0])

    starts = [schedule_u(np.array([m]))[0], fixed_u(1)[0]]
    extra = RNG.uniform(0.15, 0.9, size=(n_starts, 4))
    starts.extend(list(extra))
    best_u = starts[0]
    best_f = fun(best_u)
    for u0 in starts:
        res = minimize(fun, u0, method="L-BFGS-B", bounds=[(0, 1)] * 4)
        if res.success and res.fun < best_f:
            best_f = float(res.fun)
            best_u = res.x
    return clip_u(best_u)


def oracle_batch(q, m, crop, n_starts: int = 4) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    m = np.asarray(m, dtype=float)
    crop = np.asarray(crop, dtype=int)
    n = q.shape[0]
    # coarse grid, vectorized, then one local polish from the best cell
    grid_1d = np.linspace(0.15, 0.90, 4)
    mesh = np.stack(np.meshgrid(*([grid_1d] * 4), indexing="ij"), axis=-1).reshape(-1, 4)
    g = mesh.shape[0]
    out = np.zeros((n, 4))
    for i in range(n):
        qq = np.full(g, q[i])
        mm = np.full(g, m[i])
        cc = np.full(g, crop[i])
        met = plant(mesh, qq, mm, cc)
        j = objective_from_metrics(met)
        u0 = mesh[int(np.argmax(j))]
        def fun(u, qi=q[i], mi=m[i], ci=crop[i]):
            met_i = plant(u[None, :], np.array([qi]), np.array([mi]), np.array([ci]))
            return -float(objective_from_metrics(met_i)[0])

        res = minimize(fun, u0, method="L-BFGS-B", bounds=[(0, 1)] * 4)
        out[i] = clip_u(res.x if res.success else u0)
        if (i + 1) % 400 == 0:
            print(f"  oracle {i+1}/{n}", flush=True)
    return out


def sample_conditions(n: int, rng: np.random.Generator):
    crop = rng.integers(0, 3, size=n)
    q = rng.uniform(3.3, 8.3, size=n)
    m = rng.uniform(0.115, 0.255, size=n)
    # enrich the wet-heavy and light corners so every status appears
    k = n // 5
    q[:k] = rng.uniform(6.2, 8.4, size=k)
    m[:k] = rng.uniform(0.20, 0.26, size=k)
    q[k : 2 * k] = rng.uniform(3.3, 4.3, size=k)
    m[k : 2 * k] = rng.uniform(0.12, 0.20, size=k)
    return q, m, crop


def observe(met: dict, rng: np.random.Generator, n_window: int = 8) -> dict:
    """Draw a short noisy window around a steady operating point."""
    n = met["F"].shape[0]
    F = met["F"][:, None, :] + rng.normal(0.0, 22.0, size=(n, n_window, N_STATION))
    torque = met["torque"][:, None] + rng.normal(0.0, 0.015, size=(n, n_window))
    vib = np.maximum(0.0, met["vib"][:, None] + rng.normal(0.0, 0.03, size=(n, n_window)))
    loss_p = np.maximum(0.0, met["ls"][:, None] + rng.normal(0.0, 0.0015, size=(n, n_window)))
    ret = np.maximum(0.0, met["return"][:, None] + rng.normal(0.0, 0.01, size=(n, n_window)))
    power = np.maximum(0.0, met["power"][:, None] + rng.normal(0.0, 0.12, size=(n, n_window)))
    moist = met_m_from(met, n) if False else None
    return {"F": F, "torque": torque, "vib": vib, "loss": loss_p, "return": ret, "power": power}


def met_m_from(met, n):
    return None


def features_from_windows(windows: dict, m, q, placement: np.ndarray, rng) -> np.ndarray:
    """Window statistics. Placement selects load stations. Scalars are always present."""
    F = windows["F"][:, :, placement]  # (n, W, K)
    mu_f = F.mean(axis=1)
    sd_f = F.std(axis=1)
    mu_all = mu_f.mean(axis=1, keepdims=True)
    sd_all = mu_f.std(axis=1, keepdims=True)
    cv = sd_all / np.maximum(mu_all, 1.0)
    contrast = (mu_f[:, :1] - mu_f[:, -1:]) / np.maximum(mu_all, 1.0)
    fmax = mu_f.max(axis=1, keepdims=True)
    parts = [
        mu_f / 1000.0,
        sd_f / 100.0,
        cv,
        contrast,
        fmax / 1000.0,
        windows["torque"].mean(axis=1, keepdims=True),
        windows["torque"].std(axis=1, keepdims=True),
        windows["vib"].mean(axis=1, keepdims=True),
        windows["vib"].std(axis=1, keepdims=True),
        windows["loss"].mean(axis=1, keepdims=True) * 100.0,
        windows["loss"].std(axis=1, keepdims=True) * 100.0,
        windows["return"].mean(axis=1, keepdims=True),
        windows["return"].std(axis=1, keepdims=True),
        windows["power"].mean(axis=1, keepdims=True) / 10.0,
        windows["power"].std(axis=1, keepdims=True),
        np.asarray(m)[:, None],
        np.asarray(q)[:, None] / 10.0,
    ]
    # moisture and feed are measured with their own noise
    parts[-2] = parts[-2] + rng.normal(0.0, 0.004, size=parts[-2].shape)
    parts[-1] = parts[-1] + rng.normal(0.0, 0.012, size=parts[-1].shape)
    return np.concatenate(parts, axis=1)


def fisher_det(placement: tuple[int, ...], order: int | None = None) -> float:
    """D-criterion for a polynomial load shape along the axis.

    order 1 is the line θ0 + θ1 ξ. order 2 is the quadratic
    θ0 + θ1 ξ + θ2 ξ^2. A quadratic is singular with fewer than three stations,
    so a two-station design is scored on the line.
    """
    if order is None:
        order = 1 if len(placement) < 3 else 2
    cols = []
    for i in placement:
        xi = float(XI[i])
        row = [1.0, xi] if order == 1 else [1.0, xi, xi * xi]
        cols.append(row)
    phi = np.asarray(cols, dtype=float)
    info = phi.T @ phi
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0:
        return 0.0
    npar = info.shape[0]
    return float(np.exp(logdet / npar))


def best_d_optimal(k: int) -> tuple[tuple[int, ...], float]:
    from itertools import combinations

    order = 1 if k < 3 else 2
    best_p, best_v = None, -1.0
    for comb in combinations(range(N_STATION), k):
        v = fisher_det(comb, order=order)
        if v > best_v:
            best_p, best_v = comb, v
    return best_p, best_v


def softmax_fit(X, y, n_class=5, epochs=60, lr=0.08, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(0, 0.02, size=(d, n_class))
    b = np.zeros(n_class)
    Y = np.eye(n_class)[y]
    idx = np.arange(n)
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, 128):
            j = idx[s : s + 128]
            logits = X[j] @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            g = (P - Y[j]) / max(len(j), 1)
            W -= lr * (X[j].T @ g + l2 * W)
            b -= lr * g.sum(axis=0)
    return W, b


def softmax_predict(X, W, b):
    logits = X @ W + b
    return logits.argmax(axis=1), logits


def accuracy(yhat, y):
    return float((yhat == y).mean())


def macro_f1(yhat, y, n_class=5):
    scores = []
    for c in range(n_class):
        tp = np.sum((yhat == c) & (y == c))
        fp = np.sum((yhat == c) & (y != c))
        fn = np.sum((yhat != c) & (y == c))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        scores.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(scores))


def per_class_recall(yhat, y, n_class=5):
    out = []
    for c in range(n_class):
        mask = y == c
        out.append(float((yhat[mask] == c).mean()) if mask.any() else float("nan"))
    return out


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd


def apply_std(X, mu, sd):
    return (X - mu) / sd


class MLP:
    def __init__(self, sizes, seed=1):
        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        for a, c in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2.0 / a), size=(a, c)))
            self.b.append(np.zeros(c))

    def forward(self, X):
        hs = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            if i < len(self.W) - 1:
                hs.append(np.maximum(z, 0.0))
            else:
                hs.append(z)
        return hs

    def predict(self, X):
        return self.forward(X)[-1]


def train_mlp_regression(X, Y, Xv, Yv, sizes, epochs=160, lr=1e-3, batch=64, seed=1):
    net = MLP(sizes, seed=seed)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    best = None
    best_v = 1e9
    m_w = [np.zeros_like(W) for W in net.W]
    m_b = [np.zeros_like(b) for b in net.b]
    v_w = [np.zeros_like(W) for W in net.W]
    v_b = [np.zeros_like(b) for b in net.b]
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            j = idx[s : s + batch]
            hs = net.forward(X[j])
            delta = (hs[-1] - Y[j]) / len(j)
            step += 1
            for li in reversed(range(len(net.W))):
                h = hs[li]
                gW = h.T @ delta
                gb = delta.sum(axis=0)
                m_w[li] = b1 * m_w[li] + (1 - b1) * gW
                m_b[li] = b1 * m_b[li] + (1 - b1) * gb
                v_w[li] = b2 * v_w[li] + (1 - b2) * (gW * gW)
                v_b[li] = b2 * v_b[li] + (1 - b2) * (gb * gb)
                mw = m_w[li] / (1 - b1 ** step)
                mb = m_b[li] / (1 - b1 ** step)
                vw = v_w[li] / (1 - b2 ** step)
                vb = v_b[li] / (1 - b2 ** step)
                net.W[li] -= lr * mw / (np.sqrt(vw) + eps)
                net.b[li] -= lr * mb / (np.sqrt(vb) + eps)
                if li > 0:
                    delta = (delta @ net.W[li].T) * (h > 0)
        pred = net.predict(Xv)
        verr = float(np.mean((pred - Yv) ** 2))
        if verr < best_v:
            best_v = verr
            best = (
                [W.copy() for W in net.W],
                [b.copy() for b in net.b],
                ep,
                verr,
            )
    net.W, net.b, best_ep, best_v = best
    return net, best_ep, best_v


def train_mlp_classifier(X, y, Xv, yv, sizes, epochs=140, lr=1e-3, batch=64, seed=2):
    net = MLP(sizes, seed=seed)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_class = sizes[-1]
    Y = np.eye(n_class)[y]
    best = None
    best_v = -1.0
    m_w = [np.zeros_like(W) for W in net.W]
    m_b = [np.zeros_like(b) for b in net.b]
    v_w = [np.zeros_like(W) for W in net.W]
    v_b = [np.zeros_like(b) for b in net.b]
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            j = idx[s : s + batch]
            hs = net.forward(X[j])
            logits = hs[-1]
            logits = logits - logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            delta = (P - Y[j]) / len(j)
            step += 1
            for li in reversed(range(len(net.W))):
                h = hs[li]
                gW = h.T @ delta
                gb = delta.sum(axis=0)
                m_w[li] = b1 * m_w[li] + (1 - b1) * gW
                m_b[li] = b1 * m_b[li] + (1 - b1) * gb
                v_w[li] = b2 * v_w[li] + (1 - b2) * (gW * gW)
                v_b[li] = b2 * v_b[li] + (1 - b2) * (gb * gb)
                mw = m_w[li] / (1 - b1 ** step)
                mb = m_b[li] / (1 - b1 ** step)
                vw = v_w[li] / (1 - b2 ** step)
                vb = v_b[li] / (1 - b2 ** step)
                net.W[li] -= lr * mw / (np.sqrt(vw) + eps)
                net.b[li] -= lr * mb / (np.sqrt(vb) + eps)
                if li > 0:
                    delta = (delta @ net.W[li].T) * (h > 0)
        pred = net.predict(Xv).argmax(axis=1)
        vacc = accuracy(pred, yv)
        if vacc > best_v:
            best_v = vacc
            best = ([W.copy() for W in net.W], [b.copy() for b in net.b], ep, vacc)
    net.W, net.b, best_ep, best_v = best
    return net, best_ep, best_v


def count_params(net: MLP) -> int:
    return int(sum(W.size + b.size for W, b in zip(net.W, net.b)))


def summarize(name, met):
    return {
        "name": name,
        "loss_pct": float(met["loss"] * 100),
        "lu_pct": float(met["lu"] * 100),
        "lg_pct": float(met["lg"] * 100),
        "ls_pct": float(met["ls"] * 100),
        "eta_pct": float(met["eta"] * 100),
        "iota_pct": float(met["iota"] * 100),
        "e": float(met["e"]),
        "cv": float(met["cv"]),
        "fmax": float(met["fmax"]),
        "stab": float(met["stab"]),
        "vib": float(met["vib"]),
        "power": float(met["power"]),
    }


def policy_metrics(u, q, m, crop):
    return plant(np.asarray(u, dtype=float), np.asarray(q, dtype=float), np.asarray(m, dtype=float), np.asarray(crop, dtype=int))


def main():
    out_dir = Path(__file__).resolve().parent
    print("sampling and solving oracle commands", flush=True)
    rng = np.random.default_rng(SEED)

    n_place, n_train, n_val, n_test = 900, 1600, 350, 500
    splits = {}
    for name, n in (("place", n_place), ("train", n_train), ("val", n_val), ("test", n_test)):
        q, m, crop = sample_conditions(n, rng)
        print(f"oracle {name} n={n}", flush=True)
        u_star = oracle_batch(q, m, crop, n_starts=3)
        splits[name] = {"q": q, "m": m, "crop": crop, "u": u_star}

    # Observation conditions for the control map: schedule, oracle, and a random command.
    def pack_features(split, placement, condition: str, rng):
        q, m, crop = split["q"], split["m"], split["crop"]
        if condition == "schedule":
            u = schedule_u(m)
        elif condition == "oracle":
            u = split["u"]
        elif condition == "fixed":
            u = fixed_u(len(q))
        else:
            u = rng.uniform(0.15, 0.95, size=(len(q), 4))
        met = plant(u, q, m, crop)
        windows = {
            "F": met["F"][:, None, :] + rng.normal(0.0, 22.0, size=(len(q), 8, N_STATION)),
            "torque": met["torque"][:, None] + rng.normal(0.0, 0.015, size=(len(q), 8)),
            "vib": np.maximum(0.0, met["vib"][:, None] + rng.normal(0.0, 0.03, size=(len(q), 8))),
            "loss": np.maximum(0.0, met["ls"][:, None] + rng.normal(0.0, 0.0015, size=(len(q), 8))),
            "return": np.maximum(0.0, met["return"][:, None] + rng.normal(0.0, 0.012, size=(len(q), 8))),
            "power": np.maximum(0.0, met["power"][:, None] + rng.normal(0.0, 0.15, size=(len(q), 8))),
        }
        # The status label is the status of the machine that produced the window.
        y = status_from_metrics(met)
        X = features_from_windows(windows, m, q, np.asarray(placement), rng)
        return X, y, met

    print("D-optimal placements", flush=True)
    dopt = {}
    for k in range(2, 9):
        p, v = best_d_optimal(k)
        dopt[k] = {"placement": list(p), "d_criterion": v}
        print(f"  K={k} {p} det^(1/3)={v:.4f}", flush=True)

    # Classification-oriented selection on the place split, features under a mixture of commands.
    print("classification placements", flush=True)
    q, m, crop = splits["place"]["q"], splits["place"]["m"], splits["place"]["crop"]
    # three observation conditions stacked
    Xs, ys = [], []
    for cond in ("fixed", "schedule", "random"):
        X, y, _ = pack_features(splits["place"], np.arange(N_STATION), cond, rng)
        Xs.append(X)
        ys.append(y)
    # Full-station feature layout must be known to slice stations.
    # Rebuild features per placement instead of slicing a packed vector.
    # Cache windows once.
    cached = {}
    for cond in ("fixed", "schedule", "random"):
        if cond == "schedule":
            u = schedule_u(m)
        elif cond == "fixed":
            u = fixed_u(len(q))
        else:
            u = rng.uniform(0.15, 0.95, size=(len(q), 4))
        met = plant(u, q, m, crop)
        cached[cond] = {
            "met": met,
            "y": status_from_metrics(met),
            "windows": {
                "F": met["F"][:, None, :] + rng.normal(0.0, 22.0, size=(len(q), 8, N_STATION)),
                "torque": met["torque"][:, None] + rng.normal(0.0, 0.015, size=(len(q), 8)),
                "vib": np.maximum(0.0, met["vib"][:, None] + rng.normal(0.0, 0.03, size=(len(q), 8))),
                "loss": np.maximum(0.0, met["ls"][:, None] + rng.normal(0.0, 0.0015, size=(len(q), 8))),
                "return": np.maximum(0.0, met["return"][:, None] + rng.normal(0.0, 0.012, size=(len(q), 8))),
                "power": np.maximum(0.0, met["power"][:, None] + rng.normal(0.0, 0.15, size=(len(q), 8))),
            },
        }

    def design_matrix(placement, cond):
        return features_from_windows(cached[cond]["windows"], m, q, np.asarray(placement), rng)

    # Use a held-out slice of the place set.
    n = len(q)
    perm = rng.permutation(n)
    fit_idx, sel_idx = perm[: int(0.7 * n)], perm[int(0.7 * n) :]

    from itertools import combinations

    class_place = {}
    for k in (2, 3, 4, 5, 6, 8):
        best = None
        # Evaluate D-optimal, a front-clustered prefix, and a spread of random subsets plus all if C(8,k) is small.
        candidates = []
        candidates.append(tuple(dopt[k]["placement"]))
        candidates.append(tuple(range(k)))  # prefix, a poor spatial design
        candidates.append(tuple(np.linspace(0, 7, k).round().astype(int)))
        combs = list(combinations(range(N_STATION), k))
        if len(combs) <= 70:
            candidates.extend(combs)
        else:
            pick = rng.choice(len(combs), size=40, replace=False)
            candidates.extend(combs[i] for i in pick)
        # unique
        seen = set()
        uniq = []
        for c in candidates:
            c = tuple(sorted(set(int(i) for i in c)))
            if len(c) == k and c not in seen:
                seen.add(c)
                uniq.append(c)
        for comb in uniq:
            blocks_X, blocks_y = [], []
            for cond in ("fixed", "schedule", "random"):
                X = design_matrix(comb, cond)
                blocks_X.append(X)
                blocks_y.append(cached[cond]["y"])
            X = np.concatenate(blocks_X, axis=0)
            y = np.concatenate(blocks_y, axis=0)
            # repeat the index across the three stacked conditions
            fit, sel = [], []
            for b in range(3):
                fit.append(fit_idx + b * n)
                sel.append(sel_idx + b * n)
            fit = np.concatenate(fit)
            sel = np.concatenate(sel)
            mu, sd = standardize_fit(X[fit])
            W, b = softmax_fit(apply_std(X[fit], mu, sd), y[fit], epochs=40, lr=0.1, seed=k)
            yhat, _ = softmax_predict(apply_std(X[sel], mu, sd), W, b)
            score = macro_f1(yhat, y[sel])
            if best is None or score > best[0]:
                best = (score, comb, float(fisher_det(comb)))
        class_place[k] = {
            "placement": list(best[1]),
            "linear_val_f1": best[0],
            "d_criterion": best[2],
        }
        print(f"  K={k} class-best {best[1]} f1={best[0]:.3f}", flush=True)

    # Choose the deployed placement: smallest K at least 4 whose linear accuracy
    # is within 0.015 of the best K, preferring the classification placement.
    # Deploy the D-optimal stations at the smallest K>=4. The classification
    # search is reported beside it; the information design is the one installed,
    # because it does not depend on a particular labeled sample.
    deploy_k = 4
    placement = list(dopt[deploy_k]["placement"])
    print("deploy", deploy_k, placement, "class-best", class_place[deploy_k]["placement"], flush=True)

    # Status networks on train/val/test, features under the mixture, labels from that condition.
    def split_xy(split, placement, rng):
        Xs, ys, us = [], [], []
        for cond in ("fixed", "schedule", "random"):
            X, y, _ = pack_features(split, placement, cond, rng)
            Xs.append(X)
            ys.append(y)
            us.append(split["u"])
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(us)

    print("training status network", flush=True)
    Xtr, ytr, _ = split_xy(splits["train"], placement, rng)
    Xva, yva, _ = split_xy(splits["val"], placement, rng)
    Xte, yte, _ = split_xy(splits["test"], placement, rng)
    mu_s, sd_s = standardize_fit(Xtr)
    Xtr_s, Xva_s, Xte_s = apply_std(Xtr, mu_s, sd_s), apply_std(Xva, mu_s, sd_s), apply_std(Xte, mu_s, sd_s)
    Wlin, blin = softmax_fit(Xtr_s, ytr, epochs=80, lr=0.08, seed=3)
    yhat_lin, _ = softmax_predict(Xte_s, Wlin, blin)
    status_net, st_ep, st_acc = train_mlp_classifier(
        Xtr_s, ytr, Xva_s, yva, sizes=(Xtr.shape[1], 48, 32, 5), epochs=100, lr=8e-4, seed=4
    )
    yhat_net = status_net.predict(Xte_s).argmax(axis=1)
    # also linear and net accuracy along the K curve, on the same test disturbances, fixed placement family
    k_curve = []
    for k in (2, 3, 4, 5, 6, 8):
        pk = class_place[k]["placement"]
        Xtr_k, ytr_k, _ = split_xy(splits["train"], pk, rng)
        Xva_k, yva_k, _ = split_xy(splits["val"], pk, rng)
        Xte_k, yte_k, _ = split_xy(splits["test"], pk, rng)
        mu_k, sd_k = standardize_fit(Xtr_k)
        Wk, bk = softmax_fit(apply_std(Xtr_k, mu_k, sd_k), ytr_k, epochs=50, lr=0.08, seed=10 + k)
        yk, _ = softmax_predict(apply_std(Xte_k, mu_k, sd_k), Wk, bk)
        net_k, _, _ = train_mlp_classifier(
            apply_std(Xtr_k, mu_k, sd_k),
            ytr_k,
            apply_std(Xva_k, mu_k, sd_k),
            yva_k,
            sizes=(Xtr_k.shape[1], 48, 32, 5),
            epochs=60,
            lr=8e-4,
            seed=20 + k,
        )
        ykn = net_k.predict(apply_std(Xte_k, mu_k, sd_k)).argmax(axis=1)
        prefix = tuple(range(k))
        Xtr_p, ytr_p, _ = split_xy(splits["train"], prefix, rng)
        Xte_p, yte_p, _ = split_xy(splits["test"], prefix, rng)
        mu_p, sd_p = standardize_fit(Xtr_p)
        Wp, bp = softmax_fit(apply_std(Xtr_p, mu_p, sd_p), ytr_p, epochs=50, lr=0.08, seed=30 + k)
        yp, _ = softmax_predict(apply_std(Xte_p, mu_p, sd_p), Wp, bp)
        k_curve.append(
            {
                "K": k,
                "placement": list(pk),
                "stations_1based": [i + 1 for i in pk],
                "d_optimal": dopt[k]["placement"],
                "d_optimal_1based": [i + 1 for i in dopt[k]["placement"]],
                "d_criterion_selected": fisher_det(tuple(pk)),
                "d_criterion_optimal": dopt[k]["d_criterion"],
                "linear_test_acc": accuracy(yk, yte_k),
                "linear_test_f1": macro_f1(yk, yte_k),
                "net_test_acc": accuracy(ykn, yte_k),
                "net_test_f1": macro_f1(ykn, yte_k),
                "prefix_linear_acc": accuracy(yp, yte_p),
                "net_recall": per_class_recall(ykn, yte_k),
            }
        )
        print(
            f"  curve K={k} net={k_curve[-1]['net_test_acc']:.3f} "
            f"lin={k_curve[-1]['linear_test_acc']:.3f} prefix={k_curve[-1]['prefix_linear_acc']:.3f}",
            flush=True,
        )

    print("training control network", flush=True)
    # Control features are observed under schedule and under random commands; target is u*.
    def control_xy(split, rng):
        Xs, Ys = [], []
        for cond in ("schedule", "random", "fixed"):
            X, _, _ = pack_features(split, placement, cond, rng)
            Xs.append(X)
            Ys.append(split["u"])
        return np.concatenate(Xs), np.concatenate(Ys)

    Xtr_c, Ytr = control_xy(splits["train"], rng)
    Xva_c, Yva = control_xy(splits["val"], rng)
    Xte_c, Yte = control_xy(splits["test"], rng)
    mu_c, sd_c = standardize_fit(Xtr_c)
    ctrl, ctrl_ep, ctrl_v = train_mlp_regression(
        apply_std(Xtr_c, mu_c, sd_c),
        Ytr,
        apply_std(Xva_c, mu_c, sd_c),
        Yva,
        sizes=(Xtr_c.shape[1], 64, 64, 32, 4),
        epochs=140,
        lr=1e-3,
        seed=5,
    )
    # linear least squares baseline on the same standardized inputs, sigmoid-free clipped
    Xtr_aug = np.concatenate([apply_std(Xtr_c, mu_c, sd_c), np.ones((len(Xtr_c), 1))], axis=1)
    Xte_aug = np.concatenate([apply_std(Xte_c, mu_c, sd_c), np.ones((len(Xte_c), 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(Xtr_aug, Ytr, rcond=None)
    Y_lin = np.clip(Xte_aug @ coef, 0.0, 1.0)
    Y_net = 1.0 / (1.0 + np.exp(-ctrl.predict(apply_std(Xte_c, mu_c, sd_c))))
    # The network's last layer is linear in train_mlp_regression. Squash to (0,1).
    # Retrain note: targets are already in (0,1), a linear output can leave the box.
    # We clip at use time. Report MAE after clipping.
    Y_net = np.clip(ctrl.predict(apply_std(Xte_c, mu_c, sd_c)), 0.0, 1.0)
    mae_net = float(np.mean(np.abs(Y_net - Yte)))
    mae_lin = float(np.mean(np.abs(Y_lin - Yte)))
    n_te = len(splits["test"]["q"])
    heavy = (splits["test"]["q"] >= 6.5) & (splits["test"]["m"] >= 0.19)
    # The three observation blocks are stacked in the same order as control_xy.
    heavy3 = np.concatenate([heavy, heavy, heavy])
    mae_net_heavy = float(np.mean(np.abs(Y_net[heavy3] - Yte[heavy3]))) if heavy.any() else None
    mae_lin_heavy = float(np.mean(np.abs(Y_lin[heavy3] - Yte[heavy3]))) if heavy.any() else None
    mae_net_cmd = [float(v) for v in np.mean(np.abs(Y_net - Yte), axis=0)]
    mae_lin_cmd = [float(v) for v in np.mean(np.abs(Y_lin - Yte), axis=0)]
    print(
        f"  command MAE net={mae_net:.4f} linear={mae_lin:.4f} "
        f"heavy net={mae_net_heavy:.4f} lin={mae_lin_heavy:.4f} best_ep={ctrl_ep}",
        flush=True,
    )

    # Closed-loop evaluation on the test disturbances: one settle from the schedule.
    def rollout_command(split, kind: str):
        q, m, crop = split["q"], split["m"], split["crop"]
        if kind == "fixed":
            u = fixed_u(len(q))
        elif kind == "schedule":
            u = schedule_u(m)
        elif kind == "oracle":
            u = split["u"]
        elif kind == "linear":
            X, _, _ = pack_features(split, placement, "schedule", rng)
            Xa = np.concatenate([apply_std(X, mu_c, sd_c), np.ones((len(X), 1))], axis=1)
            u = np.clip(Xa @ coef, 0.0, 1.0)
        elif kind == "network":
            X, _, _ = pack_features(split, placement, "schedule", rng)
            u = np.clip(ctrl.predict(apply_std(X, mu_c, sd_c)), 0.0, 1.0)
        else:
            raise ValueError(kind)
        return u

    def metrics_of(split, kind):
        u = rollout_command(split, kind)
        met = plant(u, split["q"], split["m"], split["crop"])
        return met, u

    test_summary = {}
    for kind in ("fixed", "schedule", "linear", "network", "oracle"):
        met, u = metrics_of(splits["test"], kind)
        test_summary[kind] = {
            "loss_pct": float(met["loss"].mean() * 100),
            "lu_pct": float(met["lu"].mean() * 100),
            "lg_pct": float(met["lg"].mean() * 100),
            "ls_pct": float(met["ls"].mean() * 100),
            "eta_pct": float(met["eta"].mean() * 100),
            "iota_pct": float(met["iota"].mean() * 100),
            "break_pct": float(met["breakage"].mean() * 100),
            "e": float(met["e"].mean()),
            "cv": float(met["cv"].mean()),
            "fmax": float(met["fmax"].mean()),
            "stab": float(met["stab"].mean()),
            "stab_p10": float(np.quantile(met["stab"], 0.10)),
            "below_floor": float(np.mean(met["stab"] < 0.699)),
            "power": float(met["power"].mean()),
            "mae_to_oracle": float(np.mean(np.abs(u - splits["test"]["u"]))),
        }
        print(f"  test {kind}: loss={test_summary[kind]['loss_pct']:.3f} stab={test_summary[kind]['stab']:.3f} e={test_summary[kind]['e']:.3f}", flush=True)

    # Seasonal points
    print("seasonal", flush=True)
    seasonal_rows = []
    q_s = np.array([p[2] for p in POINTS])
    m_s = np.array([p[3] for p in POINTS])
    c_s = np.array([CROP_NAMES.index(p[1]) for p in POINTS])
    w_s = np.array([p[4] for p in POINTS])
    print("  seasonal oracle", flush=True)
    u_s = oracle_batch(q_s, m_s, c_s, n_starts=6)
    seasonal_split = {"q": q_s, "m": m_s, "crop": c_s, "u": u_s}
    seasonal = {}
    commands = {}
    for kind in ("fixed", "schedule", "linear", "network", "oracle"):
        met, u = metrics_of(seasonal_split, kind)
        seasonal[kind] = []
        commands[kind] = u.tolist()
        for i in range(len(POINTS)):
            seasonal[kind].append(
                {
                    "point": POINTS[i][0],
                    "crop": POINTS[i][1],
                    "q": POINTS[i][2],
                    "m": POINTS[i][3],
                    "weight": POINTS[i][4],
                    "loss_pct": float(met["loss"][i] * 100),
                    "lu_pct": float(met["lu"][i] * 100),
                    "lg_pct": float(met["lg"][i] * 100),
                    "ls_pct": float(met["ls"][i] * 100),
                    "eta_pct": float(met["eta"][i] * 100),
                    "iota_pct": float(met["iota"][i] * 100),
                    "break_pct": float(met["breakage"][i] * 100),
                    "e": float(met["e"][i]),
                    "cv": float(met["cv"][i]),
                    "fmax": float(met["fmax"][i]),
                    "stab": float(met["stab"][i]),
                    "power": float(met["power"][i]),
                    "u": [float(v) for v in u[i]],
                    "forces": [float(v) for v in met["F"][i]],
                }
            )
    def wmean(kind, key):
        return float(sum(seasonal[kind][i][key] * w_s[i] for i in range(len(POINTS))))

    seasonal_means = {
        kind: {key: wmean(kind, key) for key in ("loss_pct", "lu_pct", "lg_pct", "ls_pct", "eta_pct", "iota_pct", "break_pct", "e", "cv", "fmax", "stab", "power")}
        for kind in seasonal
    }
    for kind in seasonal_means:
        print(f"  season {kind}: loss={seasonal_means[kind]['loss_pct']:.3f} stab={seasonal_means[kind]['stab']:.3f} e={seasonal_means[kind]['e']:.3f}", flush=True)

    # Dynamic slug, wheat, moisture 0.17, network in the loop with a rate limit.
    print("slug", flush=True)
    T = np.linspace(0.0, 26.0, 131)  # 0.2 s
    q_header = 5.0 + 3.2 * np.exp(-0.5 * ((T - 12.0) / 2.2) ** 2)
    m_slug = 0.17
    crop_slug = 0

    def run_slug(kind: str):
        u = schedule_u(np.array([m_slug]))[0].copy()
        if kind == "fixed":
            u = fixed_u(1)[0].copy()
        hist = []
        window_F = []
        for t_i, q_i in enumerate(q_header):
            met = plant(u[None, :], np.array([q_i]), np.array([m_slug]), np.array([crop_slug]))
            noise_F = rng.normal(0.0, 22.0, size=(8,))
            window_F.append(met["F"][0] + noise_F)
            if len(window_F) > 8:
                window_F.pop(0)
            hist.append(
                {
                    "t": float(T[t_i]),
                    "q": float(q_i),
                    "loss_pct": float(met["loss"][0] * 100),
                    "stab": float(met["stab"][0]),
                    "fmax": float(met["fmax"][0]),
                    "cv": float(met["cv"][0]),
                    "e": float(met["e"][0]),
                    "u": [float(v) for v in u],
                }
            )
            if kind in ("network", "linear") and len(window_F) >= 4:
                # build one feature row from the available window, padded by repeating
                Fw = np.stack(window_F, axis=0)
                if Fw.shape[0] < 8:
                    Fw = np.concatenate([np.repeat(Fw[:1], 8 - Fw.shape[0], axis=0), Fw], axis=0)
                dummy_win = {
                    "F": Fw[None, :, :],
                    "torque": np.full((1, 8), met["torque"][0]) + rng.normal(0, 0.01, size=(1, 8)),
                    "vib": np.full((1, 8), met["vib"][0]),
                    "loss": np.full((1, 8), met["ls"][0]),
                    "return": np.full((1, 8), met["return"][0]),
                    "power": np.full((1, 8), met["power"][0]),
                }
                X = features_from_windows(dummy_win, np.array([m_slug]), np.array([q_i]), np.asarray(placement), rng)
                Xs = apply_std(X, mu_c, sd_c)
                if kind == "network":
                    u_tgt = np.clip(ctrl.predict(Xs)[0], 0.0, 1.0)
                else:
                    xa = np.concatenate([Xs, np.ones((1, 1))], axis=1)
                    u_tgt = np.clip((xa @ coef)[0], 0.0, 1.0)
                u = clip_u(u + np.clip(u_tgt - u, -0.08, 0.08))
            elif kind == "schedule":
                u = schedule_u(np.array([m_slug]))[0].copy()
            elif kind == "fixed":
                u = fixed_u(1)[0].copy()
        return hist

    slug = {kind: run_slug(kind) for kind in ("fixed", "schedule", "network")}
    slug_summary = {}
    for kind, hist in slug.items():
        slug_summary[kind] = {
            "loss_mean": float(np.mean([h["loss_pct"] for h in hist])),
            "loss_peak": float(np.max([h["loss_pct"] for h in hist])),
            "fmax_peak": float(np.max([h["fmax"] for h in hist])),
            "stab_min": float(np.min([h["stab"] for h in hist])),
            "cv_at_peak": float(hist[int(np.argmax(q_header))]["cv"]),
        }
        print(f"  slug {kind}: {slug_summary[kind]}", flush=True)

    # Inference timing on the deployed network
    import time

    Xb = apply_std(Xte_c[:256], mu_c, sd_c)
    for _ in range(5):
        ctrl.predict(Xb)
    t0 = time.perf_counter()
    reps = 40
    for _ in range(reps):
        ctrl.predict(Xb)
    elapsed = (time.perf_counter() - t0) / reps
    us_per = elapsed / 256 * 1e6

    # class counts on test mixture
    counts = {STATUS_NAMES[i]: int(np.sum(yte == i)) for i in range(5)}

    # parameter and flop estimate
    dims = [Xtr_c.shape[1], 64, 64, 32, 4]
    params = count_params(ctrl)
    macs = sum(a * c for a, c in zip(dims[:-1], dims[1:]))
    status_params = count_params(status_net)

    # sample feature row description
    feature_dim = int(Xtr_c.shape[1])

    # confidence: max softmax on test
    logits = status_net.predict(Xte_s)
    logits = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    conf = P.max(axis=1)
    conf_correct = float(conf[yhat_net == yte].mean()) if np.any(yhat_net == yte) else 0.0
    fallback_rate = float((conf < 0.55).mean())

    result = {
        "seed": SEED,
        "placement": placement,
        "placement_1based": [i + 1 for i in placement],
        "deploy_k": deploy_k,
        "feature_dim": feature_dim,
        "d_optimal": {str(k): dopt[k] for k in dopt},
        "class_place": {str(k): class_place[k] for k in class_place},
        "k_curve": k_curve,
        "status_test": {
            "linear_acc": accuracy(yhat_lin, yte),
            "linear_f1": macro_f1(yhat_lin, yte),
            "net_acc": accuracy(yhat_net, yte),
            "net_f1": macro_f1(yhat_net, yte),
            "net_recall": per_class_recall(yhat_net, yte),
            "linear_recall": per_class_recall(yhat_lin, yte),
            "counts": counts,
            "best_epoch": int(st_ep),
            "val_acc": float(st_acc),
            "mean_confidence_when_correct": conf_correct,
            "fallback_rate_at_0.55": fallback_rate,
            "params": status_params,
        },
        "control": {
            "mae_net": mae_net,
            "mae_linear": mae_lin,
            "mae_net_heavy": mae_net_heavy,
            "mae_linear_heavy": mae_lin_heavy,
            "mae_net_by_command": mae_net_cmd,
            "mae_linear_by_command": mae_lin_cmd,
            "best_epoch": int(ctrl_ep),
            "val_mse": float(ctrl_v),
            "params": params,
            "macs": int(macs),
            "hidden": [64, 64, 32],
            "us_per_forward_numpy": float(us_per),
            "input_dim": feature_dim,
        },
        "test_summary": test_summary,
        "seasonal_means": seasonal_means,
        "seasonal": seasonal,
        "slug_summary": slug_summary,
        "slug_samples": {
            kind: [h for h in hist if abs(h["t"] * 5 - round(h["t"] * 5)) < 1e-6 and h["t"] in (0, 4, 8, 10, 12, 14, 18, 26) or abs(h["t"] - round(h["t"])) < 1e-8 and int(round(h["t"])) in (0, 4, 8, 10, 12, 14, 18, 26)]
            for kind, hist in slug.items()
        },
        "status_names": list(STATUS_NAMES),
        "objective": {
            "eta": 1.0,
            "energy": 0.012,
            "impurity": 0.48,
            "breakage": 1.15,
            "cv": 0.025,
            "stability_hinge": 0.26,
            "stability_floor": 0.70,
            "block_hinge": 0.40,
            "block_floor_fraction": 0.82,
        },
    }
    # cleaner slug samples
    slug_samples = {}
    wanted = {0.0, 4.0, 8.0, 10.0, 12.0, 14.0, 18.0, 26.0}
    for kind, hist in slug.items():
        slug_samples[kind] = []
        for h in hist:
            if any(abs(h["t"] - w) < 0.05 for w in wanted):
                # keep the closest only once
                pass
        chosen = []
        for w in sorted(wanted):
            h = min(hist, key=lambda z, w=w: abs(z["t"] - w))
            chosen.append(h)
        slug_samples[kind] = chosen
    result["slug_samples"] = slug_samples

    dest = out_dir / "dnn_results.json"
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", dest, flush=True)


def probe() -> None:
    """Seasonal commands and a short imitator check. Used while tuning the plant."""
    rng = np.random.default_rng(SEED)
    q_s = np.array([p[2] for p in POINTS])
    m_s = np.array([p[3] for p in POINTS])
    c_s = np.array([CROP_NAMES.index(p[1]) for p in POINTS])
    print("seasonal oracle", flush=True)
    u_o = oracle_batch(q_s, m_s, c_s, n_starts=5)
    for kind, u in (
        ("fixed", fixed_u(len(POINTS))),
        ("schedule", schedule_u(m_s)),
        ("oracle", u_o),
    ):
        met = plant(u, q_s, m_s, c_s)
        print(f"\n{kind}")
        for i, p in enumerate(POINTS):
            print(
                f"  {p[0]:20} loss {100*met['loss'][i]:5.2f}  eta {100*met['eta'][i]:6.2f}  "
                f"e {met['e'][i]:5.2f}  cv {met['cv'][i]:5.3f}  F {met['fmax'][i]:7.1f}  "
                f"S {met['stab'][i]:5.3f}  b {100*met['breakage'][i]:4.2f}  "
                f"io {100*met['iota'][i]:4.2f}  u {[round(float(v),2) for v in u[i]]}"
            )
    q, m, crop = sample_conditions(240, rng)
    # class mix under fixed, schedule, random
    for cond, u in (
        ("fixed", fixed_u(240)),
        ("schedule", schedule_u(m)),
        ("random", rng.uniform(0.15, 0.95, size=(240, 4))),
    ):
        met = plant(u, q, m, crop)
        y = status_from_metrics(met)
        counts = {STATUS_NAMES[i]: int(np.sum(y == i)) for i in range(5)}
        print(cond, "classes", counts, "loss", round(float(met["loss"].mean()*100), 2), "fmax", round(float(met["fmax"].mean()), 1))


if __name__ == "__main__":
    import sys

    if "--probe" in sys.argv:
        probe()
    else:
        main()

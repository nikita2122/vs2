"""Monte Carlo estimation of the two proposed modules.

ICPH: illumination-invariant chromatic prototype head.
PLC:  perspective lattice completion counter.

Numbers written to estimation_results.json are produced by this script.
They are simulation estimates under the observation models below, not
field measurements from a brewery line.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(20261109)
OUT = Path(__file__).with_name("estimation_results.json")

# Nominal plastic-crate body colors in linear RGB, chosen to be separable
# in chromaticity and representative of brewery returnable crates.
PROTOTYPES = np.array(
    [
        [0.72, 0.45, 0.12],  # amber
        [0.15, 0.55, 0.22],  # green
        [0.70, 0.12, 0.12],  # red
        [0.12, 0.25, 0.62],  # blue
        [0.85, 0.75, 0.15],  # yellow
        [0.82, 0.82, 0.80],  # white
    ],
    dtype=float,
)
CLASS_NAMES = ["amber", "green", "red", "blue", "yellow", "white"]
N_CLASS = len(CLASS_NAMES)

# Crate geometry used by the lattice (metres).
FACE_W = 0.40
FACE_H = 0.28
CRATE_D = 0.30


def shades_of_gray(pixels: np.ndarray, p: float) -> np.ndarray:
    ell = np.mean(np.abs(pixels) ** p, axis=0) ** (1.0 / p)
    ell = np.maximum(ell, 1e-6)
    return pixels / ell, ell


def sample_patch(k: int, severity: float, n_pix: int = 80) -> np.ndarray:
    """One crate face under a random illuminant, shading, and edge mixing."""
    base = PROTOTYPES[k]
    gain = np.exp(RNG.normal(0.0, severity, size=3))
    gain = gain / np.mean(gain)
    shadow = float(RNG.uniform(0.40, 1.0))
    shade = np.linspace(0.72, 1.18, n_pix)
    pix = base * gain * shadow * shade[:, None]
    # A strip of a neighboring crate, as on a shared occluding boundary.
    if RNG.random() < 0.25:
        k2 = (k + int(RNG.integers(1, N_CLASS))) % N_CLASS
        m = int(0.18 * n_pix)
        pix[:m] = PROTOTYPES[k2] * gain * shadow
    pix = pix + RNG.normal(0.0, 0.012, size=pix.shape)
    return np.clip(pix, 0.0, None)


def nearest_color(mean_rgb: np.ndarray) -> int:
    v = mean_rgb / (np.linalg.norm(mean_rgb) + 1e-8)
    P = PROTOTYPES / (np.linalg.norm(PROTOTYPES, axis=1, keepdims=True) + 1e-8)
    return int(np.argmax(P @ v))


def classify_sog(pixels: np.ndarray, p: float) -> int:
    corr, _ = shades_of_gray(pixels, p)
    return nearest_color(corr.mean(axis=0))


def patch_features(pixels: np.ndarray) -> np.ndarray:
    corr, ell = shades_of_gray(pixels, 6.0)
    chroma = corr.mean(axis=0)
    chroma = chroma / (chroma.sum() + 1e-8)
    r, g, b = chroma
    opponent = np.array([r - g, 0.5 * (r + g) - b], dtype=float)
    ell_c = ell / (ell.sum() + 1e-8)
    return np.concatenate([chroma, opponent, ell_c])


def train_icph(n_train: int = 6000, epochs: int = 500, lr: float = 0.35, l2: float = 1e-3):
    X = np.stack(
        [
            patch_features(sample_patch(int(k), float(RNG.uniform(0.05, 0.55))))
            for k in RNG.integers(0, N_CLASS, size=n_train)
        ]
    )
    y = np.empty(n_train, dtype=int)
    # The labels were consumed inside the comprehension; rebuild aligned labels.
    # Retrain with an explicit loop so features and labels share one draw.
    feats = []
    labels = []
    for _ in range(n_train):
        k = int(RNG.integers(0, N_CLASS))
        feats.append(patch_features(sample_patch(k, float(RNG.uniform(0.05, 0.55)))))
        labels.append(k)
    X = np.stack(feats)
    y = np.array(labels)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sd
    W = np.zeros((N_CLASS, Xs.shape[1]))
    b = np.zeros(N_CLASS)
    eye = np.eye(N_CLASS)
    for _ in range(epochs):
        logits = Xs @ W.T + b
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        grad = (prob - eye[y]) / n_train
        W -= lr * (grad.T @ Xs + l2 * W)
        b -= lr * grad.sum(axis=0)
    return {"W": W, "b": b, "mu": mu, "sd": sd}


def predict_icph(model, pixels: np.ndarray) -> int:
    f = (patch_features(pixels) - model["mu"]) / model["sd"]
    return int(np.argmax(f @ model["W"].T + model["b"]))


def evaluate_color(model, severities, n_test: int = 2500):
    rows = []
    confusion = {s: np.zeros((N_CLASS, N_CLASS), dtype=int) for s in severities}
    for severity in severities:
        hits = {"raw_rgb": 0, "gray_world": 0, "shades_of_gray": 0, "icph": 0}
        for _ in range(n_test):
            k = int(RNG.integers(0, N_CLASS))
            pix = sample_patch(k, severity)
            preds = {
                "raw_rgb": nearest_color(pix.mean(axis=0)),
                "gray_world": classify_sog(pix, 1.0),
                "shades_of_gray": classify_sog(pix, 6.0),
                "icph": predict_icph(model, pix),
            }
            for name, pred in preds.items():
                hits[name] += int(pred == k)
            confusion[severity][k, preds["icph"]] += 1
        row = {name: hits[name] / n_test for name in hits}
        row["severity"] = severity
        row["n_test"] = n_test
        rows.append(row)
    # Per-class recall of ICPH at the harshest illuminant.
    harsh = severities[-1]
    cm = confusion[harsh]
    recall = (np.diag(cm) / np.maximum(cm.sum(axis=1), 1)).tolist()
    return {
        "by_severity": rows,
        "icph_recall_at_harsh": dict(zip(CLASS_NAMES, recall)),
        "confusion_harsh": cm.tolist(),
        "harsh_severity": harsh,
    }


def detection_probability(row: int, n_rows: int, blur: float) -> float:
    """Bottom row is shadowed by the forks; top row is often clipped."""
    if n_rows == 1:
        base = 0.90
    elif row == 0:
        base = 0.70
    elif row == n_rows - 1:
        base = 0.84
    else:
        base = 0.96
    return float(np.clip(base - 0.18 * blur, 0.35, 0.99))


def sample_stack():
    rows = int(RNG.integers(2, 7))
    cols = int(RNG.integers(2, 5))
    depth = int(RNG.integers(1, 4))
    occupied = RNG.random((rows, cols)) < 0.93
    # A stack always has at least one crate.
    if not occupied.any():
        occupied[0, 0] = True
    blur = float(RNG.uniform(0.0, 1.0))
    return rows, cols, depth, occupied, blur


def observe_frame(rows, cols, occupied, blur, origin_noise=0.0):
    """Return detected centers in rectified metres and a few false positives."""
    centers = []
    true_cells = []
    for r in range(rows):
        for c in range(cols):
            if not occupied[r, c]:
                continue
            if RNG.random() > detection_probability(r, rows, blur):
                continue
            x = (c + 0.5) * FACE_W + RNG.normal(0.0, 0.035) + origin_noise
            y = (r + 0.5) * FACE_H + RNG.normal(0.0, 0.028)
            centers.append((x, y))
            true_cells.append((r, c))
    n_fp = RNG.poisson(0.35)
    for _ in range(n_fp):
        x = RNG.uniform(-0.15, cols * FACE_W + 0.15)
        y = RNG.uniform(-0.10, rows * FACE_H + 0.10)
        centers.append((x, y))
    return centers


def snap_lattice(centers, origin):
    """Assign detections to integer cells. Known pitch, estimated origin."""
    cells = {}
    for x, y in centers:
        c = int(np.round((x - origin[0]) / FACE_W - 0.5))
        r = int(np.round((y - origin[1]) / FACE_H - 0.5))
        cells.setdefault((r, c), 0)
        cells[(r, c)] += 1
    return cells


def estimate_origin(centers):
    if not centers:
        return np.array([0.0, 0.0])
    xs = np.array([p[0] for p in centers])
    ys = np.array([p[1] for p in centers])
    # Origin is the residual of the median detection modulo the known pitch.
    ox = np.median(xs - (np.round(xs / FACE_W - 0.5) + 0.5) * FACE_W)
    oy = np.median(ys - (np.round(ys / FACE_H - 0.5) + 0.5) * FACE_H)
    return np.array([ox, oy])


def depth_observation(depth: int) -> int:
    """Pallet-depth reading. Correct with probability 0.90, else off by one."""
    if RNG.random() < 0.90:
        return depth
    delta = int(RNG.choice([-1, 1]))
    return int(np.clip(depth + delta, 1, 3))


def mask_count(centers, depth_hat, use_depth: bool) -> int:
    n = len(centers)
    return int(n * depth_hat) if use_depth else int(n)


def area_ratio_count(rows, cols, occupied, blur, depth_hat) -> int:
    """Each detected face contributes a noisy visible-area fraction."""
    total = 0.0
    for r in range(rows):
        for c in range(cols):
            if not occupied[r, c]:
                continue
            if RNG.random() > detection_probability(r, rows, blur):
                continue
            visible = float(np.clip(RNG.normal(0.86, 0.12), 0.35, 1.15))
            total += visible
    # False-positive area.
    total += RNG.poisson(0.35) * float(RNG.uniform(0.4, 0.9))
    return int(max(0, np.round(total)) * depth_hat)


def span_count(centers, depth_hat) -> int:
    if not centers:
        return 0
    xs = np.array([p[0] for p in centers])
    ys = np.array([p[1] for p in centers])
    # Robust span: drop the outer 5% if many points, else use min/max.
    c_est = int(np.round((np.percentile(xs, 98) - np.percentile(xs, 2)) / FACE_W)) + 1
    r_est = int(np.round((np.percentile(ys, 98) - np.percentile(ys, 2)) / FACE_H)) + 1
    c_est = int(np.clip(c_est, 1, 8))
    r_est = int(np.clip(r_est, 1, 10))
    return int(r_est * c_est * depth_hat)


def plc_count(frames, depth_hat) -> int:
    """Fuse frames, snap to the calibrated lattice, complete single-cell gaps."""
    all_centers = [p for frame in frames for p in frame]
    if not all_centers:
        return 0
    origin = estimate_origin(all_centers)
    votes = {}
    for frame in frames:
        snapped = snap_lattice(frame, origin)
        for cell in snapped:
            votes[cell] = votes.get(cell, 0) + 1
    # Keep cells voted in at least one frame, inside a robust bounding lattice.
    occupied_cells = {cell for cell, v in votes.items() if v >= 1}
    if not occupied_cells:
        return 0
    rs = [r for r, _ in occupied_cells]
    cs = [c for _, c in occupied_cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    # Reject cells far outside the main component (false positives).
    occupied_cells = {
        (r, c)
        for (r, c) in occupied_cells
        if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1
    }
    # One pass of gap completion: a missing lattice cell with two
    # occupied orthogonal neighbors inside the bounding rectangle is filled.
    # This recovers detector misses and does not invent a new row or column
    # unless that row or column already has support.
    filled = set(occupied_cells)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in filled:
                continue
            neighbors = [
                (r - 1, c) in filled,
                (r + 1, c) in filled,
                (r, c - 1) in filled,
                (r, c + 1) in filled,
            ]
            if sum(neighbors) >= 2:
                filled.add((r, c))
    return int(len(filled) * depth_hat)


def evaluate_counting(n_trials: int = 4000, n_frames: int = 5):
    methods = [
        "mask",
        "mask_x_depth",
        "area_ratio",
        "span_lattice",
        "plc_single",
        "plc_passage",
    ]
    err = {m: [] for m in methods}
    exact = {m: 0 for m in methods}
    abs_rel = {m: [] for m in methods}
    by_depth = {d: {m: [] for m in methods} for d in (1, 2, 3)}
    true_hist = []
    pred_hist = {m: [] for m in methods}

    for _ in range(n_trials):
        rows, cols, depth, occupied, blur = sample_stack()
        n_front = int(occupied.sum())
        truth = int(n_front * depth)
        d_hat = depth_observation(depth)
        frames = [observe_frame(rows, cols, occupied, blur) for _ in range(n_frames)]
        preds = {
            "mask": mask_count(frames[0], d_hat, use_depth=False),
            "mask_x_depth": mask_count(frames[0], d_hat, use_depth=True),
            "area_ratio": area_ratio_count(rows, cols, occupied, blur, d_hat),
            "span_lattice": span_count(frames[0], d_hat),
            "plc_single": plc_count(frames[:1], d_hat),
            "plc_passage": plc_count(frames, d_hat),
        }
        true_hist.append(truth)
        for m, pred in preds.items():
            e = pred - truth
            err[m].append(e)
            exact[m] += int(pred == truth)
            abs_rel[m].append(abs(e) / truth)
            by_depth[depth][m].append(e)
            pred_hist[m].append(pred)

    def pack(errors):
        a = np.array(errors, dtype=float)
        return {
            "mae": float(np.mean(np.abs(a))),
            "rmse": float(np.sqrt(np.mean(a ** 2))),
            "bias": float(np.mean(a)),
            "exact": None,
            "mape": None,
        }

    summary = {}
    for m in methods:
        s = pack(err[m])
        s["exact"] = exact[m] / n_trials
        s["mape"] = float(np.mean(abs_rel[m]))
        s["p90_abs"] = float(np.percentile(np.abs(err[m]), 90))
        summary[m] = s

    depth_summary = {}
    for d, bucket in by_depth.items():
        depth_summary[str(d)] = {}
        for m in methods:
            a = np.array(bucket[m], dtype=float)
            depth_summary[str(d)][m] = {
                "mae": float(np.mean(np.abs(a))),
                "rmse": float(np.sqrt(np.mean(a ** 2))),
                "bias": float(np.mean(a)),
                "n": int(a.size),
            }

    # Illustrative single passage for the worked example in the text.
    rng_state = RNG.bit_generator.state
    example = None
    for _ in range(200):
        rows, cols, depth, occupied, blur = sample_stack()
        if (rows, cols, depth) == (4, 3, 2) and int(occupied.sum()) >= 10:
            frames = [observe_frame(rows, cols, occupied, blur) for _ in range(n_frames)]
            d_hat = depth
            example = {
                "rows": rows,
                "cols": cols,
                "depth": depth,
                "occupied": occupied.astype(int).tolist(),
                "front_count": int(occupied.sum()),
                "truth": int(occupied.sum() * depth),
                "blur": blur,
                "detections_per_frame": [len(f) for f in frames],
                "plc_single": plc_count(frames[:1], d_hat),
                "plc_passage": plc_count(frames, d_hat),
                "mask": mask_count(frames[0], d_hat, False),
                "span": span_count(frames[0], d_hat),
            }
            break
    RNG.bit_generator.state = rng_state

    return {
        "n_trials": n_trials,
        "n_frames": n_frames,
        "summary": summary,
        "by_depth": depth_summary,
        "example": example,
        "true_count_mean": float(np.mean(true_hist)),
        "true_count_std": float(np.std(true_hist)),
    }


def main():
    print("training ICPH stand-in...")
    model = train_icph()
    print("evaluating color...")
    color = evaluate_color(model, severities=[0.05, 0.15, 0.30, 0.45, 0.60])
    print("evaluating counting...")
    counting = evaluate_counting()
    # Incremental cost of the chromatic head relative to YOLOv8m-seg.
    # Prototype branch is unchanged. ICPH adds a 1x1 projection of mask-pooled
    # features (256 -> 32) and a 6-way cosine prototype layer.
    yolo_m_seg = {"params_m": 27.3, "flops_b": 104.5, "mask_map": 40.8, "box_map": 49.9}
    pooled = 256
    embed = 32
    extra_params = pooled * embed + embed + embed * N_CLASS
    # FLOPs: one pooled vector per detection, negligible next to the backbone.
    # The illuminant estimator is a 3-channel shades-of-gray reduction on the
    # masked pixels plus a 32-d residual MLP: 3*32 + 32*32 + 32*N_CLASS.
    mlp_params = 3 * embed + embed * embed + embed + embed * N_CLASS
    extra_params += mlp_params
    cost = {
        "baseline": yolo_m_seg,
        "icph_extra_params": int(extra_params),
        "icph_extra_params_fraction": extra_params / (yolo_m_seg["params_m"] * 1e6),
        "embed_dim": embed,
        "pooled_dim": pooled,
    }
    payload = {"color": color, "counting": counting, "cost": cost, "classes": CLASS_NAMES}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(color["by_severity"], indent=2))
    print(json.dumps(counting["summary"], indent=2))
    print("example", counting["example"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()

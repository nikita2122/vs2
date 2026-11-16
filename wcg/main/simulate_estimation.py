"""Numerical estimation for the wooden-seed planter models.

The equations implemented here are the same discrete models written in
Chapters 2 and 3 of the thesis. Results are model solutions, not field data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RNG_SEED = 20261029
D_CELL_MM = 24.0
H_STAR_MM = 30.0
X_T_M = 0.25  # target in-row spacing
N_EVENTS = 4000

LOTS = {
    "pin_oak": {"d_mm": 15.0, "cv": 0.12, "label": "Pin oak"},
    "red_oak": {"d_mm": 20.5, "cv": 0.15, "label": "Red oak"},
    "bur_oak": {"d_mm": 32.0, "cv": 0.20, "label": "Bur oak"},
    "mixed": {"d_mm": 22.0, "cv": 0.32, "label": "Mixed hardwood"},
}


def logit(z: float) -> float:
    z = float(np.clip(z, -20.0, 20.0))
    return 1.0 / (1.0 + math.exp(-z))


def f_opt_hz(d_mm: float) -> float:
    """Low-frequency agitation suited to heavy wooden seeds."""
    return float(np.clip(4.0 + 5.0 * (D_CELL_MM / max(d_mm, 8.0)), 3.0, 11.0))


def sample_diameters(rng: np.random.Generator, d_mm: float, cv: float, n: int) -> np.ndarray:
    sigma = math.sqrt(math.log(1.0 + cv * cv))
    mu = math.log(d_mm) - 0.5 * sigma * sigma
    return rng.lognormal(mu, sigma, n)


def outcome_probs(d_mm: float, f_eff: float, f_cmd: float, slip: float, p_kpa: float,
                  h_mm: float, speed_scale: float, vision: float) -> tuple[float, float, float, float]:
    """Return (p_miss, p_single, p_double, sigma_m) for one metering event."""
    f_star = f_opt_hz(d_mm)
    agitation = math.exp(-((f_cmd - f_star) / 2.6) ** 2)
    too_big = max(d_mm - D_CELL_MM, 0.0) / D_CELL_MM
    too_small = max(D_CELL_MM - d_mm, 0.0) / D_CELL_MM
    harsh = max(f_eff - 10.0, 0.0) / 10.0
    overspeed = max(speed_scale - 1.08, 0.0)
    underspeed = max(0.92 - speed_scale, 0.0)
    depth_err = abs(h_mm - H_STAR_MM) / H_STAR_MM

    # Vision lowers the command when a bridge is likely, which is folded into speed_scale
    # and into an explicit relief term so that a merely slower wheel is not credited.
    z_m = (
        -1.55
        + 5.2 * too_big
        + 2.4 * harsh
        + 2.6 * slip
        + 2.2 * overspeed
        + 1.4 * underspeed
        + 1.3 * depth_err
        - 1.7 * agitation * (0.35 + too_big)
        - 0.85 * vision * agitation
    )
    z_d = (
        -2.05
        + 3.6 * too_small
        + 1.7 * (p_kpa / 8.0) * (0.4 + too_small)
        + 1.1 * max(6.0 - f_cmd, 0.0) / 6.0 * too_small
        + 0.6 * overspeed
        - 0.9 * vision
        - 0.5 * agitation
    )
    p_m = logit(z_m)
    p_d = logit(z_d) * (1.0 - p_m)
    p_s = max(0.0, 1.0 - p_m - p_d)
    sigma = X_T_M * (
        0.07
        + 0.045 * harsh
        + 0.035 * depth_err
        + 0.03 * abs(speed_scale - 1.0)
        + 0.02 * (1.0 - 0.6 * vision)
        + 0.015 * abs(f_cmd - f_star) / 6.0
    )
    return p_m, p_s, p_d, sigma


def mpc_choose(d_mean: float, cv: float, f_dist: float, v: float) -> tuple[float, float, float]:
    """One-step grid search of the factor-coordination cost in Chapter 2."""
    best = None
    best_u = (6.0, 1.0, H_STAR_MM)
    slip = 0.02  # radar speed; slip is rejected by the electronic drive
    for f_cmd in np.linspace(2.0, 12.0, 11):
        for p_kpa in (0.0, 1.0, 2.0, 3.5, 5.0):
            h_mm = H_STAR_MM
            f_eff = math.sqrt(f_dist ** 2 * 0.35 + f_cmd ** 2)
            # Expected probabilities on the mean diameter, inflated by the lot CV.
            d_hi = d_mean * (1.0 + cv)
            d_lo = d_mean * (1.0 - cv)
            j = 0.0
            for d_mm, w in ((d_mean, 0.5), (d_hi, 0.25), (d_lo, 0.25)):
                p_m, p_s, p_d, sigma = outcome_probs(
                    d_mm, f_eff, f_cmd, slip, p_kpa, h_mm, 1.0, vision=0.0
                )
                j += w * (9.0 * p_m + 6.0 * p_d + 4.0 * (sigma / X_T_M) + 0.004 * f_cmd + 0.01 * p_kpa)
            if best is None or j < best:
                best = j
                best_u = (float(f_cmd), float(p_kpa), float(h_mm))
    return best_u


def simulate_case(lot: str, v: float, slope_deg: float, f_dist: float, mode: str, rng: np.random.Generator) -> dict:
    spec = LOTS[lot]
    diameters = sample_diameters(rng, spec["d_mm"], spec["cv"], N_EVENTS)
    d_mean = float(np.mean(diameters))
    use_radar = mode != "ground"
    use_vision = mode in {"vision", "both"}
    use_mpc = mode in {"mpc", "both"}

    if use_mpc:
        f_cmd, p_kpa, h_cmd = mpc_choose(d_mean, spec["cv"], f_dist, v)
    else:
        f_cmd, p_kpa, h_cmd = 0.0, 3.0, H_STAR_MM

    # Wheel slip grows with slope. Electronic drive uses radar and rejects it.
    slip_mean = (0.07 + 0.012 * slope_deg) if not use_radar else 0.015
    positions = []
    cursor = 0.0
    n_miss = 0
    n_single = 0
    n_double = 0
    bridge_events = 0
    energy = 0.0

    # Vision state: a slow occupancy estimate.
    rho = 0.55
    for k, d_mm in enumerate(diameters):
        slip = float(np.clip(rng.normal(slip_mean, 0.015), 0.0, 0.40))
        h_mm = h_cmd + float(rng.normal(0.0, 1.2 if use_mpc else 6.5))
        # Disturbance and commanded agitation both reach the seed box.
        f_eff = math.sqrt((f_dist * (1.0 if not use_mpc else 0.55)) ** 2 + f_cmd ** 2)

        # Feed-speed command.
        if mode == "ground":
            speed_scale = 1.0 - slip
        else:
            # Fuzzy-PID tracking residual after slip rejection.
            track_err = float(rng.normal(0.0, 0.012))
            speed_scale = 1.0 + track_err
            if use_vision:
                # Bridge index from diameter spread and a lagged occupancy state.
                circ_penalty = min(1.0, spec["cv"] / 0.25)
                bridge = rho * (0.45 + 0.55 * circ_penalty) * (1.0 + max(d_mm - D_CELL_MM, 0.0) / D_CELL_MM)
                if bridge > 0.42:
                    bridge_events += 1
                    relief = min(0.22, 0.55 * (bridge - 0.42))
                    speed_scale *= 1.0 - relief
                    rho = 0.92 * rho + 0.08 * 0.48
                else:
                    rho = 0.92 * rho + 0.08 * (0.50 + 0.15 * circ_penalty)
            else:
                rho = min(0.95, rho + 0.002 * max(d_mm - D_CELL_MM, 0.0))

        p_m, p_s, p_d, sigma = outcome_probs(
            float(d_mm), f_eff, f_cmd, slip if mode == "ground" else slip * 0.25,
            p_kpa, h_mm, speed_scale, 1.0 if use_vision else 0.0,
        )
        draw = rng.random()
        # Interval advances with the realized meter rate. speed_scale < 1 lengthens spacing.
        interval = X_T_M / max(speed_scale, 0.55)
        cursor += interval
        if draw < p_m:
            n_miss += 1
        elif draw < p_m + p_s:
            n_single += 1
            positions.append(cursor + float(rng.normal(0.0, sigma)))
        else:
            n_double += 1
            jitter = abs(float(rng.normal(0.0, 0.35 * sigma)))
            positions.append(cursor + float(rng.normal(0.0, sigma)))
            positions.append(cursor + 0.22 * X_T_M + jitter)

        omega = speed_scale * (2.0 * math.pi * (v / X_T_M) / 12.0)  # 12-cell wheel
        energy += 0.08 * omega ** 2 + 0.012 * f_cmd ** 2 + 0.02 * p_kpa ** 2

    positions = np.array(sorted(p for p in positions if p > 0.0))
    spacings = np.diff(positions)
    spacings = spacings[spacings > 0.01]
    mult = spacings <= 0.5 * X_T_M
    qual = (spacings > 0.5 * X_T_M) & (spacings <= 1.5 * X_T_M)
    miss = spacings > 1.5 * X_T_M
    qfi = 100.0 * float(np.mean(qual)) if len(spacings) else 0.0
    mi = 100.0 * float(np.mean(mult)) if len(spacings) else 0.0
    miss_i = 100.0 * float(np.mean(miss)) if len(spacings) else 0.0
    if np.any(qual):
        precision = 100.0 * float(np.std(spacings[qual]) / np.mean(spacings[qual]))
    else:
        precision = 100.0
    duration_s = positions[-1] / v if len(positions) else 1.0
    qualified_per_min = float(np.sum(qual)) / duration_s * 60.0
    target_per_min = (v / X_T_M) * 60.0
    eta = 100.0 * qualified_per_min / target_per_min
    mean_power = energy / max(N_EVENTS, 1)
    return {
        "lot": lot,
        "label": spec["label"],
        "v_m_s": v,
        "slope_deg": slope_deg,
        "f_dist_hz": f_dist,
        "mode": mode,
        "qfi": round(qfi, 2),
        "multiple": round(mi, 2),
        "miss": round(miss_i, 2),
        "precision": round(precision, 2),
        "qualified_per_min": round(qualified_per_min, 1),
        "target_per_min": round(target_per_min, 1),
        "eta_pct": round(eta, 2),
        "mean_event_energy": round(mean_power, 4),
        "bridge_flags": int(bridge_events),
        "f_cmd": round(f_cmd, 2),
        "p_kpa": round(p_kpa, 2),
        "n_events": N_EVENTS,
        "n_spacings": int(len(spacings)),
        "event_miss_pct": round(100.0 * n_miss / N_EVENTS, 2),
        "event_single_pct": round(100.0 * n_single / N_EVENTS, 2),
        "event_double_pct": round(100.0 * n_double / N_EVENTS, 2),
    }


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    modes = ["ground", "pid", "vision", "mpc", "both"]
    scenarios = [
        ("red_oak", 0.70, 0.0, 6.0),
        ("red_oak", 1.10, 0.0, 6.0),
        ("red_oak", 1.10, 12.0, 6.0),
        ("red_oak", 1.10, 0.0, 16.0),
        ("pin_oak", 1.10, 0.0, 6.0),
        ("bur_oak", 0.70, 0.0, 6.0),
        ("bur_oak", 1.10, 8.0, 12.0),
        ("mixed", 1.10, 8.0, 12.0),
    ]
    rows = []
    for lot, v, slope, fdist in scenarios:
        for mode in modes:
            rows.append(simulate_case(lot, v, slope, fdist, mode, rng))

    # Repeat the central case for a small Monte Carlo of the composite gain.
    gains = []
    for rep in range(12):
        sub = np.random.default_rng(RNG_SEED + 17 * rep)
        base = simulate_case("mixed", 1.10, 8.0, 12.0, "ground", sub)
        full = simulate_case("mixed", 1.10, 8.0, 12.0, "both", sub)
        gains.append(full["eta_pct"] - base["eta_pct"])

    out = {
        "x_t_m": X_T_M,
        "d_cell_mm": D_CELL_MM,
        "h_star_mm": H_STAR_MM,
        "n_events": N_EVENTS,
        "rows": rows,
        "mixed_gain_mean": round(float(np.mean(gains)), 2),
        "mixed_gain_std": round(float(np.std(gains, ddof=1)), 2),
        "mixed_gains": [round(float(g), 2) for g in gains],
    }
    dest = Path(__file__).with_name("estimation_results.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    hdr = f"{'lot':<10} {'v':>4} {'sl':>4} {'fd':>4} {'mode':<8} {'QFI':>7} {'mult':>7} {'miss':>7} {'prec':>7} {'eta':>7} {'P':>5} {'f':>5}"
    print(hdr)
    for r in rows:
        print(
            f"{r['lot']:<10} {r['v_m_s']:4.2f} {r['slope_deg']:4.0f} {r['f_dist_hz']:4.0f} {r['mode']:<8} "
            f"{r['qfi']:7.2f} {r['multiple']:7.2f} {r['miss']:7.2f} {r['precision']:7.2f} {r['eta_pct']:7.2f} "
            f"{r['p_kpa']:5.1f} {r['f_cmd']:5.1f}"
        )
    print("mixed gain mean/std", out["mixed_gain_mean"], out["mixed_gain_std"])


if __name__ == "__main__":
    main()

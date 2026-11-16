"""
Reduced-order estimator for a longitudinal-axial thresher.

Two innovations, both distinct from rotor-speed / concave-clearance / fan-speed
vision loops:

  ZWGVC  zone-wise guide-vane controller (eight vane angles + feeder buffer)
  DRSS   delayed-return shoe scheduler (chaffer, sieve, stroke, frequency, gate)

Rotor speed, concave clearance, and cleaning-fan speed are exogenous schedules
of grain moisture. They are identical for every policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

# Geometry
L = 2.40
N = 8
DZ = L / N
R = 0.300
KAPPA = 0.20
ALPHA_MIN = 10.0
ALPHA_MAX = 35.0
DALPHA_MAX = 8.0

# Reference point at which rate coefficients equal 1
OMEGA0 = 920.0 * 2.0 * np.pi / 60.0
C0 = 18.0
M0 = 0.15
FAN0 = 1200.0

# Threshing and separation rates (1/s) at the reference point
LAM_TH0 = 7.4
LAM_SEP0 = 9.5
P_OMEGA = 0.65
P_C_TH = 0.40
P_C_SEP = 0.30
BETA_M_TH = 5.8
BETA_M_SEP = 3.2

# Load and shaft power. k_F and MU are calibrated so that a nominal wheat
# point sits near 1.4 kWh per tonne of grain. See Chapter 3.
K_F = 0.42
MU = 0.095
F_BLOCK = 1680.0  # N, local concave-station blockage threshold

# Shoe reference settings
TH_C0 = 16.0
TH_S0 = 12.0
F0 = 5.0
A0 = 22.0
Q0 = 3.2

# Tailings, sieve loss, impurity coefficients
PSI0 = 0.115
A_C = 0.055
A_F = 0.30
A_A = 0.40
A_Q = 0.70
A_M = 4.5
A_FAN = 0.50

ELL0 = 0.0090
B_F = 0.34
B_S = 0.060
B_A = 0.45
B_Q = 0.40
B_FAN = 0.60
B_M = 2.0

IOTA0 = 0.018
C_C = 0.06
C_F = 0.18
C_FAN = 0.35

# Secondary cleaner used when the return gate is partly closed
ETA_SEC = 0.94
P_SEC = 900.0  # W, only while sigma < 0.98

# Breakage (guarded constraint, not a vision objective)
LAM_B0 = 0.011
P_B = 1.8
M_B = 2.2

# Feeder-house buffer
M_BUF_MAX = 18.0  # kg


@dataclass(frozen=True)
class Crop:
    name: str
    phi: float
    th_scale: float
    sep_scale: float
    chaff: float
    break_scale: float


WHEAT = Crop("wheat", 0.46, 1.00, 1.00, 0.22, 1.00)
RICE = Crop("rice", 0.41, 0.82, 0.90, 0.28, 1.15)
SOY = Crop("soybean", 0.55, 1.08, 0.78, 0.16, 0.85)


def exogenous(moisture: float) -> tuple[float, float, float]:
    """Operator schedule shared by every policy. Not a decision variable."""
    m = float(moisture)
    rpm = float(np.clip(900.0 + 700.0 * (m - 0.14), 820.0, 1080.0))
    clearance = float(np.clip(15.0 + 40.0 * (m - 0.14), 12.0, 30.0))
    fan = float(np.clip(1100.0 + 1400.0 * (m - 0.14), 1000.0, 1450.0))
    return rpm * 2.0 * np.pi / 60.0, clearance, fan


def axial_speed(alpha_deg: np.ndarray, omega: float) -> np.ndarray:
    alpha = np.deg2rad(np.asarray(alpha_deg, dtype=float))
    return np.maximum(KAPPA * omega * R * np.tan(alpha), 0.08)


def rates(omega: float, clearance: float, moisture: float, crop: Crop):
    w = (omega / OMEGA0) ** P_OMEGA
    wet_th = np.exp(-BETA_M_TH * (moisture - M0))
    wet_sep = np.exp(-BETA_M_SEP * (moisture - M0))
    lam_th = LAM_TH0 * crop.th_scale * w * (C0 / clearance) ** P_C_TH * wet_th
    lam_sep = LAM_SEP0 * crop.sep_scale * (clearance / C0) ** P_C_SEP * wet_sep
    return float(lam_th), float(lam_sep)


def rotor_pass(alphas, q_biomass, moisture, crop: Crop, omega, clearance):
    """One pass of biomass through the eight zones. No recirculation yet."""
    lam_th, lam_sep = rates(omega, clearance, moisture, crop)
    vz = axial_speed(alphas, omega)
    dt = DZ / vz
    grain_in = q_biomass * crop.phi
    u = grain_in
    g = 0.0
    straw = q_biomass * (1.0 - crop.phi)
    forces = np.zeros(N)
    sep_grain = 0.0
    exposure = 0.0
    for i in range(N):
        dur = float(dt[i])
        newly = u * (1.0 - np.exp(-lam_th * dur))
        u = u - newly
        g = g + newly
        sep = g * (1.0 - np.exp(-lam_sep * dur))
        g = g - sep
        sep_grain += sep
        line = (u + g + straw) / float(vz[i])
        v_theta = omega * R
        forces[i] = K_F * line * (v_theta ** 2)
        exposure += dur * (v_theta / (OMEGA0 * R)) ** P_B
    chaff = crop.chaff * straw * (1.0 - np.exp(-lam_sep * float(np.sum(dt))))
    power = float(np.sum(MU * forces * omega * R))
    breakage = 1.0 - np.exp(-LAM_B0 * crop.break_scale * exposure * np.exp(M_B * max(moisture - 0.18, 0.0)))
    return {
        "unthreshed": float(u),
        "free_rear": float(g),
        "sep_grain": float(sep_grain),
        "chaff": float(chaff),
        "forces": forces,
        "power": power,
        "breakage": float(breakage),
        "vz": vz,
        "dt": dt,
        "grain_in": float(grain_in),
    }


def shoe_maps(q_grain, q_chaff, moisture, theta_c, theta_s, freq, amp, fan):
    q = max(q_grain + q_chaff, 1e-6)
    fan_r = fan / FAN0
    psi = PSI0 * np.exp(-A_C * (theta_c - TH_C0)) * np.exp(-A_F * (freq - F0))
    psi *= (A0 / max(amp, 8.0)) ** A_A
    psi *= (q / Q0) ** A_Q
    psi *= np.exp(A_M * (moisture - M0))
    psi *= np.exp(-0.045 * (theta_s - TH_S0))
    psi *= fan_r ** (-A_FAN)
    psi = float(np.clip(psi, 0.02, 0.60))
    ell = ELL0 * np.exp(B_F * (freq - F0)) * np.exp(-B_S * (theta_s - TH_S0))
    ell *= np.exp(B_A * (amp - A0) / A0)
    ell *= (q / Q0) ** B_Q
    ell *= fan_r ** B_FAN
    ell *= np.exp(B_M * max(moisture - 0.20, 0.0))
    ell = float(np.clip(ell, 0.001, 0.08))
    iota = IOTA0 * np.exp(C_C * (theta_c - TH_C0)) * np.exp(0.07 * (theta_s - TH_S0))
    iota /= 1.0 + C_F * max(freq - 3.5, 0.0)
    iota *= fan_r ** (-C_FAN)
    iota = float(np.clip(iota, 0.002, 0.06))
    return psi, ell, iota


def steady(alphas, shoe, sigma, q_fresh, moisture, crop: Crop):
    """Fixed point of the tailings loop. Returned material re-enters the rotor."""
    omega, clearance, fan = exogenous(moisture)
    theta_c, theta_s, freq, amp = shoe
    sigma = float(np.clip(sigma, 0.0, 1.0))
    q = float(q_fresh)
    last = None
    for _ in range(25):
        passed = rotor_pass(alphas, q, moisture, crop, omega, clearance)
        psi, ell, iota = shoe_maps(
            passed["sep_grain"], passed["chaff"], moisture, theta_c, theta_s, freq, amp, fan
        )
        shoe_mass = passed["sep_grain"] + passed["chaff"]
        tailings = psi * shoe_mass
        # Grain share of tailings tracks the grain share of the shoe feed.
        g_share = passed["sep_grain"] / max(shoe_mass, 1e-6)
        tail_grain = tailings * g_share
        tail_chaff = tailings - tail_grain
        returned = sigma * tailings
        # Biomass returned is tailings; fresh biomass is q_fresh.
        q_next = float(q_fresh + returned)
        last = (passed, psi, ell, iota, tailings, tail_grain, tail_chaff, returned, q)
        if abs(q_next - q) < 1e-4:
            q = q_next
            break
        q = 0.55 * q + 0.45 * q_next
    passed, psi, ell, iota, tailings, tail_grain, tail_chaff, returned, q_used = last
    # Recompute once at the converged inlet so reported flows match q_used.
    passed = rotor_pass(alphas, q_used, moisture, crop, omega, clearance)
    psi, ell, iota = shoe_maps(
        passed["sep_grain"], passed["chaff"], moisture, theta_c, theta_s, freq, amp, fan
    )
    shoe_mass = passed["sep_grain"] + passed["chaff"]
    tailings = psi * shoe_mass
    g_share = passed["sep_grain"] / max(shoe_mass, 1e-6)
    tail_grain = tailings * g_share
    returned = sigma * tailings
    sieve_loss = ell * passed["sep_grain"]
    # Grain in the diverted tailings that the secondary cleaner does not save.
    diverted_grain = (1.0 - sigma) * tail_grain
    sec_loss = diverted_grain * (1.0 - ETA_SEC)
    grain_fresh = float(q_fresh) * crop.phi
    # Unthreshed and free grain leaving the rear are charged to the fresh grain,
    # but recirculated grain is already inside q_used. Scale rotor losses by the
    # fresh share of the inlet so the same grain is not counted twice.
    fresh_share = float(q_fresh) / max(q_used, 1e-6)
    unth = passed["unthreshed"] * fresh_share
    free_rear = passed["free_rear"] * fresh_share
    loss_mass = unth + free_rear + sieve_loss * fresh_share + sec_loss
    loss_frac = loss_mass / max(grain_fresh, 1e-6)
    forces = passed["forces"]
    cv = float(np.std(forces) / max(np.mean(forces), 1e-6))
    power = passed["power"]
    if sigma < 0.98:
        power += P_SEC
    specific = power / max(grain_fresh, 1e-6) / 3600.0  # kWh/t
    loop_gain = float(sigma * min(0.95, 1.25 * psi / max(g_share, 0.25)) * 0.88)
    return {
        "loss_pct": 100.0 * loss_frac,
        "unth_pct": 100.0 * unth / grain_fresh,
        "sep_pct": 100.0 * free_rear / grain_fresh,
        "sieve_pct": 100.0 * sieve_loss * fresh_share / grain_fresh,
        "sec_pct": 100.0 * sec_loss / grain_fresh,
        "return_ratio": float(returned / max(grain_fresh, 1e-6)),
        "cv": cv,
        "specific_kwh_t": float(specific),
        "break_pct": 100.0 * passed["breakage"],
        "impurity_pct": 100.0 * iota,
        "loop_gain": loop_gain,
        "forces": forces.tolist(),
        "power_kw": power / 1000.0,
        "alphas": np.asarray(alphas, dtype=float).tolist(),
        "shoe": [float(theta_c), float(theta_s), float(freq), float(amp)],
        "sigma": sigma,
        "omega_rpm": float(omega * 60.0 / (2.0 * np.pi)),
        "clearance_mm": float(clearance),
        "fan_rpm": float(fan),
        "q_rotor": float(q_used),
        "residence_s": float(np.sum(DZ / axial_speed(alphas, omega))),
    }


BASE_ALPHA = np.array([18.0, 18.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0])
BASE_SHOE = (16.0, 12.0, 5.0, 22.0)
BASE_SIGMA = 1.0


def pack(x):
    alphas = x[:8]
    shoe = (x[8], x[9], x[10], x[11])
    sigma = x[12]
    return alphas, shoe, sigma


def cost_of(x, q, moisture, crop, mode: str):
    alphas, shoe, sigma = pack(x)
    if mode == "vanes":
        shoe = BASE_SHOE
        sigma = BASE_SIGMA
    elif mode == "shoe":
        alphas = BASE_ALPHA
    met = steady(alphas, shoe, sigma, q, moisture, crop)
    smooth = float(np.sum(np.diff(alphas) ** 2))
    block = float(np.sum(np.maximum(np.asarray(met["forces"]) - F_BLOCK, 0.0)))
    j = 0.0
    j += 6.5 * met["loss_pct"]
    j += 4.0 * met["cv"]
    j += 2.6 * met["specific_kwh_t"]
    j += 1.4 * met["impurity_pct"]
    j += 18.0 * max(met["break_pct"] - 1.25, 0.0) ** 2
    j += 28.0 * max(met["loop_gain"] - 0.16, 0.0)
    j += 0.012 * block
    j += 0.02 * smooth
    j += 7.0 * max(met["return_ratio"] - 0.09, 0.0)
    return j


def bounds_and_cons():
    bounds = [(ALPHA_MIN, ALPHA_MAX)] * 8
    bounds += [(10.0, 28.0), (6.0, 20.0), (3.5, 7.5), (14.0, 32.0), (0.25, 1.0)]
    cons = []
    for i in range(7):
        cons.append(
            {
                "type": "ineq",
                "fun": lambda x, i=i: DALPHA_MAX - abs(x[i + 1] - x[i]),
            }
        )
    return bounds, cons


def optimize(q, moisture, crop: Crop, mode: str):
    starts = [
        np.array([18, 18, 24, 24, 24, 24, 24, 24, 16, 12, 5.0, 22, 1.00]),
        np.array([14, 16, 18, 20, 22, 26, 30, 33, 20, 10, 4.4, 26, 0.65]),
        np.array([26, 22, 18, 15, 14, 16, 22, 30, 18, 14, 4.2, 20, 0.55]),
    ]
    bounds, cons = bounds_and_cons()
    best = None
    for x0 in starts:
        x0 = x0.astype(float)
        if mode == "vanes":
            x0[8:12] = np.array(BASE_SHOE)
            x0[12] = BASE_SIGMA
        if mode == "shoe":
            x0[:8] = BASE_ALPHA
        res = minimize(
            lambda x: cost_of(x, q, moisture, crop, mode),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 60, "ftol": 1e-6, "disp": False},
        )
        alphas, shoe, sigma = pack(res.x)
        if mode == "vanes":
            shoe, sigma = BASE_SHOE, BASE_SIGMA
        elif mode == "shoe":
            alphas = BASE_ALPHA.copy()
        met = steady(alphas, shoe, sigma, q, moisture, crop)
        met["success"] = bool(res.success)
        met["cost"] = float(cost_of(res.x, q, moisture, crop, mode))
        if best is None or met["cost"] < best["cost"]:
            best = met
    return best


def baseline(q, moisture, crop: Crop):
    return steady(BASE_ALPHA, BASE_SHOE, BASE_SIGMA, q, moisture, crop)


def run_suite():
    cases = [
        ("wheat-nominal", WHEAT, 6.0, 0.145),
        ("wheat-wet", WHEAT, 6.0, 0.22),
        ("wheat-dry-heavy", WHEAT, 8.0, 0.12),
        ("wheat-light", WHEAT, 3.5, 0.16),
        ("rice-nominal", RICE, 5.0, 0.18),
        ("rice-wet", RICE, 6.4, 0.25),
        ("soy-nominal", SOY, 4.5, 0.13),
        ("soy-wet", SOY, 4.2, 0.18),
    ]
    # Seasonal weights: a wheat-dominant harvest with a rice and soybean share.
    weights = {
        "wheat-nominal": 0.22,
        "wheat-wet": 0.14,
        "wheat-dry-heavy": 0.12,
        "wheat-light": 0.10,
        "rice-nominal": 0.14,
        "rice-wet": 0.12,
        "soy-nominal": 0.10,
        "soy-wet": 0.06,
    }
    out = {"cases": {}, "season": {}}
    keys = [
        "loss_pct",
        "unth_pct",
        "sep_pct",
        "sieve_pct",
        "sec_pct",
        "return_ratio",
        "cv",
        "specific_kwh_t",
        "break_pct",
        "impurity_pct",
        "loop_gain",
        "power_kw",
        "residence_s",
        "q_rotor",
    ]
    for name, crop, q, m in cases:
        print(f"solving {name} ...", flush=True)
        b = baseline(q, m, crop)
        p1 = optimize(q, m, crop, "vanes")
        p2 = optimize(q, m, crop, "shoe")
        p12 = optimize(q, m, crop, "both")
        out["cases"][name] = {
            "crop": crop.name,
            "q_fresh": q,
            "moisture": m,
            "baseline": {k: b[k] for k in keys} | {"forces": b["forces"], "alphas": b["alphas"], "sigma": b["sigma"], "shoe": b["shoe"], "rpm": b["omega_rpm"], "gap": b["clearance_mm"], "fan": b["fan_rpm"]},
            "vanes": {k: p1[k] for k in keys} | {"forces": p1["forces"], "alphas": p1["alphas"], "sigma": p1["sigma"], "shoe": p1["shoe"]},
            "shoe": {k: p2[k] for k in keys} | {"forces": p2["forces"], "alphas": p2["alphas"], "sigma": p2["sigma"], "shoe": p2["shoe"]},
            "both": {k: p12[k] for k in keys} | {"forces": p12["forces"], "alphas": p12["alphas"], "sigma": p12["sigma"], "shoe": p12["shoe"]},
        }
    season = {pol: {k: 0.0 for k in keys} for pol in ("baseline", "vanes", "shoe", "both")}
    for name, w in weights.items():
        for pol in season:
            for k in keys:
                season[pol][k] += w * out["cases"][name][pol][k]
    out["season"] = season
    out["weights"] = weights
    return out


def simulate_slug():
    """Twenty-five-second wheat slug. Online policy is a rate-limited step toward
    the nonlinear setpoint recomputed from the measured inlet, not a preview."""
    crop = WHEAT
    moisture = 0.17
    t = np.arange(0.0, 28.0, 0.2)
    q_header = 5.0 + 3.2 * np.exp(-0.5 * ((t - 12.0) / 2.2) ** 2)
    # Two closed-loop traces: baseline (no buffer, fixed settings) and joint policy.
    traces = {}
    for pol in ("baseline", "both"):
        alphas = BASE_ALPHA.copy()
        shoe = np.array(BASE_SHOE, dtype=float)
        sigma = BASE_SIGMA
        delay_steps = int(4.0 / 0.2)
        ret_hist = [0.0] * delay_steps
        buf = 2.0
        q_cmd = 5.0
        rows = []
        target = None
        for k, tk in enumerate(t):
            q_ret = ret_hist[k % delay_steps] if False else ret_hist[0]
            # shift later
            q_in_req = q_header[k]
            if pol == "both":
                # Feeder buffer: hold back inlet when the command is below the header.
                room = M_BUF_MAX - buf
                # release at least enough to respect the buffer cap
                q_cmd = min(q_in_req, q_cmd)
                overflow = max(0.0, buf + 0.2 * (q_in_req - q_cmd) - M_BUF_MAX) / 0.2
                q_to_rotor = q_cmd + overflow
                buf = float(np.clip(buf + 0.2 * (q_in_req - q_to_rotor), 0.0, M_BUF_MAX))
            else:
                q_to_rotor = q_in_req
                buf = 0.0
            q_rotor_fresh_equiv = q_to_rotor  # return added inside steady()
            # Use the algebraic loop at the current fresh command PLUS we inject
            # return by raising q. The steady() model already closes the loop, so
            # for a dynamic trace we instead run one rotor pass with q_to_rotor + q_ret.
            omega, clearance, fan = exogenous(moisture)
            passed = rotor_pass(alphas, q_to_rotor + q_ret, moisture, crop, omega, clearance)
            psi, ell, iota = shoe_maps(
                passed["sep_grain"], passed["chaff"], moisture,
                shoe[0], shoe[1], shoe[2], shoe[3], fan,
            )
            tail = psi * (passed["sep_grain"] + passed["chaff"])
            ret_now = sigma * tail
            ret_hist.append(ret_now)
            ret_hist.pop(0)
            grain = (q_to_rotor + q_ret) * crop.phi
            fresh_share = q_to_rotor / max(q_to_rotor + q_ret, 1e-6)
            loss = (
                passed["unthreshed"] * fresh_share
                + passed["free_rear"] * fresh_share
                + ell * passed["sep_grain"] * fresh_share
                + (1.0 - sigma) * psi * passed["sep_grain"] * (1.0 - ETA_SEC)
            )
            forces = passed["forces"]
            rows.append(
                {
                    "t": float(tk),
                    "q_header": float(q_header[k]),
                    "q_rotor": float(q_to_rotor + q_ret),
                    "buf": float(buf),
                    "front_N": float(forces[0]),
                    "cv": float(np.std(forces) / max(np.mean(forces), 1e-6)),
                    "loss_pct": float(100.0 * loss / max(q_to_rotor * crop.phi, 1e-6)),
                    "return": float(ret_now),
                    "sigma": float(sigma),
                }
            )
            if pol == "both" and k % 3 == 0:
                # Recompute a setpoint from the measured header rate. Rate-limit it.
                target = optimize(float(q_header[k]), moisture, crop, "both")
                a_star = np.array(target["alphas"])
                alphas = alphas + np.clip(a_star - alphas, -1.2, 1.2)
                sh_star = np.array(target["shoe"])
                shoe = shoe + np.clip(sh_star - shoe, -0.6, 0.6)
                sigma = float(np.clip(sigma + np.clip(target["sigma"] - sigma, -0.08, 0.08), 0.25, 1.0))
                # Buffer command: if front station is hot, admit less than the header.
                if forces[0] > 0.72 * F_BLOCK:
                    q_cmd = max(3.0, q_header[k] - 1.4)
                else:
                    q_cmd = q_header[k]
        traces[pol] = rows
    return traces


def main():
    suite = run_suite()
    print("simulating slug ...", flush=True)
    # The slug optimizer is expensive if called via full SLSQP every 0.6 s.
    # Cache setpoints on a feed grid and interpolate during the slug.
    grid_q = [3.5, 5.0, 6.5, 8.0, 9.0]
    grid = {}
    for q in grid_q:
        print(f"  slug grid q={q}", flush=True)
        grid[q] = optimize(q, 0.17, WHEAT, "both")
    suite["slug_grid"] = {
        str(q): {
            "alphas": grid[q]["alphas"],
            "shoe": grid[q]["shoe"],
            "sigma": grid[q]["sigma"],
            "loss_pct": grid[q]["loss_pct"],
            "cv": grid[q]["cv"],
        }
        for q in grid_q
    }
    slug = simulate_slug_from_grid(grid)
    # Downsample for the thesis tables
    suite["slug"] = {
        pol: rows[::5]
        for pol, rows in slug.items()
    }
    suite["slug_summary"] = {}
    for pol, rows in slug.items():
        loss = np.array([r["loss_pct"] for r in rows])
        front = np.array([r["front_N"] for r in rows])
        ret = np.array([r["return"] for r in rows])
        suite["slug_summary"][pol] = {
            "loss_peak": float(loss.max()),
            "loss_mean": float(loss.mean()),
            "front_peak": float(front.max()),
            "return_peak": float(ret.max()),
            "buf_peak": float(max(r["buf"] for r in rows)),
        }
    path = r"C:\Users\Administrator\Workspace\Own\Thesis\kw\main\model\results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(suite, f, indent=2)
    print(json.dumps(suite["season"], indent=2))
    print(json.dumps(suite["slug_summary"], indent=2))


def simulate_slug_from_grid(grid):
    crop = WHEAT
    moisture = 0.17
    qs = np.array(sorted(grid.keys()))
    t = np.arange(0.0, 28.0, 0.2)
    q_header = 5.0 + 3.2 * np.exp(-0.5 * ((t - 12.0) / 2.2) ** 2)
    traces = {}
    for pol in ("baseline", "both"):
        alphas = BASE_ALPHA.copy()
        shoe = np.array(BASE_SHOE, dtype=float)
        sigma = float(BASE_SIGMA)
        ret_hist = [0.0] * 20  # 4.0 s at 0.2 s
        buf = 2.0
        q_cmd = 5.0
        rows = []
        for k, tk in enumerate(t):
            q_ret = ret_hist[0]
            q_in_req = float(q_header[k])
            if pol == "both":
                # q_cmd may lie below the header (store) or above it (release).
                draw = max(0.0, q_cmd - q_in_req)
                q_to_rotor = q_in_req - max(0.0, q_in_req - q_cmd) + min(draw, buf / 0.2)
                overflow = max(0.0, buf + 0.2 * (q_in_req - q_to_rotor) - M_BUF_MAX) / 0.2
                q_to_rotor += overflow
                buf = float(np.clip(buf + 0.2 * (q_in_req - q_to_rotor), 0.0, M_BUF_MAX))
            else:
                q_to_rotor = q_in_req
            omega, clearance, fan = exogenous(moisture)
            passed = rotor_pass(alphas, q_to_rotor + q_ret, moisture, crop, omega, clearance)
            psi, ell, iota = shoe_maps(
                passed["sep_grain"], passed["chaff"], moisture,
                float(shoe[0]), float(shoe[1]), float(shoe[2]), float(shoe[3]), fan,
            )
            tail = psi * (passed["sep_grain"] + passed["chaff"])
            ret_now = float(sigma * tail)
            ret_hist.append(ret_now)
            ret_hist.pop(0)
            fresh_share = q_to_rotor / max(q_to_rotor + q_ret, 1e-6)
            loss = (
                passed["unthreshed"] * fresh_share
                + passed["free_rear"] * fresh_share
                + ell * passed["sep_grain"] * fresh_share
                + (1.0 - sigma) * psi * passed["sep_grain"] * (1.0 - ETA_SEC)
            )
            forces = passed["forces"]
            rows.append(
                {
                    "t": float(tk),
                    "q_header": q_in_req,
                    "q_rotor": float(q_to_rotor + q_ret),
                    "buf": float(buf),
                    "front_N": float(forces[0]),
                    "cv": float(np.std(forces) / max(np.mean(forces), 1e-6)),
                    "loss_pct": float(100.0 * loss / max(q_to_rotor * crop.phi, 1e-6)),
                    "return": ret_now,
                    "sigma": float(sigma),
                }
            )
            if pol == "both" and k % 3 == 0:
                # Linear interpolation of the stored nonlinear setpoints.
                a = np.vstack([grid[q]["alphas"] for q in qs])
                sh = np.vstack([grid[q]["shoe"] for q in qs])
                sg = np.array([grid[q]["sigma"] for q in qs])
                a_star = np.array([np.interp(q_in_req, qs, a[:, i]) for i in range(8)])
                sh_star = np.array([np.interp(q_in_req, qs, sh[:, i]) for i in range(4)])
                s_star = float(np.interp(q_in_req, qs, sg))
                alphas = alphas + np.clip(a_star - alphas, -1.2, 1.2)
                shoe = shoe + np.clip(sh_star - shoe, -0.5, 0.5)
                sigma = float(np.clip(sigma + np.clip(s_star - sigma, -0.08, 0.08), 0.25, 1.0))
                if forces[0] > 0.85 * F_BLOCK:
                    q_cmd = max(3.2, q_in_req - 1.6)
                else:
                    # Return the reserve toward 2 kg once the front station has cooled.
                    release = min(max(buf - 2.0, 0.0) / 3.0, 0.9)
                    q_cmd = q_in_req + release
        traces[pol] = rows
    return traces


if __name__ == "__main__":
    main()

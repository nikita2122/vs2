"""
Three-valve high-gravity beer blending simulation.

Compares:
  PID    industrial ratio PI with analyser trim and fixed header pressure
  LADRC  linear ADRC with fixed nominal gains
  PROP   pressure-compensated decoupled LADRC (inlets)
         + valve-authority delay-compensated ADRC (outlet and extract)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plant data. Flows are volumetric, m3/h. Pressures are bar gauge.
# Original extract is mass-basis Plato of the fermented beer.
# ---------------------------------------------------------------------------
RHO_H = 1012.0
RHO_W = 999.0
P_H0 = 16.0
P_STAR0 = 12.0
Q_H0 = 7.50
# Mass dilution: P = (rho_h Q_h P_h) / (rho_h Q_h + rho_w Q_w)
R0 = (RHO_H * P_H0 / P_STAR0 - RHO_H) / RHO_W
Q_W0 = R0 * Q_H0
Q_B0 = Q_H0 + Q_W0
RHO_B0 = (RHO_H * Q_H0 + RHO_W * Q_W0) / Q_B0

P_H_SUP0 = 4.00
P_W_SUP0 = 4.20
P_DOWN = 0.55
P_M0 = 2.40
X_H0, X_W0, X_B0 = 0.58, 0.55, 0.38

TAU_V = 2.4
STIC_S = 0.014
STIC_J = 0.005
L_PIPE = 14.0
D_OUT = 0.050
A_OUT = np.pi * (D_OUT / 2) ** 2
A_H = np.pi * (0.050 / 2) ** 2
A_W = np.pi * (0.038 / 2) ** 2
T_ANALYSER = 1.2
DT = 0.01
T_END = 560.0
RNG = np.random.default_rng(7)


def kv_from_point(q, x, dp, rho):
    return q / (x * np.sqrt(dp * 1000.0 / rho))


KV_H = kv_from_point(Q_H0, X_H0, P_H_SUP0 - P_M0, RHO_H)
KV_W = kv_from_point(Q_W0, X_W0, P_W_SUP0 - P_M0, RHO_W)
KV_B = kv_from_point(Q_B0, X_B0, P_M0 - P_DOWN, RHO_B0)


def valve_flow(kv, x, dp, rho):
    if dp <= 0.0 or x <= 0.0:
        return 0.0
    return float(kv * x * np.sqrt(dp * 1000.0 / rho))


def header_pressure(xh, xw, xb, ph, pw, rho_b, kv_h=None):
    if kv_h is None:
        kv_h = KV_H
    lo = P_DOWN + 1e-4
    hi = min(ph, pw) - 1e-4

    def f(pm):
        return (
            valve_flow(kv_h, xh, ph - pm, RHO_H)
            + valve_flow(KV_W, xw, pw - pm, RHO_W)
            - valve_flow(KV_B, xb, pm - P_DOWN, rho_b)
        )

    flo, fhi = f(lo), f(hi)
    if flo <= 0.0:
        return lo
    if fhi >= 0.0:
        return hi
    return float(brentq(f, lo, hi))


def installed_gains(qh, qw, qb, xh, xw, xb, ph, pw, pm):
    """Total derivatives of volumetric flow w.r.t. the three stem positions."""
    dph = max(ph - pm, 0.12)
    dpw = max(pw - pm, 0.12)
    dpb = max(pm - P_DOWN, 0.12)
    xh = max(xh, 0.05)
    xw = max(xw, 0.05)
    xb = max(xb, 0.05)
    qh = max(qh, 0.05)
    qw = max(qw, 0.05)
    qb = max(qb, 0.05)
    sh = qh / (2.0 * dph)
    sw = qw / (2.0 * dpw)
    sb = qb / (2.0 * dpb)
    den = sh + sw + sb
    g_hx = qh / xh
    g_wx = qw / xw
    g_bx = qb / xb
    g_hh = g_hx * (sw + sb) / den
    g_hw = -(sh / den) * g_wx
    g_wh = -(sw / den) * g_hx
    g_ww = g_wx * (sh + sb) / den
    g_bp = -(g_bx) / den  # d(pm)/d(xb)
    lam = 1.0 / (1.0 - (sh * sw) / ((sw + sb) * (sh + sb)))
    return {
        "Ghh": g_hh,
        "Ghw": g_hw,
        "Gwh": g_wh,
        "Gww": g_ww,
        "Gbp": g_bp,
        "RGA": lam,
        "Sh": sh,
        "Sw": sw,
        "Sb": sb,
    }


class Actuator:
    def __init__(self, x0):
        self.x = float(x0)
        self.moving = False

    def step(self, u):
        u = float(np.clip(u, 0.0, 1.0))
        err = u - self.x
        if (not self.moving) and abs(err) <= STIC_S:
            return self.x
        self.moving = True
        self.x += (1.0 - np.exp(-DT / TAU_V)) * err
        if abs(u - self.x) < STIC_J:
            self.moving = False
            self.x = u if abs(u - self.x) < STIC_J else self.x
        self.x = float(np.clip(self.x, 0.0, 1.0))
        return self.x


class PI:
    """Velocity-form PI with clamping anti-windup. Works for either sign of kp."""

    def __init__(self, kp, ti, umin, umax, u0):
        self.kp = float(kp)
        self.ti = float(ti)
        self.umin = umin
        self.umax = umax
        self.u = float(u0)
        self.err_prev = 0.0

    def step(self, err):
        pushing_high = self.u >= self.umax - 1e-9 and err * self.kp > 0.0
        pushing_low = self.u <= self.umin + 1e-9 and err * self.kp < 0.0
        du = self.kp * (err - self.err_prev)
        if not (pushing_high or pushing_low):
            du += self.kp * (DT / self.ti) * err
        self.u = float(np.clip(self.u + du, self.umin, self.umax))
        self.err_prev = float(err)
        return self.u


class LADRC:
    """First-order linear ADRC. Optional known cross-input."""

    def __init__(self, wc, wo, umin, umax):
        self.wc = wc
        self.wo = wo
        self.umin = umin
        self.umax = umax
        self.z1 = 0.0
        self.z2 = 0.0
        self.u = 0.0
        self.b_prev = None
        self.bc_prev = 0.0

    def reset(self, y, u, b_self, b_cross=0.0, u_cross=0.0):
        b = float(b_self)
        self.z1 = float(y)
        self.u = float(u)
        self.z2 = float(-b * u - b_cross * u_cross)
        self.b_prev = b
        self.bc_prev = float(b_cross)

    def step(self, y, r, b_self, b_cross=0.0, u_cross=0.0):
        b = float(b_self)
        if abs(b) < 1e-5:
            b = 1e-5 if b >= 0.0 else -1e-5
        b_cross = float(b_cross)
        # Keep z2 + b u invariant when the scheduled gain changes, so the
        # valve command does not jump just because a new pressure sample arrived.
        if self.b_prev is not None:
            self.z2 += (self.b_prev - b) * self.u + (self.bc_prev - b_cross) * u_cross
        self.b_prev = b
        self.bc_prev = b_cross
        e = y - self.z1
        self.z1 += DT * (self.z2 + b * self.u + b_cross * u_cross + 2.0 * self.wo * e)
        self.z2 += DT * (self.wo ** 2 * e)
        u0 = self.wc * (r - self.z1)
        u = (u0 - self.z2 - b_cross * u_cross) / b
        self.u = float(np.clip(u, self.umin, self.umax))
        return self.u


def dP_dR(r, p_h=P_H0):
    den = RHO_H + RHO_W * r
    return -(RHO_H * p_h * RHO_W) / den ** 2


def plato_of(qh, qw, p_h):
    den = RHO_H * qh + RHO_W * qw
    if den <= 1e-9:
        return 0.0
    return float(RHO_H * qh * p_h / den)


def scenario(t):
    """Exogenous signals. Returns ph, pw, p_h_true, qh_sp, p_star, kv_scale."""
    ph = P_H_SUP0
    pw = P_W_SUP0
    p_true = P_H0
    qh = Q_H0
    p_star = P_STAR0
    kv_scale = 1.0
    if t >= 50.0:
        ph = 3.05
    if 130.0 <= t < 210.0:
        pw = P_W_SUP0 + 0.28 * np.sin(2.0 * np.pi * (t - 130.0) / 26.0)
    if t >= 230.0:
        a = np.clip((t - 230.0) / 25.0, 0.0, 1.0)
        p_star = P_STAR0 + a * (11.20 - P_STAR0)
    if t >= 320.0:
        a = np.clip((t - 320.0) / 20.0, 0.0, 1.0)
        qh = Q_H0 + a * (9.00 - Q_H0)
    if t >= 400.0:
        a = np.clip((t - 400.0) / 40.0, 0.0, 1.0)
        p_true = P_H0 + a * 0.60
    if t >= 480.0:
        kv_scale = 0.90
    return ph, pw, p_true, qh, p_star, kv_scale


def authority_pm(qh, qw, ph, pw, rho_b):
    qb = qh + qw
    lo = P_DOWN + 0.40
    hi = min(ph, pw) - 0.40
    if hi <= lo:
        return float(np.clip(P_M0, P_DOWN + 0.3, min(ph, pw) - 0.3))
    grid = np.linspace(lo, hi, 100)
    best_pm, best_c = grid[0], 1e18
    for pm in grid:
        sf_h = np.sqrt(max(ph - pm, 1e-6) * 1000.0 / RHO_H)
        sf_w = np.sqrt(max(pw - pm, 1e-6) * 1000.0 / RHO_W)
        sf_b = np.sqrt(max(pm - P_DOWN, 1e-6) * 1000.0 / rho_b)
        xh = qh / (KV_H * sf_h)
        xw = qw / (KV_W * sf_w)
        xb = qb / (KV_B * sf_b)

        def pen(x, target):
            return (x - target) ** 2 + 15.0 * max(0.0, x - 0.82) ** 2 + 15.0 * max(0.0, 0.25 - x) ** 2

        cost = pen(xh, X_H0) + pen(xw, X_W0) + 0.25 * pen(xb, X_B0)
        if cost < best_c:
            best_c, best_pm = cost, pm
    return float(best_pm)


def simulate(mode: str):
    n = int(T_END / DT) + 1
    act_h, act_w, act_b = Actuator(X_H0), Actuator(X_W0), Actuator(X_B0)
    # nominal gains
    g0 = installed_gains(Q_H0, Q_W0, Q_B0, X_H0, X_W0, X_B0, P_H_SUP0, P_W_SUP0, P_M0)
    b_h0 = g0["Ghh"] / TAU_V
    b_w0 = g0["Gww"] / TAU_V
    b_p0 = g0["Gbp"] / TAU_V
    b_r0 = dP_dR(R0) / 2.2

    # PI tunings from the local gains (SIMC-like, then detuned on the delayed quality loop)
    pi_h = PI(kp=TAU_V / (g0["Ghh"] * 1.15), ti=TAU_V, umin=0.02, umax=0.98, u0=X_H0)
    pi_w = PI(kp=TAU_V / (g0["Gww"] * 1.15), ti=TAU_V, umin=0.02, umax=0.98, u0=X_W0)
    # pressure: positive error (pm - sp) opens the outlet, and Gbp is negative,
    # so the PI gain is negative
    pi_p = PI(kp=-TAU_V / (g0["Gbp"] * 3.5), ti=8.0, umin=0.02, umax=0.98, u0=X_B0)
    pi_r = PI(kp=2.2 / (dP_dR(R0) * 28.0), ti=22.0, umin=0.08, umax=0.85, u0=R0)

    wc_f, wo_f = 0.85, 2.5
    ad_h = LADRC(wc_f, wo_f, 0.02, 0.98)
    ad_w = LADRC(wc_f, wo_f, 0.02, 0.98)
    ad_p = LADRC(0.20, 0.80, 0.02, 0.98)
    if mode == "LADRC":
        ad_r = LADRC(0.055, 0.22, 0.08, 0.85)
        ad_h.reset(Q_H0, X_H0, b_h0)
        ad_w.reset(Q_W0, X_W0, b_w0)
        ad_p.reset(P_M0, X_B0, b_p0)
        ad_r.reset(P_STAR0, R0, b_r0)
    elif mode == "PROP":
        ad_r = LADRC(0.12, 0.48, 0.08, 0.85)
        ad_h.reset(Q_H0, X_H0, b_h0)
        ad_w.reset(Q_W0, X_W0, b_w0)
        ad_p.reset(P_M0, X_B0, b_p0)
        ad_r.reset(P_STAR0, R0, b_r0)
    else:
        ad_r = None

    # filtered measurements and coordinator state
    qh_f, qw_f, pm_f = Q_H0, Q_W0, P_M0
    xh_f, xw_f, xb_f = X_H0, X_W0, X_B0
    b_h_f, b_w_f = b_h0, b_w0
    b_hw_f, b_wh_f = g0["Ghw"] / TAU_V, g0["Gwh"] / TAU_V
    b_p_f = b_p0
    pm_ref_f = P_M0
    scale_f = 1.0
    bias = 0.0
    r_cmd = R0
    p_soft_f = P_STAR0
    u_h, u_w, u_b = X_H0, X_W0, X_B0
    uh_c, uw_c, ub_c = X_H0, X_W0, X_B0

    p_hist = np.full(n, P_STAR0)
    soft_hist = np.full(n, P_STAR0)
    p_meas = P_STAR0

    keys = [
        "t", "P", "Psoft", "Pmeas", "Pstar", "Qh", "Qw", "Qb",
        "vh", "vw", "vb", "xh", "xw", "xb", "pm", "pref", "ph", "pw",
        "R", "bias", "z2h", "wc_p", "eso_p", "Qhsp",
    ]
    rec = {k: np.zeros(n) for k in keys}

    noise_q = RNG.normal(0.0, 0.012, n)
    noise_qw = RNG.normal(0.0, 0.006, n)
    noise_p = RNG.normal(0.0, 0.0035, n)
    noise_pl = RNG.normal(0.0, 0.006, n)
    noise_x = RNG.normal(0.0, 0.0015, (n, 3))

    for k in range(n):
        t = k * DT
        ph, pw, p_true, qh_sp, p_star, kv_scale = scenario(t)
        kv_h = KV_H * kv_scale

        xh, xw, xb = act_h.x, act_w.x, act_b.x
        # density of the blend from the current volumetric flows, one pass
        # use previous flows' density; fixed-point once
        qh = valve_flow(kv_h, xh, ph - pm_f, RHO_H)
        qw = valve_flow(KV_W, xw, pw - pm_f, RHO_W)
        rho_b = (RHO_H * qh + RHO_W * qw) / max(qh + qw, 1e-6)
        pm = header_pressure(xh, xw, xb, ph, pw, rho_b, kv_h)
        qh = valve_flow(kv_h, xh, ph - pm, RHO_H)
        qw = valve_flow(KV_W, xw, pw - pm, RHO_W)
        qb = valve_flow(KV_B, xb, pm - P_DOWN, rho_b)
        p_now = plato_of(qh, qw, p_true)

        # measurements
        qh_m = qh + noise_q[k]
        qw_m = qw + noise_qw[k]
        pm_m = pm + noise_p[k]
        xh_m = xh + noise_x[k, 0]
        xw_m = xw + noise_x[k, 1]
        xb_m = xb + noise_x[k, 2]
        af = DT / 0.30
        qh_f += af * (qh_m - qh_f)
        qw_f += af * (qw_m - qw_f)
        pm_f += af * (pm_m - pm_f)
        xh_f += af * (xh_m - xh_f)
        xw_f += af * (xw_m - xw_f)
        xb_f += af * (xb_m - xb_f)

        # analyser: variable transport delay + lag
        vh = (qh / 3600.0) / A_H
        vw = (qw / 3600.0) / A_W
        vb = max((qb / 3600.0) / A_OUT, 0.2)
        tau_d = np.clip(L_PIPE / vb, 3.0, 30.0)
        p_hist[k] = p_now
        n_del = int(round(tau_d / DT))
        p_del = p_hist[k - n_del] if k >= n_del else P_STAR0
        p_meas += (DT / T_ANALYSER) * ((p_del + noise_pl[k]) - p_meas)

        # soft sensor from measured speeds, nominal mother-beer extract
        p_soft = plato_of(max(qh_f, 0.0), max(qw_f, 0.0), P_H0)
        p_soft_f += (DT / 0.4) * (p_soft - p_soft_f)
        soft_hist[k] = p_soft_f
        n_del2 = int(round((tau_d + T_ANALYSER) / DT))
        p_soft_del = soft_hist[k - n_del2] if k >= n_del2 else P_STAR0
        innov = p_meas - (p_soft_del + bias)
        bias += DT * innov / 28.0
        bias = float(np.clip(bias, -1.5, 1.5))
        p_ctrl_prop = p_soft_f + bias

        qw_sp_from_r = r_cmd * qh_sp
        g = installed_gains(max(qh_f, 0.2), max(qw_f, 0.1), max(qh_f + qw_f, 0.3),
                            xh_f, xw_f, xb_f, ph, pw, pm_f)
        # gains use the possibly drifted Kv only through measured flow, which is correct
        # (measured Q already reflects the drifted valve). Installed derivatives from
        # Q and dp remain valid. A 10% Kv error is therefore visible as flow loss
        # until the loop corrects the stem.
        b_h = g["Ghh"] / TAU_V
        b_w = g["Gww"] / TAU_V
        b_hw = g["Ghw"] / TAU_V
        b_wh = g["Gwh"] / TAU_V
        b_p = g["Gbp"] / TAU_V
        ag = DT / 2.5
        b_h_f += ag * (b_h - b_h_f)
        b_w_f += ag * (b_w - b_w_f)
        b_hw_f += ag * (b_hw - b_hw_f)
        b_wh_f += ag * (b_wh - b_wh_f)
        b_p_f += ag * (b_p - b_p_f)

        if mode == "PID":
            # quality trim from the delayed analyser only
            r_cmd = pi_r.step(p_meas - p_star)
            qw_sp = r_cmd * qh_sp
            u_h = pi_h.step(qh_sp - qh_f)
            u_w = pi_w.step(qw_sp - qw_f)
            u_b = pi_p.step(pm_f - P_M0)
            pm_ref = P_M0
            wc_now = 0.55
            z2h = 0.0
        elif mode == "LADRC":
            r_cmd = ad_r.step(p_meas, p_star, b_r0)
            qw_sp = r_cmd * qh_sp
            u_h = ad_h.step(qh_f, qh_sp, b_h0)
            u_w = ad_w.step(qw_f, qw_sp, b_w0)
            u_b = ad_p.step(pm_f, P_M0, b_p0)
            pm_ref = P_M0
            wc_now = 0.55
            z2h = ad_h.z2
        else:
            viol = max(0.0, xh_f - 0.74, 0.32 - xh_f, xw_f - 0.74, 0.32 - xw_f)
            scale_tgt = 1.0 + 1.2 * np.clip(viol / 0.12, 0.0, 1.0)
            scale_f += (DT / 2.0) * (scale_tgt - scale_f)
            wc_now = 0.20 * scale_f
            wo_now = 0.80 * scale_f
            ad_p.wc = wc_now
            ad_p.wo = min(wo_now, 1.6)
            rho_b_sp = (RHO_H * qh_sp + RHO_W * max(r_cmd, 0.05) * qh_sp) / max(qh_sp * (1.0 + max(r_cmd, 0.05)), 1e-6)
            pm_ref_raw = authority_pm(qh_sp, max(r_cmd, 0.05) * qh_sp, ph, pw, rho_b_sp)
            ref_err = pm_ref_raw - pm_ref_f
            if abs(ref_err) > 0.04:
                pm_ref_f += (DT / 4.0) * ref_err
            r_cmd = ad_r.step(p_ctrl_prop, p_star, dP_dR(max(r_cmd, 0.05)) / 2.2)
            qw_sp = r_cmd * qh_sp
            # Installed, pressure-dependent diagonal gains. Off-diagonal
            # interaction stays in the total disturbance (RGA is near 1.1).
            u_h = ad_h.step(qh_f, qh_sp, b_h_f)
            u_w = ad_w.step(qw_f, qw_sp, b_w_f)
            u_b = ad_p.step(pm_f, pm_ref_f, b_p_f)
            pm_ref = pm_ref_f
            z2h = ad_h.z2

        # Positioner command filter, identical for every controller.
        uh_c += (DT / 0.40) * (u_h - uh_c)
        uw_c += (DT / 0.40) * (u_w - uw_c)
        ub_c += (DT / 0.40) * (u_b - ub_c)
        act_h.step(uh_c)
        act_w.step(uw_c)
        act_b.step(ub_c)
        if mode != "PID":
            ad_h.u = uh_c
            ad_w.u = uw_c
            ad_p.u = ub_c

        rec["t"][k] = t
        rec["P"][k] = p_now
        rec["Psoft"][k] = p_ctrl_prop
        rec["Pmeas"][k] = p_meas
        rec["Pstar"][k] = p_star
        rec["Qh"][k] = qh
        rec["Qw"][k] = qw
        rec["Qb"][k] = qb
        rec["vh"][k] = vh
        rec["vw"][k] = vw
        rec["vb"][k] = vb
        rec["xh"][k] = xh
        rec["xw"][k] = xw
        rec["xb"][k] = xb
        rec["pm"][k] = pm
        rec["pref"][k] = pm_ref
        rec["ph"][k] = ph
        rec["pw"][k] = pw
        rec["R"][k] = r_cmd
        rec["bias"][k] = bias
        rec["z2h"][k] = z2h
        rec["wc_p"][k] = wc_now
        rec["eso_p"][k] = ad_p.z2 if mode != "PID" else 0.0
        rec["Qhsp"][k] = qh_sp

    return rec


def iae(t, e, t0, t1):
    m = (t >= t0) & (t <= t1)
    return float(np.trapezoid(np.abs(e[m]), t[m]))


def peak(e, t, t0, t1):
    m = (t >= t0) & (t <= t1)
    return float(np.max(np.abs(e[m])))


def settle(t, e, t0, t1, band=0.05):
    m = np.where((t >= t0) & (t <= t1))[0]
    if len(m) == 0:
        return None
    # last time |e| enters and stays inside the band
    outside = np.abs(e[m]) > band
    if not np.any(outside):
        return 0.0
    last = np.where(outside)[0][-1]
    if last == len(m) - 1:
        return None
    return float(t[m[last + 1]] - t0)


def outside_time(t, e, t0, t1, band=0.05):
    m = (t >= t0) & (t <= t1)
    return float(np.trapezoid((np.abs(e[m]) > band).astype(float), t[m]))


def travel(x, t, t0, t1):
    m = np.where((t >= t0) & (t <= t1))[0]
    return float(np.sum(np.abs(np.diff(x[m]))))


def main():
    print("Design point")
    print(f"  R0={R0:.4f} Qw0={Q_W0:.3f} Qb0={Q_B0:.3f} rho_b={RHO_B0:.2f}")
    print(f"  KvH={KV_H:.3f} KvW={KV_W:.3f} KvB={KV_B:.3f}")
    g0 = installed_gains(Q_H0, Q_W0, Q_B0, X_H0, X_W0, X_B0, P_H_SUP0, P_W_SUP0, P_M0)
    for k, v in g0.items():
        print(f"  {k}={v:.4f}")
    v_h = (Q_H0 / 3600.0) / A_H
    v_w = (Q_W0 / 3600.0) / A_W
    v_b = (Q_B0 / 3600.0) / A_OUT
    print(f"  velocities m/s  beer={v_h:.3f} water={v_w:.3f} outlet={v_b:.3f}")
    print(f"  delay s = {L_PIPE / v_b:.2f} + analyser {T_ANALYSER}")
    print(f"  dP/dR = {dP_dR(R0):.3f}")

    results = {}
    for mode in ("PID", "LADRC", "PROP"):
        print("simulating", mode)
        results[mode] = simulate(mode)

    windows = [
        ("steady", 10, 50),
        ("pressure_step", 50, 130),
        ("water_osc", 130, 210),
        ("recipe", 230, 320),
        ("throughput", 320, 400),
        ("extract_drift", 400, 480),
        ("kv_drift", 480, 560),
        ("full", 0, 560),
    ]
    summary = {"design": {
        "R0": R0, "Qw0": Q_W0, "Qb0": Q_B0, "rho_b": RHO_B0,
        "KvH": KV_H, "KvW": KV_W, "KvB": KV_B,
        "vh": v_h, "vw": v_w, "vb": v_b,
        "delay_pipe": L_PIPE / v_b,
        "gains": g0,
        "dPdR": dP_dR(R0),
        "Ph": P_H_SUP0, "Pw": P_W_SUP0, "Pm": P_M0, "Pd": P_DOWN,
    }}
    for mode, rec in results.items():
        e = rec["P"] - rec["Pstar"]
        e_q = rec["Qh"] - rec["Qhsp"]
        summary[mode] = {}
        for name, a, b in windows:
            summary[mode][name] = {
                "IAE": iae(rec["t"], e, a, b),
                "IAE_flow": iae(rec["t"], e_q, a, b),
                "peak": peak(e, rec["t"], a, b),
                "settle_0p05": settle(rec["t"], e, a, b),
                "time_outside_0p05": outside_time(rec["t"], e, a, b),
                "travel_h": travel(rec["xh"], rec["t"], a, b),
                "travel_w": travel(rec["xw"], rec["t"], a, b),
                "travel_b": travel(rec["xb"], rec["t"], a, b),
            }
        print(mode, json.dumps(summary[mode]["full"]))
        tt = rec["t"]
        for ts in (40, 80, 160, 280, 360, 450, 520):
            i = int(ts / DT)
            print(
                f"  t={ts:6.0f} P={rec['P'][i]:7.3f} sp={rec['Pstar'][i]:6.2f} "
                f"Qh={rec['Qh'][i]:6.2f} Qw={rec['Qw'][i]:6.2f} "
                f"x=({rec['xh'][i]:.2f},{rec['xw'][i]:.2f},{rec['xb'][i]:.2f}) "
                f"pm={rec['pm'][i]:5.2f}"
            )

    out_json = FIG / "metrics.json"
    # gains contain numpy floats
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, float) and (np.isnan(o)):
            return None
        return o

    out_json.write_text(json.dumps(conv(summary), indent=2), encoding="utf-8")

    # downsample for lighter pdf figures
    step = 8  # 0.08 s
    colors = {"PID": "#1f4e79", "LADRC": "#c47b00", "PROP": "#1b7a3a"}
    labels = {
        "PID": "PID ratio control",
        "LADRC": "Fixed-gain LADRC",
        "PROP": "Proposed ADRC",
    }
    events = [50, 130, 210, 230, 320, 400, 480]

    def deco(ax):
        for ev in events:
            ax.axvline(ev, color="0.75", lw=0.6, zorder=0)
        ax.set_xlim(0, 560)
        ax.grid(True, alpha=0.35)

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    t = results["PID"]["t"][::step]
    ax.plot(t, results["PID"]["Pstar"][::step], "k--", lw=1.0, label="Setpoint")
    for mode in ("PID", "LADRC", "PROP"):
        ax.plot(t, results[mode]["P"][::step], color=colors[mode], lw=1.15, label=labels[mode])
    ax.axhline(0, color="none")
    deco(ax)
    ax.set_ylabel("Original extract (°P)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "plato_full.pdf")
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=False)
    for ax, (t0, t1, title) in zip(axes, (
        (40, 140, "Supply-pressure step of the high-gravity beer"),
        (220, 340, "Recipe change and the start of the throughput ramp"),
    )):
        m = (results["PID"]["t"] >= t0) & (results["PID"]["t"] <= t1)
        tt = results["PID"]["t"][m][::2]
        ax.plot(tt, results["PID"]["Pstar"][m][::2], "k--", lw=1.0, label="Setpoint")
        for mode in ("PID", "LADRC", "PROP"):
            ax.plot(tt, results[mode]["P"][m][::2], color=colors[mode], lw=1.15, label=labels[mode])
        ax.set_ylabel("Original extract (°P)")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(FIG / "plato_zoom.pdf")
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.6), sharex=True)
    series = (("Qh", "High-gravity beer flow (m$^3$/h)"),
              ("Qw", "Dilution-water flow (m$^3$/h)"),
              ("Qb", "Final-beer flow (m$^3$/h)"))
    for ax, (key, ylab) in zip(axes, series):
        for mode in ("PID", "LADRC", "PROP"):
            ax.plot(t, results[mode][key][::step], color=colors[mode], lw=1.0, label=labels[mode])
        deco(ax)
        ax.set_ylabel(ylab)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "flows.pdf")
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.6), sharex=True)
    for ax, (key, ylab) in zip(axes, (
        ("xh", "High-gravity beer valve, stem fraction"),
        ("xw", "Dilution-water valve, stem fraction"),
        ("xb", "Outlet valve, stem fraction"),
    )):
        for mode in ("PID", "LADRC", "PROP"):
            ax.plot(t, results[mode][key][::step], color=colors[mode], lw=1.0, label=labels[mode])
        ax.axhspan(0.30, 0.74, color="#1b7a3a", alpha=0.06, zorder=0)
        deco(ax)
        ax.set_ylabel(ylab)
        ax.set_ylim(0.0, 1.0)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "valves.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(t, results["PROP"]["ph"][::step], color="#7a1f1f", lw=1.0, label="Beer supply pressure")
    ax.plot(t, results["PROP"]["pw"][::step], color="#3d4c8a", lw=1.0, label="Water supply pressure")
    ax.plot(t, results["PID"]["pm"][::step], color=colors["PID"], lw=1.05, label="Header, PID")
    ax.plot(t, results["LADRC"]["pm"][::step], color=colors["LADRC"], lw=1.05, label="Header, fixed-gain LADRC")
    ax.plot(t, results["PROP"]["pm"][::step], color=colors["PROP"], lw=1.15, label="Header, proposed")
    ax.plot(t, results["PROP"]["pref"][::step], color=colors["PROP"], lw=1.0, ls="--", label="Proposed header setpoint")
    deco(ax)
    ax.set_ylabel("Pressure (bar, gauge)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="best", fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "pressures.pdf")
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True)
    rec = results["PROP"]
    axes[0].plot(t, rec["P"][::step], color="#111111", lw=1.15, label="True original extract")
    axes[0].plot(t, rec["Pmeas"][::step], color="#1f4e79", lw=1.0, label="Delayed analyser")
    axes[0].plot(t, rec["Psoft"][::step], color="#1b7a3a", lw=1.0, label="Corrected soft sensor")
    axes[0].plot(t, rec["Pstar"][::step], "k--", lw=0.9, label="Setpoint")
    deco(axes[0])
    axes[0].set_ylabel("Original extract (°P)")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(t, rec["bias"][::step], color="#6b3fa0", lw=1.1)
    deco(axes[1])
    axes[1].set_ylabel("Soft-sensor bias (°P)")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(FIG / "softsensor.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.plot(t, results["PROP"]["wc_p"][::step], color="#1b7a3a", lw=1.2)
    deco(ax)
    ax.set_ylabel("Outlet-loop bandwidth (rad/s)")
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(FIG / "bandwidth.pdf")
    plt.close()

    # bar chart of IAE on the four main upset windows
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    names = ["pressure_step", "water_osc", "recipe", "throughput", "extract_drift", "kv_drift"]
    nice = ["Pressure\nstep", "Water\noscillation", "Recipe\nchange", "Throughput\nramp", "Extract\ndrift", "Valve-gain\ndrift"]
    x = np.arange(len(names))
    width = 0.25
    for i, mode in enumerate(("PID", "LADRC", "PROP")):
        vals = [summary[mode][nm]["IAE"] for nm in names]
        ax.bar(x + (i - 1) * width, vals, width, color=colors[mode], label=labels[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(nice)
    ax.set_ylabel("IAE of original extract (°P·s)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG / "iae_bars.pdf")
    plt.close()

    # velocities at design, small reference figure data already printed
    np.savez(FIG / "results.npz", **{f"{m}_{k}": results[m][k][::step] for m in results for k in ("t", "P", "Qh", "xh")})
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()

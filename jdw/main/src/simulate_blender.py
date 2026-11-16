# -*- coding: utf-8 -*-
"""Case-study simulator for the high-gravity beer blending thesis.

The plant is a three-valve blend skid: high-gravity beer, deaerated dilution
water, and a final back-pressure valve, plus a carbon-dioxide injector.
Numbers written to ../generated are the source of the tables in the thesis.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, least_squares, minimize

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
GEN = ROOT / "generated"
FIG.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(29)

R_VALVE = 50.0
T_PROC = 2.0
RHO_ETH = 789.24
SALT = 0.10  # g/L, representative beer salinity in the Rammert-Pahl term

# Pipe internals
D_H, D_W, D_F = 0.066, 0.053, 0.081
L_BLEND = 18.0
A_H = math.pi * D_H**2 / 4.0
A_W = math.pi * D_W**2 / 4.0
A_F = math.pi * D_F**2 / 4.0
V_MIX = 0.080  # m3
V_MIX_L = V_MIX * 1000.0

MU_H = 2.6e-3
MU_W = 1.67e-3


def balling_oe(Aw, Er):
    Aw = np.asarray(Aw, dtype=float)
    Er = np.asarray(Er, dtype=float)
    return 100.0 * (2.0665 * Aw + Er) / (100.0 + 1.0665 * Aw)


def density_linear(Aw, Er):
    return 999.94 - 1.62 * Aw + 4.05 * Er


def abv_from_aw(Aw, rho):
    return Aw * rho / RHO_ETH


def zeta_rammert(T, E_g_per_L, abv):
    """Rammert-Pahl absorption coefficient (g L^-1 bar^-1), as reproduced
    by Speers and MacIntosh (2013). E is real extract in g/L, abv in % v/v.
    """
    t = T
    a = (
        3.36764
        - 0.12723 * t
        + 0.0028256 * t**2
        - 3.3597e-5 * t**3
        + 1.5933e-7 * t**4
    )
    b = (
        0.4723
        - 0.02988 * t
        + 0.0011605 * t**2
        - 2.251e-5 * t**3
        + 1.5933e-7 * t**4
    )
    corr = E_g_per_L / 128.0 + abv / 43.0 + SALT / 27.0
    return a - b * corr


def haaland(Re, eps, D):
    Re = max(float(Re), 50.0)
    term = (eps / D / 3.7) ** 1.11 + 6.9 / Re
    return (1.0 / (-1.8 * math.log10(term))) ** 2


def phi_eq(lift):
    lift = float(np.clip(lift, 0.0, 1.0))
    return R_VALVE ** (lift - 1.0)


class Model:
    def __init__(
        self,
        Kv=(32.0, 12.0, 64.0),
        eps=2.0e-6,
        K_minor=14.0,
        zeta_scale=0.93,
        contraction=0.00012,
        FL=(0.90, 0.93, 0.86),
        p_bias=0.0,
        name="plant",
    ):
        self.Kv = np.array(Kv, dtype=float)
        self.eps = float(eps)
        self.K_minor = float(K_minor)
        self.zeta_scale = float(zeta_scale)
        self.contraction = float(contraction)
        self.FL = np.array(FL, dtype=float)
        self.p_bias = float(p_bias)
        self.name = name

    def density(self, Aw, Er):
        rho = density_linear(Aw, Er)
        rho *= 1.0 - self.contraction * Aw * Er
        return float(rho)

    def properties(self, Aw, Er, co2):
        rho = self.density(Aw, Er)
        abv = abv_from_aw(Aw, rho)
        oe = float(balling_oe(Aw, Er))
        E_g = Er / 100.0 * rho
        zeta = self.zeta_scale * float(zeta_rammert(T_PROC, E_g, abv))
        p_eq = co2 / max(zeta, 1e-6)
        return {
            "rho": rho,
            "abv": float(abv),
            "oe": oe,
            "zeta": float(zeta),
            "p_eq": float(p_eq),
            "E_g": float(E_g),
        }

    def hydraulics(self, lifts, p_h, p_w, p_b, Aw, Er, co2):
        """Quasi-steady network. Pressures are absolute bar.
        The transmitter bias is applied to the beer-header measurement that
        this model *believes*; the true plant passes p_bias=0 and the true header.
        """
        lh, lw, lf = [float(x) for x in lifts]
        prop = self.properties(Aw, Er, co2)
        rho_f = prop["rho"]
        # Inlet densities do not use the mixer state.
        # Caller passes mixer composition; inlet densities are supplied via
        # the headers helper below. Here inlet streams are approximated by
        # storing them on the object just before the call (set_inlets).
        rho_h = self._rho_h
        rho_w = self._rho_w
        sg_h = rho_h / 1000.0
        sg_w = rho_w / 1000.0
        sg_f = rho_f / 1000.0
        Kv_h = self.Kv[0] * phi_eq(lh)
        Kv_w = self.Kv[1] * phi_eq(lw)
        Kv_f = self.Kv[2] * phi_eq(lf)
        p_h_used = p_h + self.p_bias

        def q_of(Kv, dp, sg):
            if dp <= 1e-8 or Kv <= 0:
                return 0.0
            return Kv * math.sqrt(dp / sg)

        def residual(pm):
            qh = q_of(Kv_h, p_h_used - pm, sg_h)
            qw = q_of(Kv_w, p_w - pm, sg_w)
            m_dot = rho_h * qh + rho_w * qw  # kg/h
            qf_mass = m_dot / rho_f  # m3/h
            v = (qf_mass / 3600.0) / A_F
            mu = 0.0022
            Re = rho_f * max(v, 1e-6) * D_F / mu
            f = haaland(Re, self.eps, D_F)
            dp_pipe = (f * L_BLEND / D_F + self.K_minor) * rho_f * v * v / 2.0 / 1e5
            pd = pm - dp_pipe
            qf_valve = q_of(Kv_f, pd - p_b, sg_f)
            return qf_valve - qf_mass

        lo = p_b + 0.03
        hi = min(p_h_used, p_w) - 0.02
        if hi <= lo + 1e-4:
            pm = lo
        else:
            fa, fb = residual(lo), residual(hi)
            if fa == 0:
                pm = lo
            elif fb == 0:
                pm = hi
            elif fa * fb < 0:
                pm = brentq(residual, lo, hi, xtol=1e-5, maxiter=80)
            else:
                pm = lo if abs(fa) < abs(fb) else hi

        qh = q_of(Kv_h, p_h_used - pm, sg_h)
        qw = q_of(Kv_w, p_w - pm, sg_w)
        m_dot = rho_h * qh + rho_w * qw
        qf = m_dot / max(rho_f, 1.0)
        v_f = (qf / 3600.0) / A_F
        mu = 0.0022
        Re = rho_f * max(v_f, 1e-6) * D_F / mu
        f = haaland(Re, self.eps, D_F)
        dp_pipe = (f * L_BLEND / D_F + self.K_minor) * rho_f * v_f * v_f / 2.0 / 1e5
        pd = pm - dp_pipe
        # vena contracta
        def pvc(p1, p2, FL):
            dp = max(p1 - p2, 0.0)
            return p1 - dp / max(FL**2, 0.05)

        out = {
            "pm": float(pm),
            "pd": float(pd),
            "dp_pipe": float(dp_pipe),
            "qh": float(qh),
            "qw": float(qw),
            "qf": float(qf),
            "vh": float((qh / 3600.0) / A_H),
            "vw": float((qw / 3600.0) / A_W),
            "vf": float(v_f),
            "pvc_h": float(pvc(p_h_used, pm, self.FL[0])),
            "pvc_w": float(pvc(p_w, pm, self.FL[1])),
            "pvc_f": float(pvc(pd, p_b, self.FL[2])),
            "f": float(f),
            "Re": float(Re),
            "rho_f": rho_f,
            "rho_h": rho_h,
            "rho_w": rho_w,
            **prop,
        }
        return out

    def set_inlets(self, Aw_h, Er_h, Aw_w=0.0, Er_w=0.0):
        self._rho_h = self.density(Aw_h, Er_h)
        self._rho_w = self.density(Aw_w, Er_w)


def strip_co2(c, p_vc, zeta, eff=0.35):
    p_eq = c / max(zeta, 1e-6)
    if p_vc >= p_eq:
        return c, 0.0
    c_eq = zeta * max(p_vc, 0.05)
    lost = eff * max(c - c_eq, 0.0)
    return c - lost, lost


def steady_targets(model, w, co2_sp):
    """Achievable outlet at mother mass fraction w, with CO2 made up by dosing."""
    Aw = w * MOTHER["Aw"]
    Er = w * MOTHER["Er"]
    prop = model.properties(Aw, Er, co2_sp)
    return {"w": w, "Aw": Aw, "Er": Er, "oe": prop["oe"], "abv": prop["abv"], "co2": co2_sp}


# Inlet recipes
MOTHER0 = {"Aw": 5.40, "Er": 6.10, "CO2": 5.50, "DO": 8.0}
WATER0 = {"Aw": 0.0, "Er": 0.0, "CO2": 0.03, "DO": 12.0}
MOTHER = dict(MOTHER0)
WATER = dict(WATER0)


def make_plant():
    return Model(
        Kv=(32.0, 12.0, 64.0),
        eps=2.0e-6,
        K_minor=14.0,
        zeta_scale=0.93,
        contraction=0.00012,
        FL=(0.90, 0.93, 0.86),
        p_bias=0.0,
        name="plant",
    )


def make_nominal():
    """Controller model before commissioning: worn-valve guess, smooth-pipe error,
    no contraction, solubility scale left at the pure polynomial."""
    return Model(
        Kv=(29.4, 13.2, 58.0),
        eps=1.5e-5,
        K_minor=8.0,
        zeta_scale=1.00,
        contraction=0.0,
        FL=(0.90, 0.90, 0.90),
        p_bias=0.10,  # believes the biased transmitter
        name="nominal",
    )


def clone_model(m, **kw):
    data = dict(
        Kv=tuple(m.Kv),
        eps=m.eps,
        K_minor=m.K_minor,
        zeta_scale=m.zeta_scale,
        contraction=m.contraction,
        FL=tuple(m.FL),
        p_bias=m.p_bias,
        name=m.name,
    )
    data.update(kw)
    return Model(**data)


# ---------------------------------------------------------------------------
# Steady solve and linear algebra of the valve map
# ---------------------------------------------------------------------------

def mix_from_flows(hyd, mother, water, dose_gps):
    qh = hyd["qh"] / 3600.0 * 1000.0  # L/s
    qw = hyd["qw"] / 3600.0 * 1000.0
    qf = hyd["qf"] / 3600.0 * 1000.0
    mh = hyd["rho_h"] * hyd["qh"] / 3600.0
    mw = hyd["rho_w"] * hyd["qw"] / 3600.0
    mt = mh + mw
    if mt < 1e-9 or qf < 1e-9:
        return None
    w = mh / mt
    Aw = w * mother["Aw"]
    Er = w * mother["Er"]
    co2_in = (qh * mother["CO2"] + qw * water["CO2"] + dose_gps) / qf
    # beer-valve strip then mixer, then final-valve strip uses mixer zeta
    # Approximate inlet strip with mother zeta
    return {"w": w, "Aw": Aw, "Er": Er, "co2_unstrip": co2_in, "mh": mh, "mw": mw, "qf_Ls": qf}


def evaluate(model, lifts, dose, mother, water, headers, comp=None):
    model.set_inlets(mother["Aw"], mother["Er"], water["Aw"], water["Er"])
    if comp is None:
        # one pass: hydraulics at a guessed outlet composition, then update
        Aw_g = 0.7 * mother["Aw"]
        Er_g = 0.7 * mother["Er"]
        hyd = model.hydraulics(lifts, *headers, Aw_g, Er_g, 5.0)
        mixed = mix_from_flows(hyd, mother, water, dose)
        if mixed is None:
            return None
        hyd = model.hydraulics(lifts, *headers, mixed["Aw"], mixed["Er"], mixed["co2_unstrip"])
        mixed = mix_from_flows(hyd, mother, water, dose)
    else:
        hyd = model.hydraulics(lifts, *headers, comp["Aw"], comp["Er"], comp["CO2"])
        mixed = mix_from_flows(hyd, mother, water, dose)
    if mixed is None:
        return None
    prop = model.properties(mixed["Aw"], mixed["Er"], mixed["co2_unstrip"])
    c_out, lost = strip_co2(mixed["co2_unstrip"], hyd["pvc_f"], prop["zeta"])
    # DO volumetric blend plus pickup
    qh = hyd["qh"] / 3600.0 * 1000.0
    qw = hyd["qw"] / 3600.0 * 1000.0
    qf = max(hyd["qf"] / 3600.0 * 1000.0, 1e-9)
    do = (qh * mother["DO"] + qw * water["DO"]) / qf
    margin_break = max(0.0, prop["p_eq"] - hyd["pvc_f"])
    do = do + 30.0 * margin_break
    prop_out = model.properties(mixed["Aw"], mixed["Er"], c_out)
    return {
        **hyd,
        "Aw": mixed["Aw"],
        "Er": mixed["Er"],
        "w": mixed["w"],
        "co2": c_out,
        "co2_lost": lost,
        "DO": do,
        "oe": prop_out["oe"],
        "abv": prop_out["abv"],
        "p_eq": prop_out["p_eq"],
        "zeta": prop_out["zeta"],
        "rho_f": prop_out["rho"],
    }


def solve_steady(model, mother, water, headers, sp, x0=None):
    if x0 is None:
        x0 = np.array([0.72, 0.70, 0.68, 4.0])

    def cost(x):
        ev = evaluate(model, x[:3], x[3], mother, water, headers)
        if ev is None or ev["qf"] < 1.0:
            return 1e5
        c = 0.0
        c += ((ev["oe"] - sp["oe"]) / 0.05) ** 2
        c += ((ev["abv"] - sp["abv"]) / 0.03) ** 2
        c += ((ev["co2"] - sp["co2"]) / 0.05) ** 2
        c += ((ev["qf"] - sp["qf"]) / 0.4) ** 2
        c += 8.0 * max(0.0, ev["p_eq"] + 0.45 - ev["pm"]) ** 2
        c += 8.0 * max(0.0, ev["p_eq"] + 0.10 - ev["pvc_f"]) ** 2
        c += 4.0 * max(0.0, ev["DO"] - 30.0) ** 2 / 100.0
        c += 0.15 * ((ev["pm"] - (ev["p_eq"] + 0.95)) / 0.3) ** 2
        return c

    bounds = [(0.20, 0.88), (0.20, 0.88), (0.20, 0.88), (0.0, 15.0)]
    res = minimize(cost, x0, method="SLSQP", bounds=bounds, options={"maxiter": 60, "ftol": 1e-10})
    ev = evaluate(model, res.x[:3], res.x[3], mother, water, headers)
    return res.x, ev, float(res.fun)


def gain_matrix(model, lifts, dose, mother, water, headers, comp):
    """dy/du with y = (qh, qw, pm), u = (lh, lw, lf)."""
    base = model.hydraulics(lifts, *headers, comp["Aw"], comp["Er"], comp["CO2"])
    y0 = np.array([base["qh"], base["qw"], base["pm"]])
    G = np.zeros((3, 3))
    du = 0.012
    for j in range(3):
        lp = list(lifts)
        lp[j] += du
        yp = model.hydraulics(lp, *headers, comp["Aw"], comp["Er"], comp["CO2"])
        y1 = np.array([yp["qh"], yp["qw"], yp["pm"]])
        G[:, j] = (y1 - y0) / du
    RGA = G * np.linalg.inv(G).T
    return G, RGA


# ---------------------------------------------------------------------------
# Dynamic plant
# ---------------------------------------------------------------------------

class Skid:
    def __init__(self, model):
        self.model = model
        self.Aw = 3.78
        self.Er = 4.27
        self.CO2 = 5.2
        self.DO = 10.0
        self.pos = np.array([0.7, 0.7, 0.7])
        self.deadband = 0.0

    def set_state_from_eval(self, ev, lifts):
        self.Aw = ev["Aw"]
        self.Er = ev["Er"]
        self.CO2 = ev["co2"]
        self.DO = ev["DO"]
        self.pos = np.array(lifts, dtype=float)

    def _apply_stiction(self, cmd):
        cmd = np.array(cmd, dtype=float)
        out = self.pos.copy()
        for i in range(3):
            db = self.deadband if i == 1 else 0.0
            if abs(cmd[i] - self.pos[i]) + 1e-12 >= max(db, 0.0):
                if db > 0 and abs(cmd[i] - self.pos[i]) < db:
                    out[i] = self.pos[i]
                else:
                    out[i] = cmd[i]
            else:
                out[i] = self.pos[i]
        # The test above is tangled. Re-do cleanly.
        out = self.pos.copy()
        for i in range(3):
            db = self.deadband if i == 1 else 0.0
            if abs(cmd[i] - self.pos[i]) >= db - 1e-12:
                out[i] = cmd[i]
            else:
                out[i] = self.pos[i]
        self.pos = out
        return out

    def step(self, cmd, dose, mother, water, headers, dt):
        lifts = self._apply_stiction(cmd)
        self.model.set_inlets(mother["Aw"], mother["Er"])
        hyd = self.model.hydraulics(lifts, *headers, self.Aw, self.Er, self.CO2)
        qh = hyd["qh"] / 3600.0
        qw = hyd["qw"] / 3600.0
        mh = hyd["rho_h"] * qh
        mw = hyd["rho_w"] * qw
        M = max(self.model.density(self.Aw, self.Er) * V_MIX, 1.0)
        Aw = self.Aw + dt * (mh * (mother["Aw"] - self.Aw) + mw * (water["Aw"] - self.Aw)) / M
        Er = self.Er + dt * (mh * (mother["Er"] - self.Er) + mw * (water["Er"] - self.Er)) / M
        qh_L = hyd["qh"] / 3600.0 * 1000.0
        qw_L = hyd["qw"] / 3600.0 * 1000.0
        qf_L = max(hyd["qf"] / 3600.0 * 1000.0, 1e-9)
        co2_in = (qh_L * mother["CO2"] + qw_L * water["CO2"] + dose) / qf_L
        CO2 = self.CO2 + dt * (qf_L * (co2_in - self.CO2)) / V_MIX_L
        do_in = (qh_L * mother["DO"] + qw_L * water["DO"]) / qf_L
        DO = self.DO + dt * (qf_L * (do_in - self.DO)) / V_MIX_L
        self.Aw = float(np.clip(Aw, 0.0, 12.0))
        self.Er = float(np.clip(Er, 0.0, 15.0))
        self.CO2 = float(np.clip(CO2, 0.0, 12.0))
        self.DO = float(max(DO, 0.0))
        prop = self.model.properties(self.Aw, self.Er, self.CO2)
        c_out, lost = strip_co2(self.CO2, hyd["pvc_f"], prop["zeta"])
        margin_break = max(0.0, prop["p_eq"] - hyd["pvc_f"])
        do_out = self.DO + 30.0 * margin_break
        prop_o = self.model.properties(self.Aw, self.Er, c_out)
        return {
            **hyd,
            "Aw": self.Aw,
            "Er": self.Er,
            "oe": prop_o["oe"],
            "abv": prop_o["abv"],
            "co2": c_out,
            "co2_tank": self.CO2,
            "DO": do_out,
            "p_eq": self.model.properties(self.Aw, self.Er, c_out)["p_eq"],
            "lifts": lifts.copy(),
            "cmd": np.array(cmd, dtype=float),
            "lost": lost,
        }


def rate_limit(cmd, prev, dlmax):
    cmd = np.array(cmd, dtype=float)
    prev = np.array(prev, dtype=float)
    return np.clip(cmd, prev - dlmax, prev + dlmax)


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

class PIRatio:
    """Decentralised ratio station: Balling feedforward, flow PIs, pressure PI.

    The flow and pressure loops are incremental, so a quiet plant is not
    kicked off its operating point when the controller is switched on.
    """

    def __init__(self, p_sp):
        self.p_sp = p_sp
        self.ie = 0.0
        self.ic = 0.0
        self.prev_cmd = np.array([0.7, 0.7, 0.7])
        self.eh_prev = 0.0
        self.ew_prev = 0.0
        self.ep_prev = 0.0
        self.qh_sp = 14.0
        self.qw_sp = 6.0

    def reset_near(self, lifts):
        self.prev_cmd = np.array(lifts, dtype=float)
        self.eh_prev = 0.0
        self.ew_prev = 0.0
        self.ep_prev = 0.0

    def act(self, meas, sp, mother, water, dt, dlmax):
        Aw_h = mother["Aw"]
        Er_h = mother["Er"]

        def oe_of(w):
            return float(balling_oe(w * Aw_h, w * Er_h))

        try:
            w_oe = brentq(lambda w: oe_of(w) - sp["oe"], 0.35, 0.92)
        except ValueError:
            w_oe = 0.70

        def abv_of(w):
            Aw = w * Aw_h
            Er = w * Er_h
            return abv_from_aw(Aw, density_linear(Aw, Er))

        try:
            w_ab = brentq(lambda w: abv_of(w) - sp["abv"], 0.35, 0.92)
        except ValueError:
            w_ab = w_oe
        w_ff = 0.55 * w_oe + 0.45 * w_ab
        e_oe = sp["oe"] - meas["oe_d"]
        self.ie = float(np.clip(self.ie + e_oe * dt, -0.8, 0.8))
        w = float(np.clip(w_ff + 0.025 * e_oe + 0.006 * self.ie, 0.40, 0.90))
        rho_h = density_linear(Aw_h, Er_h)
        rho_w = 999.94
        rho_f = density_linear(w * Aw_h, w * Er_h)
        m_f = rho_f * sp["qf"]
        self.qh_sp = w * m_f / rho_h
        self.qw_sp = (1.0 - w) * m_f / rho_w
        eh = self.qh_sp - meas["qh"]
        ew = self.qw_sp - meas["qw"]
        ep = meas["pm"] - self.p_sp
        # Algebraic hydraulics: keep the discrete loop gain Ki*G*dt near 0.25.
        du_h = 0.018 * (eh - self.eh_prev) + 0.0018 * eh * dt
        du_w = 0.028 * (ew - self.ew_prev) + 0.0030 * ew * dt
        du_f = 0.12 * (ep - self.ep_prev) + 0.022 * ep * dt
        self.eh_prev, self.ew_prev, self.ep_prev = eh, ew, ep
        lh = float(np.clip(self.prev_cmd[0] + du_h, 0.18, 0.90))
        lw = float(np.clip(self.prev_cmd[1] + du_w, 0.18, 0.90))
        lf = float(np.clip(self.prev_cmd[2] + du_f, 0.18, 0.90))
        qf_L = max(meas["qf"], 0.1) / 3.6
        qh_L = max(meas["qh"], 0.0) / 3.6
        qw_L = max(meas["qw"], 0.0) / 3.6
        dose_ff = sp["co2"] * qf_L - mother["CO2"] * qh_L - water["CO2"] * qw_L
        e_c = sp["co2"] - meas["co2_d"]
        self.ic = float(np.clip(self.ic + e_c * dt, -4.0, 4.0))
        dose = float(np.clip(dose_ff + 0.6 * e_c + 0.15 * self.ic, 0.0, 15.0))
        cmd = rate_limit([lh, lw, lf], self.prev_cmd, dlmax * dt)
        self.prev_cmd = cmd
        return cmd, dose


class NMPC:
    def __init__(self, model, know_deadband=False):
        self.model = model
        self.know_deadband = know_deadband
        self.prev = np.array([0.7, 0.7, 0.7, 4.0])
        self.deadband = 0.0

    def reset_near(self, lifts, dose):
        self.prev = np.array([lifts[0], lifts[1], lifts[2], dose], dtype=float)

    def act(self, skid_state, sp, mother, water, headers, dlmax, dt_ctrl=4.0):
        # Shared measured state (mixer) is passed in skid_state.
        Aw0 = skid_state["Aw"]
        Er0 = skid_state["Er"]
        CO20 = skid_state["CO2"]
        DO0 = skid_state["DO"]
        model = self.model

        def rollout(u_blocks):
            Aw, Er, CO2, DO = Aw0, Er0, CO20, DO0
            prev = self.prev[:3].copy()
            cost = 0.0
            # two blocks, 3 steps of 4 s each = 12 s, then 3 more
            steps = (3, 3)
            ui = 0
            for nstep in steps:
                u = u_blocks[ui]
                ui += 1
                cmd = u[:3]
                # rate limit inside the prediction
                for _ in range(nstep):
                    cmd_r = rate_limit(cmd, prev, dlmax * 2.0)
                    # predicted stiction
                    if self.know_deadband and self.deadband > 0:
                        lifts = prev.copy()
                        for i in range(3):
                            db = self.deadband if i == 1 else 0.0
                            if abs(cmd_r[i] - prev[i]) >= db:
                                lifts[i] = cmd_r[i]
                            else:
                                lifts[i] = prev[i]
                    else:
                        lifts = cmd_r
                    prev = lifts
                    model.set_inlets(mother["Aw"], mother["Er"])
                    hyd = model.hydraulics(lifts, *headers, Aw, Er, CO2)
                    qh = hyd["qh"] / 3600.0
                    qw = hyd["qw"] / 3600.0
                    mh = hyd["rho_h"] * qh
                    mw = hyd["rho_w"] * qw
                    M = max(model.density(Aw, Er) * V_MIX, 1.0)
                    dt = 2.0
                    Aw = Aw + dt * (mh * (mother["Aw"] - Aw) + mw * (0.0 - Aw)) / M
                    Er = Er + dt * (mh * (mother["Er"] - Er) + mw * (0.0 - Er)) / M
                    qh_L = hyd["qh"] / 3.6
                    qw_L = hyd["qw"] / 3.6
                    qf_L = max(hyd["qf"] / 3.6, 1e-9)
                    co2_in = (qh_L * mother["CO2"] + qw_L * water["CO2"] + u[3]) / qf_L
                    CO2 = CO2 + dt * qf_L * (co2_in - CO2) / V_MIX_L
                    do_in = (qh_L * mother["DO"] + qw_L * water["DO"]) / qf_L
                    DO = DO + dt * qf_L * (do_in - DO) / V_MIX_L
                    Aw = float(np.clip(Aw, 0, 12))
                    Er = float(np.clip(Er, 0, 15))
                    CO2 = float(np.clip(CO2, 0, 12))
                    prop = model.properties(Aw, Er, CO2)
                    c_out, _ = strip_co2(CO2, hyd["pvc_f"], prop["zeta"])
                    prop_o = model.properties(Aw, Er, c_out)
                    do_out = DO + 30.0 * max(0.0, prop_o["p_eq"] - hyd["pvc_f"])
                    cost += ((prop_o["oe"] - sp["oe"]) / 0.05) ** 2
                    cost += ((prop_o["abv"] - sp["abv"]) / 0.03) ** 2
                    cost += ((c_out - sp["co2"]) / 0.05) ** 2
                    cost += 0.35 * ((hyd["qf"] - sp["qf"]) / 0.5) ** 2
                    cost += 6.0 * max(0.0, prop_o["p_eq"] + 0.40 - hyd["pm"]) ** 2
                    cost += 6.0 * max(0.0, prop_o["p_eq"] + 0.08 - hyd["pvc_f"]) ** 2
                    cost += 3.0 * max(0.0, do_out - 30.0) ** 2 / 80.0
                    cost += 0.05 * ((hyd["pm"] - (prop_o["p_eq"] + 0.95)) / 0.4) ** 2
                    dlu = lifts - self.prev[:3]
                    cost += 0.4 * float(np.dot(dlu, dlu)) / 0.01
            return cost

        x0 = np.concatenate([self.prev, self.prev])

        def obj(z):
            try:
                return rollout((z[:4], z[4:]))
            except Exception:
                return 1.0e6

        bounds = [(0.18, 0.90)] * 3 + [(0.0, 15.0)]
        bounds = bounds + bounds
        res = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 6, "ftol": 1e-3, "disp": False},
        )
        u = res.x[:4] if np.isfinite(res.fun) else self.prev.copy()
        cmd = u[:3].copy()
        if self.know_deadband and self.deadband > 0:
            # lead the water valve by the stick-band in the direction of travel
            du = cmd[1] - self.prev[1]
            if abs(du) > 0.004:
                cmd[1] = cmd[1] + math.copysign(self.deadband, du)
                cmd[1] = float(np.clip(cmd[1], 0.18, 0.90))
        cmd = rate_limit(cmd, self.prev[:3], dlmax * dt_ctrl)
        dose = float(u[3])
        self.prev = np.array([cmd[0], cmd[1], cmd[2], dose])
        return cmd, dose


# ---------------------------------------------------------------------------
# Commissioning estimation
# ---------------------------------------------------------------------------

def commissioning(plant, headers, mother, water):
    plant.set_inlets(mother["Aw"], mother["Er"])
    rows = []
    for k in range(28):
        lifts = RNG.uniform(0.40, 0.82, size=3)
        dose = float(RNG.uniform(2.0, 7.0))
        ev = evaluate(plant, lifts, dose, mother, water, headers)
        if ev is None:
            continue
        noise_q = RNG.normal(0, 0.03, size=3)
        noise_p = RNG.normal(0, 0.008, size=2)
        rows.append(
            {
                "l": lifts,
                "dose": dose,
                "qh": ev["qh"] + noise_q[0],
                "qw": ev["qw"] + noise_q[1],
                "qf": ev["qf"] + noise_q[2],
                "pm": ev["pm"] + noise_p[0],
                "pd": ev["pd"] + noise_p[1],
                "rho": ev["rho_f"] + RNG.normal(0, 0.15),
                "Aw": ev["Aw"],
                "Er": ev["Er"],
                "co2": ev["co2"],
                "true_pm": ev["pm"],
            }
        )
    data = rows
    # measured header is biased
    bias_true = 0.10
    p_h_meas = headers[0] + bias_true

    def unpack(theta):
        Kv = np.exp(theta[0:3])
        eps = math.exp(theta[3])
        K = math.exp(theta[4])
        bias = theta[5]
        return Kv, eps, K, bias

    def resid(theta):
        Kv, eps, K, bias = unpack(theta)
        m = Model(
            Kv=tuple(Kv),
            eps=eps,
            K_minor=K,
            zeta_scale=1.0,
            contraction=0.0,
            FL=(0.90, 0.90, 0.90),
            p_bias=0.0,
            name="est",
        )
        r = []
        for row in data:
            # The estimator sees p_h_meas and estimates bias so the model header
            # is p_h_meas - bias.
            m.set_inlets(mother["Aw"], mother["Er"])
            hyd = m.hydraulics(row["l"], p_h_meas - bias, headers[1], headers[2], row["Aw"], row["Er"], 5.2)
            r.extend(
                [
                    (hyd["qh"] - row["qh"]) / 0.08,
                    (hyd["qw"] - row["qw"]) / 0.05,
                    (hyd["qf"] - row["qf"]) / 0.08,
                    (hyd["pm"] - row["pm"]) / 0.02,
                ]
            )
        return np.array(r)

    theta0 = np.array(
        [
            math.log(29.4),
            math.log(13.2),
            math.log(58.0),
            math.log(1.5e-5),
            math.log(8.0),
            0.0,
        ]
    )
    fit = least_squares(resid, theta0, method="trf", xtol=1e-10, ftol=1e-10, max_nfev=80)
    Kv, eps, K, bias = unpack(fit.x)
    # standard errors
    J = fit.jac
    dof = max(len(fit.fun) - len(fit.x), 1)
    sigma2 = float(np.sum(fit.fun**2) / dof)
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(fit.x), np.nan)
    # SE of Kv is se(log Kv) * Kv
    se_Kv = se[0:3] * Kv
    se_eps = se[3] * eps
    se_K = se[4] * K
    se_bias = se[5]

    # Bench solubility: one sample, plant zeta vs polynomial
    prop = plant.properties(3.78, 4.27, 5.2)
    zeta_poly = float(zeta_rammert(T_PROC, prop["E_g"], prop["abv"]))
    # "measured" equilibrium pressure at 5.2 g/L
    p_meas = 5.2 / prop["zeta"] + float(RNG.normal(0, 0.01))
    zeta_scale_hat = 5.2 / (zeta_poly * p_meas)

    # Residual network on flow errors after the physical fit
    m_hat = Model(
        Kv=tuple(Kv),
        eps=eps,
        K_minor=K,
        zeta_scale=zeta_scale_hat,
        contraction=0.0,
        FL=(0.90, 0.93, 0.86),
        p_bias=0.0,
        name="hat",
    )
    X = []
    Y = []
    for row in data:
        m_hat.set_inlets(mother["Aw"], mother["Er"])
        hyd = m_hat.hydraulics(
            row["l"], p_h_meas - bias, headers[1], headers[2], row["Aw"], row["Er"], 5.2
        )
        X.append([row["l"][0], row["l"][1], row["l"][2], row["Aw"], row["Er"], hyd["pm"]])
        Y.append([row["qh"] - hyd["qh"], row["qw"] - hyd["qw"], row["qf"] - hyd["qf"]])
    X = np.array(X)
    Y = np.array(Y)
    # hold out last 6
    Xtr, Ytr = X[:-6], Y[:-6]
    Xte, Yte = X[-6:], Y[-6:]
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-6
    net = train_pinn( (Xtr - mu) / sd, Ytr, X, Y, mu, sd, m_hat, data, p_h_meas, bias, headers, mother)
    Yhat_te = net.predict((Xte - mu) / sd)
    rmse_before = float(np.sqrt(np.mean(Yte**2)))
    rmse_after = float(np.sqrt(np.mean((Yte - Yhat_te) ** 2)))

    # predictions for parity plot (all points, physical model)
    q_true = np.array([[r["qh"], r["qw"], r["qf"]] for r in data])
    q_nom = []
    q_hat = []
    nominal = make_nominal()
    for row in data:
        nominal.set_inlets(mother["Aw"], mother["Er"])
        # nominal uses biased pressure as if it were truth (p_bias built into model)
        hyd_n = nominal.hydraulics(row["l"], headers[0], headers[1], headers[2], row["Aw"], row["Er"], 5.2)
        m_hat.set_inlets(mother["Aw"], mother["Er"])
        hyd_h = m_hat.hydraulics(row["l"], p_h_meas - bias, headers[1], headers[2], row["Aw"], row["Er"], 5.2)
        corr = net.predict((np.array([[row["l"][0], row["l"][1], row["l"][2], row["Aw"], row["Er"], hyd_h["pm"]]]) - mu) / sd)[0]
        q_nom.append([hyd_n["qh"], hyd_n["qw"], hyd_n["qf"]])
        q_hat.append([hyd_h["qh"] + corr[0], hyd_h["qw"] + corr[1], hyd_h["qf"] + corr[2]])
    q_nom = np.array(q_nom)
    q_hat = np.array(q_hat)

    contr_samples = []
    for row in data:
        rho_lin = density_linear(row["Aw"], row["Er"])
        denom = row["Aw"] * row["Er"]
        if denom > 1.0 and rho_lin > 0:
            contr_samples.append((1.0 - row["rho"] / rho_lin) / denom)
    contr_hat = float(np.median(contr_samples)) if contr_samples else 0.0

    est = {
        "contraction": contr_hat,
        "contraction_true": 0.00012,
        "Kv": Kv.tolist(),
        "Kv_true": [32.0, 12.0, 64.0],
        "se_Kv": se_Kv.tolist(),
        "eps": eps,
        "eps_true": 2.0e-6,
        "se_eps": float(se_eps),
        "K": K,
        "K_true": 14.0,
        "se_K": float(se_K),
        "bias": float(bias),
        "bias_true": bias_true,
        "se_bias": float(se_bias),
        "zeta_scale": float(zeta_scale_hat),
        "zeta_scale_true": 0.93,
        "rmse_flow_before": rmse_before,
        "rmse_flow_after": rmse_after,
        "n": len(data),
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "cost": float(np.sum(fit.fun**2)),
        "sigma": float(math.sqrt(sigma2)),
        "loss_coeff": float(haaland(40000.0, eps, D_F) * L_BLEND / D_F + K),
        "loss_coeff_true": float(haaland(40000.0, 2.0e-6, D_F) * L_BLEND / D_F + 14.0),
    }
    return est, (q_true, q_nom, q_hat), m_hat, net, (mu, sd)


class ResidualNet:
    def __init__(self):
        rng = np.random.default_rng(7)
        self.W1 = rng.normal(0, 0.2, size=(8, 6))
        self.b1 = np.zeros(8)
        self.W2 = rng.normal(0, 0.2, size=(3, 8))
        self.b2 = np.zeros(3)

    def predict(self, Xn):
        Z = np.tanh(Xn @ self.W1.T + self.b1)
        return Z @ self.W2.T + self.b2

    def copy_from(self, W1, b1, W2, b2):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2


def train_pinn(Xn, Y, X_raw, Y_raw, mu, sd, model_hat, data, p_h_meas, bias, headers, mother):
    net = ResidualNet()
    # Pack parameters
    def pack(W1, b1, W2, b2):
        return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])

    def unpack(theta):
        i = 0
        W1 = theta[i : i + 48].reshape(8, 6)
        i += 48
        b1 = theta[i : i + 8]
        i += 8
        W2 = theta[i : i + 24].reshape(3, 8)
        i += 24
        b2 = theta[i : i + 3]
        return W1, b1, W2, b2

    theta = pack(net.W1, net.b1, net.W2, net.b2)
    # Adam
    m1 = np.zeros_like(theta)
    m2 = np.zeros_like(theta)
    lr = 0.02
    for step in range(900):
        W1, b1, W2, b2 = unpack(theta)
        Z = np.tanh(Xn @ W1.T + b1)
        Yhat = Z @ W2.T + b2
        err = Yhat - Y
        # data grad
        gW2 = (2.0 / len(Xn)) * (err.T @ Z) + 2e-3 * W2
        gb2 = (2.0 / len(Xn)) * err.sum(axis=0) + 2e-3 * b2
        dZ = err @ W2 * (1 - Z**2)
        gW1 = (2.0 / len(Xn)) * (dZ.T @ Xn) + 2e-3 * W1
        gb1 = (2.0 / len(Xn)) * dZ.sum(axis=0) + 2e-3 * b1
        # physics: mass residual of corrected flows, subsample
        if step % 5 == 0:
            phys = np.zeros_like(theta)
            # finite-difference physics gradient is expensive; use a penalty on
            # the corrected mass balance evaluated at the current prediction,
            # backpropagated only through Yhat (flows).
            # mass = rho_h*(qh+dqh)+rho_w*(qw+dqw)-rho_f*(qf+dqf)
            # We approximate rho from linear density of the row.
            take = slice(0, len(Xn), 1)
            dq = Yhat
            # rebuild quickly from training rows (data aligned with X order)
            rows = data[: len(Xn)]
            mass = []
            for i, row in enumerate(rows):
                rho_h = density_linear(mother["Aw"], mother["Er"])
                rho_w = 999.94
                rho_f = density_linear(row["Aw"], row["Er"])
                qh = row["qh"] - Y[i, 0] + dq[i, 0]  # model + correction = measured target? 
                # Physical model flow is measured - residual_target, plus correction:
                # q_model = q_meas - Y, q_corr = q_model + dq
                q_model = np.array(
                    [
                        row["qh"] - Y[i, 0],
                        row["qw"] - Y[i, 1],
                        row["qf"] - Y[i, 2],
                    ]
                )
                q = q_model + dq[i]
                mass.append(rho_h * q[0] + rho_w * q[1] - rho_f * q[2])
            mass = np.array(mass)
            # d(mass^2)/d(dq) = 2 mass * [rho_h, rho_w, -rho_f]
            g_dq = np.zeros_like(dq)
            for i, row in enumerate(rows):
                rho_h = density_linear(mother["Aw"], mother["Er"])
                rho_w = 999.94
                rho_f = density_linear(row["Aw"], row["Er"])
                g_dq[i] = (2.0 / len(rows)) * mass[i] * np.array([rho_h, rho_w, -rho_f]) / 500.0
            # add to W2, b2, W1, b1
            gW2 = gW2 + 0.15 * (g_dq.T @ Z)
            gb2 = gb2 + 0.15 * g_dq.sum(axis=0)
            dZ2 = g_dq @ W2 * (1 - Z**2)
            gW1 = gW1 + 0.15 * (dZ2.T @ Xn)
            gb1 = gb1 + 0.15 * dZ2.sum(axis=0)
        g = pack(gW1, gb1, gW2, gb2)
        m1 = 0.9 * m1 + 0.1 * g
        m2 = 0.999 * m2 + 0.001 * (g * g)
        m1h = m1 / (1 - 0.9 ** (step + 1))
        m2h = m2 / (1 - 0.999 ** (step + 1))
        theta = theta - lr * m1h / (np.sqrt(m2h) + 1e-8)
    W1, b1, W2, b2 = unpack(theta)
    net.copy_from(W1, b1, W2, b2)
    return net


# ---------------------------------------------------------------------------
# Closed-loop scenarios
# ---------------------------------------------------------------------------

def delayed(buf, value, delay_steps):
    buf.append(value)
    if len(buf) <= delay_steps:
        return buf[0]
    return buf[-1 - delay_steps]


def run_scenario(name, controller_name, controller, plant_model, mother_fun, water_fun, header_fun, sp_fun, t_end, event_time, dlmax=0.04):
    skid = Skid(plant_model)
    # initial steady state on the true plant at the initial setpoint
    mother = mother_fun(0.0)
    water = water_fun(0.0)
    headers = header_fun(0.0)
    sp0 = sp_fun(0.0)
    x, ev, fun = solve_steady(plant_model, mother, water, headers, sp0)
    if ev is None:
        raise RuntimeError("no initial steady state")
    skid.set_state_from_eval(ev, x[:3])
    if hasattr(controller, "reset_near"):
        if isinstance(controller, NMPC):
            controller.reset_near(x[:3], x[3])
        else:
            controller.reset_near(x[:3])
    dt = 1.0
    n = int(t_end / dt)
    # analyzer buffers
    oe_buf, abv_buf, co2_buf, do_buf = [], [], [], []
    in_buf = []
    rows = []
    cmd = x[:3].copy()
    dose = float(x[3])
    ctrl_every = 4
    for k in range(n):
        t = k * dt
        mother = mother_fun(t)
        water = water_fun(t)
        headers = header_fun(t)
        sp = sp_fun(t)
        if name.startswith("S3"):
            skid.deadband = 0.03 if t >= event_time else 0.0
            if isinstance(controller, NMPC):
                controller.deadband = 0.03 if (controller.know_deadband and t >= event_time + 8) else 0.0
        meas_raw = {
            "qh": skid.pos[0] and None,
        }
        # step plant with current command (held between control updates)
        out = skid.step(cmd, dose, mother, water, headers, dt)
        # sensors
        qh = out["qh"] + float(RNG.normal(0, 0.02))
        qw = out["qw"] + float(RNG.normal(0, 0.015))
        qf = out["qf"] + float(RNG.normal(0, 0.02))
        pm = out["pm"] + float(RNG.normal(0, 0.006))
        oe_m = delayed(oe_buf, out["oe"] + float(RNG.normal(0, 0.008)), 12)
        abv_m = delayed(abv_buf, out["abv"] + float(RNG.normal(0, 0.008)), 12)
        co2_m = delayed(co2_buf, out["co2"] + float(RNG.normal(0, 0.015)), 12)
        do_m = delayed(do_buf, out["DO"] + float(RNG.normal(0, 1.5)), 12)
        in_m = delayed(in_buf, {"Aw": mother["Aw"], "Er": mother["Er"], "CO2": mother["CO2"], "DO": mother["DO"]}, 8)
        meas = {
            "qh": qh,
            "qw": qw,
            "qf": qf,
            "pm": pm,
            "oe_d": oe_m,
            "abv_d": abv_m,
            "co2_d": co2_m,
            "DO_d": do_m,
        }
        if k % ctrl_every == 0 and k > 0:
            if isinstance(controller, NMPC):
                # observer state: use true mixer state lagged conceptually by
                # flow integration. For the case study the flow-based observer
                # is given the plant mixer state corrupted by a small error,
                # because measured flows make it observable; we add noise.
                st = {
                    "Aw": skid.Aw + float(RNG.normal(0, 0.01)),
                    "Er": skid.Er + float(RNG.normal(0, 0.01)),
                    "CO2": skid.CO2 + float(RNG.normal(0, 0.02)),
                    "DO": skid.DO + float(RNG.normal(0, 1.0)),
                }
                mother_c = dict(mother)
                mother_c["Aw"] = in_m["Aw"]
                mother_c["Er"] = in_m["Er"]
                mother_c["CO2"] = in_m["CO2"]
                cmd, dose = controller.act(st, sp, mother_c, water, headers, dlmax, dt_ctrl=ctrl_every)
            else:
                mother_c = dict(mother)
                mother_c["Aw"] = in_m["Aw"]
                mother_c["Er"] = in_m["Er"]
                mother_c["CO2"] = in_m["CO2"]
                cmd, dose = controller.act(meas, sp, mother_c, water, ctrl_every, dlmax)
        rows.append(
            {
                "t": t,
                "oe": out["oe"],
                "abv": out["abv"],
                "co2": out["co2"],
                "DO": out["DO"],
                "pm": out["pm"],
                "p_eq": out["p_eq"],
                "qh": out["qh"],
                "qw": out["qw"],
                "qf": out["qf"],
                "vh": out["vh"],
                "vw": out["vw"],
                "vf": out["vf"],
                "lh": float(out["lifts"][0]),
                "lw": float(out["lifts"][1]),
                "lf": float(out["lifts"][2]),
                "pvc_f": out["pvc_f"],
                "oe_sp": sp["oe"],
                "abv_sp": sp["abv"],
                "co2_sp": sp["co2"],
                "qf_sp": sp["qf"],
                "dose": dose,
            }
        )
    return rows, {"x0": x.tolist(), "ev0": {k: ev[k] for k in ("oe", "abv", "co2", "DO", "pm", "p_eq", "qh", "qw", "qf", "vh", "vw", "vf", "pvc_f", "pvc_h", "pvc_w", "pd", "Re", "f", "w", "Aw", "Er", "zeta")}, "fun": fun}


def kpis(rows, t0):
    tt = np.array([r["t"] for r in rows])
    mask = tt >= t0
    def iae(key, spkey):
        e = np.array([r[key] - r[spkey] for r in rows])
        return float(np.sum(np.abs(e[mask])) * 1.0)

    oe = np.array([r["oe"] - r["oe_sp"] for r in rows])
    # settling: first time after t0 such that remaining |e|<0.05
    settle = None
    idx = np.where(tt >= t0)[0]
    for j in idx:
        if np.all(np.abs(oe[j:]) < 0.05):
            settle = float(tt[j] - t0)
            break
    if settle is None:
        settle = float(tt[-1] - t0)
    travel = 0.0
    for a, b in zip(rows[:-1], rows[1:]):
        if b["t"] < t0:
            continue
        travel += abs(b["lh"] - a["lh"]) + abs(b["lw"] - a["lw"]) + abs(b["lf"] - a["lf"])
    margin = np.array([r["pm"] - r["p_eq"] for r in rows])[mask]
    pvc_viol = np.array([r["pvc_f"] < r["p_eq"] for r in rows])[mask]
    tail = tt >= (tt[-1] - 20)
    return {
        "iae_oe": iae("oe", "oe_sp"),
        "iae_abv": iae("abv", "abv_sp"),
        "iae_co2": iae("co2", "co2_sp"),
        "iae_q": iae("qf", "qf_sp"),
        "settle_oe": settle,
        "travel": float(travel),
        "min_margin": float(np.min(margin)),
        "pvc_viol_s": float(np.sum(pvc_viol) * 1.0),
        "max_DO": float(np.max(np.array([r["DO"] for r in rows])[mask])),
        "tail_oe": float(np.mean(np.array([r["oe"] - r["oe_sp"] for r in rows])[tail])),
        "tail_abv": float(np.mean(np.array([r["abv"] - r["abv_sp"] for r in rows])[tail])),
        "tail_co2": float(np.mean(np.array([r["co2"] - r["co2_sp"] for r in rows])[tail])),
        "max_abs_oe": float(np.max(np.abs(oe[mask]))),
    }


def series(rows, key):
    return np.array([r[key] for r in rows])


def plot_quality(path, bundle, title):
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.6), sharex=True)
    keys = [("oe", "oe_sp", "Original extract (°P)"), ("abv", "abv_sp", "Alcohol (% v/v)"), ("co2", "co2_sp", "Dissolved CO$_2$ (g/L)")]
    styles = {
        "Ratio-PI": ("0.35", "--"),
        "3V-NMPC": ("0.55", "-."),
        "3V-NMPC + PIP-Twin": ("0.0", "-"),
    }
    for ax, (k, sp, ylab) in zip(axes, keys):
        for name, rows in bundle.items():
            ax.plot(series(rows, "t"), series(rows, k), color=styles[name][0], ls=styles[name][1], lw=1.4, label=name)
        ax.plot(series(bundle["Ratio-PI"], "t"), series(bundle["Ratio-PI"], sp), color="0.15", lw=0.8, label="Setpoint")
        ax.set_ylabel(ylab)
        ax.grid(True, lw=0.3, color="0.85")
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_valves(path, bundle):
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.2), sharex=True)
    keys = [("lh", "Beer-valve lift"), ("lw", "Water-valve lift"), ("lf", "Final-valve lift")]
    styles = {
        "Ratio-PI": ("0.35", "--"),
        "3V-NMPC": ("0.55", "-."),
        "3V-NMPC + PIP-Twin": ("0.0", "-"),
    }
    for ax, (k, ylab) in zip(axes, keys):
        for name, rows in bundle.items():
            ax.plot(series(rows, "t"), series(rows, k), color=styles[name][0], ls=styles[name][1], lw=1.4, label=name)
        ax.set_ylabel(ylab)
        ax.set_ylim(0.15, 0.95)
        ax.grid(True, lw=0.3, color="0.85")
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bars(path, table):
    # table[scenario][controller] = iae_oe
    scenarios = list(table.keys())
    controllers = list(next(iter(table.values())).keys())
    x = np.arange(len(scenarios))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    colors = ["0.75", "0.45", "0.15"]
    for i, cname in enumerate(controllers):
        vals = [table[s][cname] for s in scenarios]
        ax.bar(x + (i - 1) * width, vals, width, label=cname, color=colors[i], edgecolor="black", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("IAE of original extract (°P·s)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", lw=0.3, color="0.85")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_parity(path, q_true, q_nom, q_hat):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(q_true[:, 0], q_nom[:, 0], "o", ms=4, color="0.55", label="Beer flow, nominal model")
    ax.plot(q_true[:, 0], q_hat[:, 0], "s", ms=4, color="0.0", label="Beer flow, PIP-Twin")
    lims = [q_true[:, 0].min() - 0.5, q_true[:, 0].max() + 0.5]
    ax.plot(lims, lims, color="0.3", lw=0.8)
    ax.set_xlabel("Measured beer flow (m$^3$/h)")
    ax.set_ylabel("Predicted beer flow (m$^3$/h)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, lw=0.3, color="0.85")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pressure(path, ev, headers):
    # schematic stations
    labels = ["Beer\nheader", "Beer-valve\nvena contracta", "Mix\nnode", "After\nline loss", "Final-valve\nvena contracta", "Bright-beer\ntank"]
    vals = [headers[0], ev["pvc_h"], ev["pm"], ev["pd"], ev["pvc_f"], headers[2]]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(range(6), vals, "-o", color="black", lw=1.3)
    ax.axhline(ev["p_eq"], color="0.4", ls="--", label="CO$_2$ equilibrium pressure")
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Absolute pressure (bar)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, lw=0.3, color="0.85")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_char(path, model, lifts, mother, water, headers, comp):
    ls = np.linspace(0.25, 0.88, 25)
    q = []
    inher = []
    q_ref = None
    for l in ls:
        lp = [l, lifts[1], lifts[2]]
        model.set_inlets(mother["Aw"], mother["Er"])
        hyd = model.hydraulics(lp, *headers, comp["Aw"], comp["Er"], comp["CO2"])
        q.append(hyd["qh"])
    # inherent equal-percentage scaled to the flow at lift 0.85
    for l in ls:
        inher.append(phi_eq(l))
    inher = np.array(inher)
    inher = inher / phi_eq(0.85) * np.interp(0.85, ls, q)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ls, inher, color="0.45", ls="--", label="Inherent equal-percentage, scaled")
    ax.plot(ls, q, color="black", label="Installed beer flow")
    ax.set_xlabel("Beer-valve lift (fraction)")
    ax.set_ylabel("Beer flow (m$^3$/h)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, lw=0.3, color="0.85")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fmt(x, n=3):
    return f"{x:.{n}f}"


def write_tables(steady, G, RGA, est, kpi_table, design):
    lines = []
    lines.append("% auto-generated case-study numbers\n")
    ev = steady["ev"]
    lines.append("\\begin{table}[htbp]\\centering")
    lines.append("\\caption{Steady operating point of the sales-gravity lager on the plant model.}")
    lines.append("\\label{tab:steady}")
    lines.append("\\begin{tabular}{lr}\\toprule")
    lines.append("Quantity & Value \\\\\\midrule")
    rows = [
        ("Beer-valve lift", f"{steady['x'][0]:.3f}"),
        ("Water-valve lift", f"{steady['x'][1]:.3f}"),
        ("Final-valve lift", f"{steady['x'][2]:.3f}"),
        ("Carbon-dioxide dose (g/s)", f"{steady['x'][3]:.3f}"),
        ("Original extract ($^\\circ$P)", f"{ev['oe']:.3f}"),
        ("Alcohol (\\% v/v)", f"{ev['abv']:.3f}"),
        ("Dissolved CO$_2$ (g/L)", f"{ev['co2']:.3f}"),
        ("Dissolved oxygen (ppb)", f"{ev['DO']:.2f}"),
        ("Beer flow (m$^3$/h)", f"{ev['qh']:.3f}"),
        ("Water flow (m$^3$/h)", f"{ev['qw']:.3f}"),
        ("Product flow (m$^3$/h)", f"{ev['qf']:.3f}"),
        ("Beer velocity (m/s)", f"{ev['vh']:.3f}"),
        ("Water velocity (m/s)", f"{ev['vw']:.3f}"),
        ("Product velocity (m/s)", f"{ev['vf']:.3f}"),
        ("Mix pressure (bar a)", f"{ev['pm']:.3f}"),
        ("CO$_2$ equilibrium pressure (bar a)", f"{ev['p_eq']:.3f}"),
        ("Pressure margin (bar)", f"{ev['pm']-ev['p_eq']:.3f}"),
        ("Final-valve vena contracta (bar a)", f"{ev['pvc_f']:.3f}"),
        ("Blend-line Reynolds number", f"{ev['Re']:.0f}"),
        ("Darcy friction factor", f"{ev['f']:.4f}"),
        ("Mother-beer mass fraction", f"{ev['w']:.4f}"),
    ]
    for a, b in rows:
        lines.append(f"{a} & {b} \\\\")
    lines.append("\\bottomrule\\end{tabular}\\end{table}\n")

    lines.append("\\begin{table}[htbp]\\centering")
    lines.append("\\caption{Steady gain matrix and relative-gain array at the lager point. Outputs are beer flow (m$^3$/h), water flow (m$^3$/h) and mix pressure (bar). Inputs are the three valve lifts.}")
    lines.append("\\label{tab:rga}")
    lines.append("\\begin{tabular}{lccc}\\toprule")
    lines.append(" & Beer lift & Water lift & Final lift \\\\\\midrule")
    names = ["Beer flow", "Water flow", "Mix pressure"]
    for i, nm in enumerate(names):
        lines.append(f"{nm} & {G[i,0]:.2f} & {G[i,1]:.2f} & {G[i,2]:.2f} \\\\")
    lines.append("\\midrule")
    for i, nm in enumerate(["$\\Lambda$ beer flow", "$\\Lambda$ water flow", "$\\Lambda$ mix pressure"]):
        lines.append(f"{nm} & {RGA[i,0]:.2f} & {RGA[i,1]:.2f} & {RGA[i,2]:.2f} \\\\")
    lines.append("\\bottomrule\\end{tabular}\\end{table}\n")

    lines.append("\\begin{table}[htbp]\\centering")
    lines.append("\\caption{Commissioning estimates. Standard errors are from the linearised least-squares covariance.}")
    lines.append("\\label{tab:est}")
    lines.append("\\begin{tabular}{lrrr}\\toprule")
    lines.append("Parameter & True & Estimate & Std. error \\\\\\midrule")
    for i, lab in enumerate(["$K_{v,h}$ (m$^3$/h)", "$K_{v,w}$ (m$^3$/h)", "$K_{v,f}$ (m$^3$/h)"]):
        lines.append(f"{lab} & {est['Kv_true'][i]:.2f} & {est['Kv'][i]:.2f} & {est['se_Kv'][i]:.2f} \\\\")
    lines.append(f"Roughness $\\varepsilon$ ($\\mu$m) & {est['eps_true']*1e6:.2f} & {est['eps']*1e6:.2f} & {est['se_eps']*1e6:.2f} \\\\")
    lines.append(f"Minor-loss $K$ & {est['K_true']:.2f} & {est['K']:.2f} & {est['se_K']:.2f} \\\\")
    lines.append(f"Header bias (bar) & {est['bias_true']:.3f} & {est['bias']:.3f} & {est['se_bias']:.3f} \\\\")
    lines.append(f"Solubility scale $\\kappa_\\zeta$ & {est['zeta_scale_true']:.3f} & {est['zeta_scale']:.3f} & -- \\\\")
    lines.append("\\bottomrule\\end{tabular}\\end{table}\n")

    lines.append("\\begin{table}[htbp]\\centering")
    lines.append("\\caption{Closed-loop estimation of beer quality after each upset. IAE is integrated from the upset to the end of the run. Settling time is the time after the upset until original-extract error remains inside $0.05\\,^\\circ$P.}")
    lines.append("\\label{tab:kpi}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llrrrrrr}\\toprule")
    lines.append("Case & Controller & IAE OE & IAE ABV & IAE CO$_2$ & Settle (s) & Travel & Min. margin \\\\\\midrule")
    for case, ctrls in kpi_table.items():
        for cname, k in ctrls.items():
            lines.append(
                f"{case} & {cname} & {k['iae_oe']:.2f} & {k['iae_abv']:.2f} & {k['iae_co2']:.2f} & {k['settle_oe']:.0f} & {k['travel']:.3f} & {k['min_margin']:.2f} \\\\"
            )
    lines.append("\\bottomrule\\end{tabular}}\\end{table}\n")
    (GEN / "tables.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    t_wall = time.time()
    plant = make_plant()
    nominal = make_nominal()
    headers0 = (4.80, 5.30, 2.40)  # bar absolute: beer, water, bright tank

    # Achievable setpoints on the PLANT density (with contraction)
    lag = steady_targets(plant, 0.70, 5.20)
    light = steady_targets(plant, 0.58, 5.00)
    sp_lag = {"oe": lag["oe"], "abv": lag["abv"], "co2": 5.20, "qf": 20.0}
    sp_light = {"oe": light["oe"], "abv": light["abv"], "co2": 5.00, "qf": 20.0}
    print("LAGER SP", sp_lag)
    print("LIGHT SP", sp_light)
    print("MOTHER OE", float(balling_oe(MOTHER0["Aw"], MOTHER0["Er"])))

    x, ev, fun = solve_steady(plant, MOTHER0, WATER0, headers0, sp_lag)
    print("STEADY x", x, "fun", fun)
    print("STEADY ev", {k: ev[k] for k in ("oe", "abv", "co2", "DO", "pm", "p_eq", "qh", "qw", "qf", "vh", "vw", "vf", "w", "Re", "f", "pvc_f")})
    comp = {"Aw": ev["Aw"], "Er": ev["Er"], "CO2": ev["co2_tank"] if "co2_tank" in ev else ev["co2"]}
    # evaluate() doesn't return co2_tank; use unstripped approximation from Aw state
    comp = {"Aw": ev["Aw"], "Er": ev["Er"], "CO2": 5.2}
    plant.set_inlets(MOTHER0["Aw"], MOTHER0["Er"])
    G, RGA = gain_matrix(plant, x[:3], x[3], MOTHER0, WATER0, headers0, comp)
    print("G\n", G)
    print("RGA\n", RGA)
    plot_pressure(FIG / "pressure_profile.pdf", ev, headers0)
    plot_char(FIG / "installed_characteristic.pdf", plant, x[:3], MOTHER0, WATER0, headers0, comp)

    print("Commissioning...")
    est, parity, model_hat, net, norm = commissioning(plant, headers0, MOTHER0, WATER0)
    print(json.dumps(est, indent=2))
    plot_parity(FIG / "parity_flow.pdf", *parity)

    # Build calibrated twin: estimated hydraulics + plant-like FL (known from valve data sheets)
    twin = Model(
        Kv=tuple(est["Kv"]),
        eps=est["eps"],
        K_minor=est["K"],
        zeta_scale=est["zeta_scale"],
        contraction=est["contraction"],
        FL=(0.90, 0.93, 0.86),
        p_bias=0.0,
        name="twin",
    )
    # The twin uses corrected header = measurement - bias. In closed loop we pass the true
    # header because the bias has been removed by estimation. Nominal still carries p_bias.

    p_sp_pi = float(ev["p_eq"] + 0.90)
    print("PI pressure SP", p_sp_pi)

    def scenarios():
        # S1 grade change
        def s1_mother(t):
            return dict(MOTHER0)

        def s1_water(t):
            return dict(WATER0)

        def s1_header(t):
            return headers0

        def s1_sp(t):
            return sp_light if t >= 20 else sp_lag

        # S2 header drop + mother drift
        def s2_mother(t):
            m = dict(MOTHER0)
            if t >= 20:
                m["Er"] = MOTHER0["Er"] + 0.45
                m["Aw"] = MOTHER0["Aw"] - 0.25
                m["CO2"] = 5.15
            return m

        def s2_header(t):
            if t >= 20:
                return (headers0[0] - 0.45, headers0[1], headers0[2])
            return headers0

        def s2_sp(t):
            return sp_lag

        # S3 DO spike + stiction
        def s3_water(t):
            w = dict(WATER0)
            if t >= 20:
                w["DO"] = 85.0
            return w

        return [
            ("S1 grade change", s1_mother, s1_water, s1_header, s1_sp, 160, 20),
            ("S2 supply upset", s2_mother, WATER0_fun, s2_header, s2_sp, 170, 20),
            ("S3 oxygen and stiction", s1_mother, s3_water, s1_header, s2_sp, 150, 20),
        ]

    def WATER0_fun(t):
        return dict(WATER0)

    bundles = {}
    kpi_table = {}
    steady_pack = {"x": x.tolist(), "ev": {k: (float(ev[k]) if not isinstance(ev[k], str) else ev[k]) for k in ev if np.ndim(ev[k]) == 0}}

    for label, mf, wf, hf, sf, tend, tev in [
        ("S1 grade change", lambda t: dict(MOTHER0), lambda t: dict(WATER0), lambda t: headers0, lambda t: sp_light if t >= 20 else sp_lag, 150, 20),
        ("S2 supply upset", lambda t: (dict(MOTHER0, Er=MOTHER0["Er"] + 0.45, Aw=MOTHER0["Aw"] - 0.25, CO2=5.15) if t >= 20 else dict(MOTHER0)), lambda t: dict(WATER0), lambda t: ((headers0[0] - 0.45, headers0[1], headers0[2]) if t >= 20 else headers0), lambda t: sp_lag, 160, 20),
        ("S3 oxygen and stiction", lambda t: dict(MOTHER0), lambda t: (dict(WATER0, DO=85.0) if t >= 20 else dict(WATER0)), lambda t: headers0, lambda t: sp_lag, 140, 20),
    ]:
        print("===", label)
        bundle = {}
        kpi_table[label] = {}
        scenario_seed = 400 + len(kpi_table)
        controllers = [
            ("Ratio-PI", PIRatio(p_sp_pi), False),
            ("3V-NMPC", NMPC(nominal, know_deadband=False), False),
            ("3V-NMPC + PIP-Twin", NMPC(twin, know_deadband=True), True),
        ]
        for cname, ctrl, _ in controllers:
            print("  ", cname, "...")
            global RNG
            RNG = np.random.default_rng(scenario_seed)
            t1 = time.time()
            # fresh controller instance each time (the list reuses). rebuild
            if cname == "Ratio-PI":
                ctrl = PIRatio(p_sp_pi)
            elif cname == "3V-NMPC":
                ctrl = NMPC(nominal, know_deadband=False)
            else:
                ctrl = NMPC(twin, know_deadband=True)
            rows, info = run_scenario(label, cname, ctrl, plant, mf, wf, hf, sf, tend, tev)
            kp = kpis(rows, tev)
            print("   ", {k: round(v, 3) if isinstance(v, float) else v for k, v in kp.items()}, "in", round(time.time() - t1, 1), "s")
            bundle[cname] = rows
            kpi_table[label][cname] = kp
            if label.startswith("S1") and cname == "Ratio-PI":
                steady_info = info
        bundles[label] = bundle

    plot_quality(FIG / "s1_quality.pdf", bundles["S1 grade change"], "S1")
    plot_valves(FIG / "s1_valves.pdf", bundles["S1 grade change"])
    plot_quality(FIG / "s2_quality.pdf", bundles["S2 supply upset"], "S2")
    plot_quality(FIG / "s3_quality.pdf", bundles["S3 oxygen and stiction"], "S3")
    plot_bars(
        FIG / "iae_bars.pdf",
        {s: {c: kpi_table[s][c]["iae_oe"] for c in kpi_table[s]} for s in kpi_table},
    )

    design = {"sp_lag": sp_lag, "sp_light": sp_light, "headers": headers0, "p_sp_pi": p_sp_pi, "mother_oe": float(balling_oe(MOTHER0["Aw"], MOTHER0["Er"]))}
    write_tables({"x": x, "ev": ev}, G, RGA, est, kpi_table, design)

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        return o

    payload = {
        "design": conv(design),
        "steady_x": [float(v) for v in x],
        "steady": {k: float(ev[k]) for k in ev if np.ndim(ev[k]) == 0},
        "G": G.tolist(),
        "RGA": RGA.tolist(),
        "estimation": conv(est),
        "kpi": conv(kpi_table),
        "wall_s": time.time() - t_wall,
    }
    (GEN / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WALL", round(time.time() - t_wall, 1))


if __name__ == "__main__":
    main()

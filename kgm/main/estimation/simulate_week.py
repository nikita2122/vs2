"""
Reduced-order week simulation for the brewery wastewater fan-automation study.

The plant is an aerobic polishing tank downstream of anaerobic pretreatment.
Numbers printed here are the source of the tables in Chapter 3.
This is a simulation of a representative train, not a record from a named factory.
"""

from __future__ import annotations

import math

# --- plant constants (stated in the thesis) ---
V = 900.0  # m3 aerobic polishing volume
X = 3800.0  # mg/L MLVSS, held by sludge-age control
K_S = 40.0  # mg/L
K_DO = 0.35  # mg/L
K_N = 1.0
# r = Q_COD * X * monod. r_max about 55 mg COD/L/h, oxygen-feasible at CIP alpha.
Q_COD = 0.0145
Q_N = 0.00040
Y_O2_COD = 0.95  # kg O2 / kg COD removed, carbonaceous + endogenous lump
Y_O2_N = 4.57
FI = 0.08  # inert fraction of aerobic influent COD
C20 = 9.09
BETA = 0.95
THETA = 1.024
SOTE = 0.30
RHO_O2 = 0.299  # kg O2 per Nm3 air
Q_MIN = 900.0  # Nm3/h mixing floor, also above the one-fan surge proxy
Q_MAX = 9000.0  # three-blower header capacity
DT = 0.02  # h
PERMIT_COD = 125.0  # mg/L, study discharge target

CSTAR = {26: 8.11, 28: 7.83, 30: 7.54, 31: 7.43, 33: 7.16}


def cstar(T: float) -> float:
    # linear interpolation on the tabulated fresh-water saturation
    keys = sorted(CSTAR)
    if T <= keys[0]:
        return CSTAR[keys[0]]
    if T >= keys[-1]:
        return CSTAR[keys[-1]]
    for a, b in zip(keys, keys[1:]):
        if a <= T <= b:
            w = (T - a) / (b - a)
            return CSTAR[a] * (1 - w) + CSTAR[b] * w
    return CSTAR[keys[-1]]


def power_kw(q_nm3_h: float) -> float:
    """Polytropic blower power against hydrostatic head plus pipe friction."""
    q_nm3_h = max(q_nm3_h, 0.0)
    q_s = q_nm3_h / 3600.0
    m_dot = 1.293 * q_s
    dp_kpa = 52.0 + 0.55 * (q_nm3_h / 1000.0) ** 2
    p_in = 101.3
    ratio = (p_in + dp_kpa) / p_in
    expo = (1.4 - 1.0) / 1.4
    factor = (1.4 / 0.4) * (ratio**expo - 1.0)
    p_ad = m_dot * 287.0 * 293.15 * factor  # W
    eta = 0.70 * 0.95 * 0.97
    return p_ad / eta / 1000.0


def kla_per_h(q_nm3_h: float, alpha: float, T: float) -> float:
    """
    Field KLa (1/h). Clean-water KLa follows airflow to the power 0.82,
    then alpha, temperature, and the saturation correction are applied
    outside this function via the driving force. Here KLa multiplies
    (beta*C*_T - DO) * theta correction already folded into kla.
    """
    q_ref = 2500.0
    kla_ref = 6.5  # 1/h at q_ref, alpha=1, T=20, before field factors
    kla20 = kla_ref * (max(q_nm3_h, 1.0) / q_ref) ** 0.82
    return alpha * kla20 * (THETA ** (T - 20.0))


# Regimes: name, hours, Q m3/h, COD mg/L, TKN mg/L, alpha, T celsius
REGIMES = [
    ("base", 96.0, 50.0, 380.0, 26.0, 0.60, 28.0),
    ("brewhouse", 30.0, 70.0, 460.0, 34.0, 0.52, 31.0),
    ("ferm_cip", 18.0, 75.0, 430.0, 28.0, 0.40, 33.0),
    ("pack_cip", 12.0, 68.0, 410.0, 26.0, 0.46, 30.0),
    ("weekend", 12.0, 22.0, 160.0, 12.0, 0.66, 26.0),
]


def schedule(t_h: float):
    x = t_h % 168.0
    acc = 0.0
    for rec in REGIMES:
        acc += rec[1]
        if x < acc:
            return rec
    return REGIMES[-1]


def do_setpoint(name: str) -> float:
    return {
        "base": 1.8,
        "brewhouse": 2.0,
        "ferm_cip": 2.2,
        "pack_cip": 2.0,
        "weekend": 1.3,
    }[name]


def airflow_for_target(do_target, our_mg_l_h, alpha, T):
    """Invert KLa*(beta*C* - DO) = OUR, then invert KLa(Q)."""
    drive = BETA * cstar(T) - do_target
    if drive < 0.4:
        drive = 0.4
    kla = our_mg_l_h / drive
    # invert kla = alpha * kla_ref * (Q/qref)^0.82 * theta^(T-20)
    kla_ref = 6.5
    q_ref = 2500.0
    scale = alpha * (THETA ** (T - 20.0)) * kla_ref
    if scale <= 1e-6:
        return Q_MAX
    ratio = max(kla / scale, 1e-4)
    q = q_ref * (ratio ** (1.0 / 0.82))
    return min(Q_MAX, max(Q_MIN, q))


def simulate(mode: str):
    """
    mode:
      proposed  - true alpha (soft sensor, 8% noise-free bias 0, 0.25 h lag)
                  and schedule preview of influent
      pid       - PI on DO, setpoint 2.5, no schedule, no alpha
      mpc_fixed - certainty-equivalent inversion, alpha fixed at 0.55,
                  influent persistence, DO setpoint 2.0, slow bias
      fixed     - constant airflow sized for fermentation-CIP design point
    """
    sb = 80.0  # biodegradable COD mg/L
    sn = 8.0  # ammonia mg/L
    do = 2.0
    q_air = 1500.0
    integ = 0.0
    bias = 0.0
    alpha_hat = 0.55
    # fixed airflow from ferm CIP design using true alpha and DO 2.2
    q_fixed = airflow_for_target(
        2.2,
        # rough OUR at that regime, refined after a dry run; set after computing design OUR
        140.0,
        0.34,
        33.0,
    )
    # precompute fixed from stoichiometry of regime
    _, _, q_in, cod, tkn, alpha, T = REGIMES[2]
    sb_in = (1.0 - FI) * cod
    # use inversion at expected OUR = Y * (qin/V) * sb_in * removal guess
    our_guess = (Y_O2_COD * 0.90 * (q_in / V) * sb_in) + (
        Y_O2_N * 0.70 * (q_in / V) * tkn
    )
    q_fixed = airflow_for_target(2.2, our_guess, alpha, T)

    t = 0.0
    t_end = 168.0
    energy = 0.0
    air_m3 = 0.0
    viol_h = 0.0
    do_sum = 0.0
    do_min = 99.0
    cod_out_sum = 0.0
    n_steps = 0
    # per-regime accumulators
    names = [r[0] for r in REGIMES]
    e_reg = {n: 0.0 for n in names}
    h_reg = {n: 0.0 for n in names}
    viol_reg = {n: 0.0 for n in names}
    cod_sum_reg = {n: 0.0 for n in names}
    do_sum_reg = {n: 0.0 for n in names}
    surge_flags = 0
    sat_h = 0.0

    while t < t_end - 1e-9:
        name, _, q_in, cod, tkn, alpha, temp = schedule(t)
        sb_in = (1.0 - FI) * cod
        sn_in = tkn

        monod_s = sb / (K_S + sb)
        monod_do = do / (K_DO + do)
        monod_n = sn / (K_N + sn)
        r_cod = Q_COD * X * monod_s * monod_do  # mg/L/h
        r_n = Q_N * X * monod_n * monod_do
        our = Y_O2_COD * r_cod + Y_O2_N * r_n  # mg/L/h

        # controller
        if mode == "fixed":
            q_cmd = q_fixed
        elif mode == "pid":
            sp = 2.5
            err = sp - do
            # gains on airflow Nm3/h per mg/L, tuned for ~1 h response
            kp = 900.0
            ki = 280.0
            integ += err * DT
            integ = max(-8.0, min(8.0, integ))
            q_cmd = 1400.0 + kp * err + ki * integ
            q_cmd = min(Q_MAX, max(Q_MIN, q_cmd))
        elif mode == "mpc_fixed":
            # persistence forecast and design alpha
            alpha_hat = 0.55
            our_hat = our  # measured-load certainty equivalent (OUR from last step)
            # but KLa curve uses wrong alpha, so commanded air is short when alpha is low;
            # a slow integral bias trims DO error, as offset-free MPC would.
            q_ff = airflow_for_target(2.0, max(our_hat, 1.0), alpha_hat, temp)
            err = 2.0 - do
            bias += 0.15 * err * DT
            bias = max(-0.35, min(0.55, bias))
            q_cmd = min(Q_MAX, max(Q_MIN, q_ff * (1.0 + bias)))
        elif mode == "proposed":
            # soft sensor: first-order lag 0.25 h toward true alpha, then 4% scale bias
            alpha_hat += (DT / 0.25) * (alpha - alpha_hat)
            alpha_used = max(0.2, alpha_hat * 0.96)
            # schedule preview: upcoming regime at t+0.5 h (known calendar)
            name_p, _, q_p, cod_p, tkn_p, alpha_p, temp_p = schedule(t + 0.5)
            # predicted OUR from previewed load at the scheduled DO, capped by kinetics
            sb_p = (1.0 - FI) * cod_p
            monod_s_p = sb_p / (K_S + sb_p)
            sp = do_setpoint(name_p)
            monod_do_p = sp / (K_DO + sp)
            r_cod_p = Q_COD * X * monod_s_p * monod_do_p
            r_n_p = Q_N * X * (tkn_p / (K_N + tkn_p)) * monod_do_p
            our_p = Y_O2_COD * r_cod_p + Y_O2_N * r_n_p
            # blend preview OUR with measured OUR (feedback)
            our_blend = 0.65 * our_p + 0.35 * our
            q_ff = airflow_for_target(sp, max(our_blend, 1.0), alpha_used, temp_p)
            err = sp - do
            bias += 0.35 * err * DT
            bias = max(-0.2, min(0.25, bias))
            q_cmd = min(Q_MAX, max(Q_MIN, q_ff * (1.03) * (1.0 + bias)))
        else:
            raise ValueError(mode)

        # surge proximity flag: each online fan should stay above ~18% of header max / n
        # report fraction of time within 10% of per-fan surge proxy 0.22*Q_fan_rated
        q_air = q_cmd
        kla = kla_per_h(q_air, alpha, temp)
        drive = BETA * cstar(temp) - do
        otr = kla * drive  # mg/L/h

        # Euler plant
        dsb = (q_in / V) * (sb_in - sb) - r_cod
        dsn = (q_in / V) * (sn_in - sn) - r_n
        ddo = otr - our
        sb = max(0.5, sb + DT * dsb)
        sn = max(0.05, sn + DT * dsn)
        do = max(0.02, min(8.0, do + DT * ddo))

        cod_out = sb + FI * cod  # inert passes through; approximation uses current influent inert
        # more honest inert: filtered inert state would lag. Use tank inert as FI fraction of
        # a filtered influent. Keep simple: inert is not removed, CSTR on inert.
        # We fold inert into cod_out below with a separate state after the loop init.
        p = power_kw(q_air)
        energy += p * DT
        air_m3 += q_air * DT
        e_reg[name] += p * DT
        h_reg[name] += DT
        do_sum += do
        do_sum_reg[name] += do
        do_min = min(do_min, do)
        n_steps += 1
        if q_air >= 0.98 * Q_MAX:
            sat_h += DT
        # per-fan surge: stage fans at 2800 Nm3/h each (rated), surge at 22% rated
        n_fans = 1 if q_air < 2600 else (2 if q_air < 5200 else 3)
        q_each = q_air / n_fans
        if q_each < 1.10 * 0.22 * 2800:
            surge_flags += 1

        t += DT

    # second pass not needed; inert handling: recompute properly? 
    # The loop used cod_out incorrectly. Rerun inert as part of sb definition:
    # sb is biodegradable only, effluent COD = sb + inert.
    # I need an inert state. Let me do it inline by reconstructing... 
    # I'll run a corrected loop instead. See simulate2.
    return {
        "note": "use simulate_corrected",
    }


def simulate_corrected(mode: str):
    V_local = V
    sb = 40.0
    si = 30.0  # inert COD
    sn = 4.0
    do = 2.0
    do_meas = 2.0
    q_applied = 1500.0
    integ = 0.0
    bias = 0.0
    alpha_hat = 0.55
    tau_fan = 0.10  # h, blower and header lag
    tau_do = 0.08  # h, probe lag

    # fixed-speed design point: fermentation-CIP oxygen demand at the
    # feasible equilibrium, plus 15 percent, not the saturated kinetic maximum
    _, _, _, _, _, alpha_f, temp_f = REGIMES[2]
    q_fixed = airflow_for_target(2.2, 32.0, alpha_f, temp_f) * 1.15
    q_fixed = min(Q_MAX, q_fixed)

    t = 0.0
    t_end = 168.0
    energy = 0.0
    names = [r[0] for r in REGIMES]
    e_reg = {n: 0.0 for n in names}
    h_reg = {n: 0.0 for n in names}
    viol_reg = {n: 0.0 for n in names}
    cod_sum = {n: 0.0 for n in names}
    do_sum = {n: 0.0 for n in names}
    do_min = {n: 99.0 for n in names}
    sat_h = 0.0
    surge_h = 0.0
    cod_hold = []
    # sample every 0.5 h for a short series during ferm CIP
    series = []

    while t < t_end - 1e-9:
        name, _, q_in, cod, tkn, alpha, temp = schedule(t)
        sb_in = (1.0 - FI) * cod
        si_in = FI * cod

        monod_s = sb / (K_S + sb)
        monod_do = do / (K_DO + do)
        monod_n = sn / (K_N + sn)
        r_cod = Q_COD * X * monod_s * monod_do
        r_n = Q_N * X * monod_n * monod_do * monod_s
        our = Y_O2_COD * r_cod + Y_O2_N * r_n

        if mode == "fixed":
            q_cmd = q_fixed
        elif mode == "pid":
            sp = 2.5
            err = sp - do_meas
            kp, ki = 1400.0, 500.0
            integ += err * DT
            integ = max(-8.0, min(8.0, integ))
            q_cmd = 1000.0 + kp * err + ki * integ
        elif mode == "mpc_fixed":
            alpha_hat_use = 0.55
            q_ff = airflow_for_target(2.0, max(our, 1.0), alpha_hat_use, temp)
            err = 2.0 - do_meas
            bias += 0.18 * err * DT
            bias = max(-0.25, min(0.85, bias))
            q_cmd = q_ff * (1.0 + bias)
        elif mode == "proposed":
            alpha_hat += (DT / 0.30) * (alpha - alpha_hat)
            name_p, _, q_p, cod_p, tkn_p, alpha_p, temp_p = schedule(t + 0.75)
            # scheduled alpha is known from the CIP recipe; the lagged estimate
            # covers unscheduled drift. Use the lower of the two, which is the
            # more oxygen-demanding case, but do not invent a lower value.
            alpha_used = max(0.2, min(alpha_hat, alpha_p) * 0.98)
            sp = do_setpoint(name)
            sp_p = do_setpoint(name_p)
            sp_use = max(sp, sp_p)
            load_now = max(q_in * cod, 1.0)
            load_p = q_p * cod_p
            scale = max(1.0, min(1.35, load_p / load_now))
            q_ff = airflow_for_target(sp_use, max(our, 1.0) * scale, alpha_used, temp)
            err = sp_use - do_meas
            bias += 0.55 * err * DT
            bias = max(-0.2, min(0.25, bias))
            q_cmd = q_ff * (1.0 + bias)
        else:
            raise ValueError(mode)

        q_cmd = min(Q_MAX, max(Q_MIN, q_cmd))
        q_applied += (DT / tau_fan) * (q_cmd - q_applied)
        q_applied = min(Q_MAX, max(Q_MIN, q_applied))
        kla = kla_per_h(q_applied, alpha, temp)
        cs = BETA * cstar(temp)
        # semi-implicit DO step: dDO/dt = kla*(cs-DO) - OUR
        do = (do + DT * (kla * cs - our)) / (1.0 + DT * kla)
        do = max(0.02, min(cs - 0.02, do))
        do_meas += (DT / tau_do) * (do - do_meas)
        dsb = (q_in / V_local) * (sb_in - sb) - r_cod
        dsi = (q_in / V_local) * (si_in - si)
        dsn = (q_in / V_local) * (tkn - sn) - r_n
        sb = max(0.2, sb + DT * dsb)
        si = max(0.0, si + DT * dsi)
        sn = max(0.02, sn + DT * dsn)

        cod_out = sb + si
        p = power_kw(q_applied)
        q_cmd = q_applied  # reports and surge checks use delivered air
        energy += p * DT
        e_reg[name] += p * DT
        h_reg[name] += DT
        do_sum[name] += do * DT
        cod_sum[name] += cod_out * DT
        do_min[name] = min(do_min[name], do)
        if cod_out > PERMIT_COD:
            viol_reg[name] += DT
        if q_cmd >= 0.98 * Q_MAX:
            sat_h += DT
        n_fans = 1 if q_cmd < 2600 else (2 if q_cmd < 5200 else 3)
        q_each = q_cmd / n_fans
        q_surge = 0.22 * 2800.0
        if q_each < 1.15 * q_surge:
            surge_h += DT
        # record fermentation CIP window (starts at 126 h)
        if 124.0 <= t <= 148.0 and abs((t / 1.0) - round(t / 1.0)) < DT * 0.51:
            series.append((round(t, 2), round(do, 3), round(cod_out, 1), round(q_cmd, 1), round(p, 2)))
        t += DT

    total_viol = sum(viol_reg.values())
    print(f"\n=== {mode} ===")
    print(f"week energy kWh = {energy:.1f}   avg kW = {energy/168:.2f}   viol h = {total_viol:.2f}   sat h = {sat_h:.2f}   surge-margin h = {surge_h:.2f}")
    print(f"q_fixed design = {q_fixed:.0f} Nm3/h")
    for n in names:
        h = h_reg[n]
        print(
            f"  {n:12} E={e_reg[n]:8.1f} kWh  meanDO={do_sum[n]/h:5.2f}  minDO={do_min[n]:5.2f}  "
            f"meanCOD={cod_sum[n]/h:7.1f}  viol_h={viol_reg[n]:6.2f}  meankW={e_reg[n]/h:6.2f}"
        )
    return {
        "energy": energy,
        "viol": total_viol,
        "sat": sat_h,
        "surge": surge_h,
        "reg": e_reg,
        "series": series,
        "q_fixed": q_fixed,
    }


if __name__ == "__main__":
    results = {}
    for m in ("proposed", "pid", "mpc_fixed", "fixed"):
        results[m] = simulate_corrected(m)
    base = results["pid"]["energy"]
    print("\nSavings vs PID %: ", {m: 100 * (base - results[m]["energy"]) / base for m in results})
    print("Savings vs fixed %:", {m: 100 * (results["fixed"]["energy"] - results[m]["energy"]) / results["fixed"]["energy"] for m in results})

"""Illustrative figures for the expanded fault-diagnosis paper.

Waveforms are synthesized so that RMS velocity matches the vibration
severities published by You et al. (2019) and so that each state
emphasizes the harmonics those authors describe. They are not the
laboratory recordings.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.family": "DejaVu Serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FS = 10000
N = 8192
T = np.arange(N) / FS
F0 = 16.0

# Orthonormal db4 analysis filters (PyWavelets dec_lo / dec_hi).
DEC_LO = np.array([
    0.23037781330885523, 0.7148465705525415, 0.6308807679295904,
    -0.02798376941698385, -0.18703481171888114, 0.030841381835986965,
    0.032883011666982945, -0.010597401784997278,
])
DEC_HI = np.array([
    -0.010597401784997278, -0.032883011666982945, 0.030841381835986965,
    0.18703481171888114, -0.02798376941698385, -0.6308807679295904,
    0.7148465705525415, -0.23037781330885523,
])


def tone(freq, phase=0.0):
    return np.sin(2 * np.pi * freq * T + phase)


def impulses(rate, amp, width_s, rng, carrier=1800.0):
    y = np.zeros(N)
    period = 1.0 / rate
    tau = 0.15 * period
    nwin = int(width_s * FS)
    n = np.arange(nwin)
    env = np.exp(-n / (0.22 * nwin))
    burst = env * np.sin(2 * np.pi * carrier * n / FS)
    while tau < T[-1]:
        tau_j = tau + rng.uniform(-0.0003, 0.0003)
        idx = int(tau_j * FS)
        if 0 <= idx < N:
            end = min(N, idx + nwin)
            y[idx:end] += amp * burst[: end - idx]
        tau += period
    return y


def make_signal(kind, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, N)
    b, a = signal.butter(4, 900 / (FS / 2), btype="low")
    noise = signal.filtfilt(b, a, noise)
    if kind == "normal":
        x = (
            0.70 * tone(F0, 0.2)
            + 0.22 * tone(2 * F0, 0.4)
            + 0.14 * tone(3 * F0, 1.1)
            + 0.10 * tone(4 * F0, 0.7)
            + 0.06 * tone(50, 0.3)
            + 0.35 * (1 + 0.25 * tone(F0)) * tone(220, 0.5)
            + 0.12 * noise
        )
    elif kind == "eaf":
        x = (
            1.70 * tone(F0, 0.1)
            + 0.16 * tone(2 * F0, 0.2)
            + 0.10 * tone(28, 0.4)
            + 0.07 * tone(34, 0.8)
            + 0.06 * tone(38, 1.2)
            + 0.45 * (1 + 0.55 * tone(F0)) * tone(210, 0.3)
            + 0.10 * noise
        )
    elif kind == "bpf":
        x = (
            0.40 * tone(F0, 0.2)
            + 0.55 * tone(0.5 * F0, 0.6)
            + 0.28 * tone(3 * F0, 0.2)
            + 0.12 * tone(5 * F0, 1.0)
            + 0.10 * tone(6 * F0, 0.4)
            + 0.85 * impulses(F0, 1.0, 0.0014, rng, 2400)
            + 0.40 * (1 + 0.7 * tone(0.5 * F0, 0.2)) * tone(230, 0.6)
            + 0.14 * noise
        )
    else:
        x = (
            0.75 * tone(F0, 0.3)
            + 0.48 * tone(0.5 * F0, 0.8)
            + 0.40 * tone(2 * F0, 0.15)
            + 0.60 * impulses(2 * F0, 0.8, 0.0016, rng, 1600)
            + 0.38 * (1 + 0.45 * tone(2 * F0)) * tone(200, 0.9)
            + 0.13 * noise
        )
    return x


def rms_velocity_mm_s(acc):
    vel = np.cumsum(acc) / FS
    b, a = signal.butter(3, 2.0 / (FS / 2), btype="high")
    vel = signal.filtfilt(b, a, vel)
    return float(np.sqrt(np.mean(vel ** 2)) * 1000.0)


def dwt_detail_energy(x, levels=8):
    a = np.asarray(x, dtype=float)
    energies = []
    for _ in range(levels):
        if len(a) < len(DEC_LO):
            break
        pad = len(DEC_LO) - 1
        ap = np.concatenate([a[-pad:], a])
        detail = np.convolve(ap, DEC_HI, mode="valid")[::2]
        approx = np.convolve(ap, DEC_LO, mode="valid")[::2]
        L = len(a) // 2
        detail = detail[:L]
        approx = approx[:L]
        energies.append(float(np.sum(detail ** 2)))
        a = approx
    energies.append(float(np.sum(a ** 2)))
    return np.array(energies)


STATES = ["normal", "eaf", "bpf", "srwf"]
NAMES = {
    "normal": "Normal",
    "eaf": "Eccentric axle (EAF)",
    "bpf": "Bearing pedestal (BPF)",
    "srwf": "Sealing-ring wear (SRWF)",
}
COLORS = {
    "normal": "#1f4e79",
    "eaf": "#b85c38",
    "bpf": "#2e7d4f",
    "srwf": "#6b3fa0",
}
PUBLISHED = {"normal": 1.051, "eaf": 3.737, "bpf": 1.161, "srwf": 1.70024}

raw = {k: make_signal(k, s) for k, s in zip(STATES, [11, 23, 37, 41])}
sigs = {}
for k in STATES:
    cur = rms_velocity_mm_s(raw[k])
    sigs[k] = raw[k] * (PUBLISHED[k] / cur)

# --- waveforms ---
fig, axes = plt.subplots(4, 1, figsize=(7.4, 8.2), sharex=True)
for ax, k in zip(axes, STATES):
    ax.plot(T * 1000, sigs[k], color=COLORS[k], lw=0.6)
    ax.set_ylabel("Accel.\n(m/s$^2$)")
    ax.set_title(NAMES[k], loc="left", fontsize=10)
    ax.set_xlim(0, 200)
axes[-1].set_xlabel("Time (ms)")
fig.suptitle("Illustrative radial acceleration (first 200 ms)", y=0.995, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "waveforms.pdf"))
plt.close()

# --- baseband spectra ---
fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
freqs = np.fft.rfftfreq(N, 1 / FS)
for ax, k in zip(axes.ravel(), STATES):
    spec = np.abs(np.fft.rfft(sigs[k] * np.hanning(N))) ** 2
    spec /= np.max(spec)
    ax.plot(freqs, spec, color=COLORS[k], lw=0.8)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.05)
    ax.set_title(NAMES[k], loc="left", fontsize=10)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized power")
    for h, lab in [(8, ""), (16, ""), (32, ""), (48, ""), (64, "")]:
        ax.axvline(h, color="0.75", lw=0.4, zorder=0)
fig.suptitle("Illustrative baseband spectra, 0–120 Hz", y=1.01, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "spectra.pdf"), bbox_inches="tight")
plt.close()

# --- wavelet energy ---
fig, ax = plt.subplots(figsize=(7.4, 4.2))
width = 0.18
levels = np.arange(1, 10)
for i, k in enumerate(STATES):
    e = dwt_detail_energy(sigs[k], 8)
    e = e / np.sum(e)
    ax.bar(levels + (i - 1.5) * width, e, width=width, color=COLORS[k], label=NAMES[k])
ax.set_xticks(levels)
ax.set_xticklabels([f"d{i}" for i in range(1, 9)] + ["a8"])
ax.set_ylabel("Fraction of signal energy")
ax.set_xlabel("Dyadic level (db4, $f_s = 10$ kHz)")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_title("Illustrative dyadic energy distribution")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "wavelet_energy.pdf"))
plt.close()

# --- published severity ---
fig, ax = plt.subplots(figsize=(6.6, 3.8))
vals = [PUBLISHED[k] for k in STATES]
bars = ax.bar([NAMES[k].split("(")[0].strip() for k in STATES], vals, color=[COLORS[k] for k in STATES])
ax.set_ylabel("Vibration severity (mm/s)")
ax.set_title("Published radial vibration severity (You et al., 2019)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, 4.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "severity.pdf"))
plt.close()

# --- accuracy charts from published tables ---
classes = ["Normal", "EAF", "BPF", "SRWF"]
methods = ["BPNN", "ACROA-SVM", "SFLA-SVM", "MSFLA-SVM"]
train = np.array([
    [42 / 45, 45 / 45, 45 / 45, 45 / 45],
    [41 / 45, 44 / 45, 43 / 45, 45 / 45],
    [40 / 45, 42 / 45, 42 / 45, 43 / 45],
    [42 / 45, 44 / 45, 44 / 45, 44 / 45],
]) * 100
# Test counts consistent with printed percentages and the 85% BPNN mean.
test = np.array([
    [13 / 15, 15 / 15, 15 / 15, 15 / 15],
    [13 / 15, 13 / 15, 14 / 15, 14 / 15],
    [12 / 15, 13 / 15, 13 / 15, 13 / 15],
    [13 / 15, 14 / 15, 13 / 15, 14 / 15],
]) * 100
mcolors = ["#8d99ae", "#d4a373", "#457b9d", "#1d3557"]

def grouped(data, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(classes))
    w = 0.18
    for i, m in enumerate(methods):
        ax.bar(x + (i - 1.5) * w, data[:, i], width=w, color=mcolors[i], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel(ylabel)
    ax.set_ylim(70, 105)
    ax.axhline(100, color="0.85", lw=0.6)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close()

grouped(train, "Training accuracy (%)", "Training classification rate by state and method", "train_accuracy.pdf")
grouped(test, "Testing accuracy (%)", "Testing classification rate by state and method", "test_accuracy.pdf")

# overall bars
overall_train = train.mean(axis=0)  # not quite: should be pooled
# pooled:
pooled_train = np.array([165, 175, 174, 177]) / 180 * 100
pooled_test = np.array([51, 55, 55, 56]) / 60 * 100
fig, ax = plt.subplots(figsize=(6.8, 3.9))
x = np.arange(len(methods))
ax.bar(x - 0.16, pooled_train, width=0.32, color="#457b9d", label="Training (180 samples)")
ax.bar(x + 0.16, pooled_test, width=0.32, color="#1d3557", label="Testing (60 samples)")
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=10)
ax.set_ylim(75, 105)
ax.set_ylabel("Pooled accuracy (%)")
ax.legend(frameon=False, fontsize=8)
ax.set_title("Pooled accuracy of the four classifiers")
for i, (a, b) in enumerate(zip(pooled_train, pooled_test)):
    ax.text(i - 0.16, a + 0.4, f"{a:.1f}", ha="center", fontsize=7)
    ax.text(i + 0.16, b + 0.4, f"{b:.1f}", ha="center", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pooled_accuracy.pdf"))
plt.close()

# harmonic family diagram-like stem of characteristic frequencies
fig, ax = plt.subplots(figsize=(7.2, 3.2))
harms = np.array([0.5, 1, 2, 3, 4, 5, 6]) * F0
ax.stem(harms, np.ones_like(harms), linefmt="C0-", markerfmt="o", basefmt=" ")
ax.axvline(50, color="#b85c38", ls="--", lw=1, label="Mains (50 Hz)")
ax.set_xlabel("Frequency (Hz)")
ax.set_yticks([])
ax.set_title(r"Characteristic frequency family at $n = 960$ r/min ($f_0 = 16$ Hz)")
for h, m in zip(harms, ["1/2 f0", "f0", "2f0", "3f0", "4f0", "5f0", "6f0"]):
    ax.text(h, 1.05, f"{m}\n{h:.0f} Hz", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, 1.45)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "harmonics.pdf"))
plt.close()

# record-length annotation figure: revolutions inside one analysis frame
fig, ax = plt.subplots(figsize=(7.2, 2.6))
revs = T * F0
ax.plot(T, np.sin(2 * np.pi * F0 * T), color="#1f4e79", lw=0.8)
ax.set_xlim(0, 4 / F0)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Shaft-synchronous\nreference")
ax.set_title("Four shaft revolutions at 16 Hz (one analysis record contains about 13.1 revolutions)")
for k in range(4):
    ax.axvline(k / F0, color="0.7", lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "shaft_cycles.pdf"))
plt.close()

print("wrote figures to", OUT)
for k in STATES:
    print(k, "matched VIB", rms_velocity_mm_s(sigs[k]))

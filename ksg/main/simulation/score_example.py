"""Print the attention arithmetic for the navy-shell example."""
import math
import numpy as np

p = np.array([
    -1.2439188470162474, 0.028318340480852112, -0.3134663703033141,
    -1.447850889076455, -0.5006626031786545, 1.733503682730396,
    -0.09246996255616743, 0.23845640599751367, -2.0171209242662003,
    -0.08716011784200091, -1.0399511211354717, 0.5052695762499068,
    1.0824964065311116, 0.9691951302908979, -0.7067555438595792,
    0.6718255665187929, 0.8519485183141255, 1.0234018494099077,
    1.891504964138816, -1.6209892359181284,
])
w = np.exp(p[:12])
a = p[12:16]
g = p[16:20]
omega = np.array([0.34, 0.33, 0.18, 0.15])
pref = np.array([
    0.55 + g[0] * (omega[0] + omega[1]),
    0.55 + g[1] * (omega[2] + 0.5 * omega[0]),
    0.45 + g[2] * omega[3],
    0.45 + g[3] * (omega[1] + 0.25),
])
print("w", np.round(w, 3))
print("pref_open", np.round(pref, 3))

def score(name, raw):
    k_op = raw[0] + raw[1] + 1.4 * raw[2] + raw[3]
    k_m = raw[4] + raw[5] + 0.35 * raw[6] + 1.6 * raw[7]
    k_o = raw[11] + 0.3 * raw[10]
    k_s = raw[8] + raw[9] + 0.5 * raw[10]
    keys = np.array([k_op, k_m, k_o, k_s])
    node = np.array([
        w[0] * raw[0] + w[1] * raw[1] + w[2] * raw[2] + w[3] * raw[3],
        w[4] * raw[4] + w[5] * raw[5] + w[6] * raw[6] + w[7] * raw[7],
        w[11] * raw[11],
        w[8] * raw[8] + w[9] * raw[9] + w[10] * raw[10],
    ])
    logits = a + pref * keys
    z = logits - logits.max()
    alpha = np.exp(z) / np.exp(z).sum()
    psi = float(np.dot(alpha, node * pref))
    print(name)
    print("  raw", np.round(raw, 3))
    print("  keys", np.round(keys, 3))
    print("  node", np.round(node, 3))
    print("  alpha", np.round(alpha, 3))
    print("  psi", round(psi, 4))

# Cutter A, comfortable slack, one other cut waiting, balanced load.
# pt 27.268/45, setup 0, slack_pen 0, crit 0.8, h 0.58, load 0,
# energy 3.896, phi 0.106, color 0.5 (1 of 2), die 0.5, group 0.67, skill gap 0
score("A", np.array([
    27.268 / 45, 0, 0, 0.80, 0.58, 0.0, 3.896, 0.1061, 0.5, 0.5, 2 / 3, 0.0
]))
score("B", np.array([
    37.265 / 45, 0, 0, 0.95, 0.22, 0.10, 1.638, 0.0, 0.5, 0.5, 2 / 3, 0.0
]))

import json

d = json.load(open(r"data/results.json", encoding="utf-8"))
for sc, methods in d["scenarios"].items():
    print("==", sc)
    for m, s in methods.items():
        print(
            f"{m:18} cmax {s['cmax_mean']:.1f}+/-{s['cmax_std']:.1f} "
            f"tard {s['tardiness_mean']:.1f}+/-{s['tardiness_std']:.1f} "
            f"E {s['energy_mean']:.1f}+/-{s['energy_std']:.1f} "
            f"fail {s['failures_mean']:.2f}+/-{s['failures_std']:.2f} "
            f"on {s['ontime_mean']:.3f}+/-{s['ontime_std']:.3f} "
            f"imb {s['imbalance_mean']:.1f}+/-{s['imbalance_std']:.1f} "
            f"setup {s['setups_mean']:.1f} pm {s['pms_mean']:.2f} bd {s['bd_delay_mean']:.1f}"
        )
for i, f in enumerate(d["nsga_fronts"]):
    cs = [p["cmax"] for p in f]
    ts = [p["tardiness"] for p in f]
    es = [p["energy"] for p in f]
    print(
        "nsga",
        i,
        "n",
        len(f),
        "bestc",
        round(min(cs), 1),
        "bestt",
        round(min(ts), 1),
        "beste",
        round(min(es), 1),
    )

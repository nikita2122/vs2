"""School-bag factory simulator for multi-target scheduling estimation.

The shop is a dual-resource flexible job shop: every operation needs one
eligible machine and one skilled operator, and cutting or colour-sensitive
operations also take a shared token (die or thread lot). Machine health
lengthens the cycle, raises energy, and creates an expected breakdown delay.
RCH-GAT is realised here as one-hop heterogeneous attention over the
operation, machine, operator, and shop nodes touched by a candidate
assignment. SSDT-CC adds a chance constraint: if the predicted failure
probability exceeds epsilon, a short preventive maintenance is inserted
before the operation and the health state is renewed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


MACHINE_GROUPS = ("cut", "print", "sew", "sew_heavy", "hardware", "qc", "pack")

MACHINES = [
    {"name": "Cutter-A", "group": "cut", "power": 6.8, "wear": 0.00115, "speed": 1.28, "h0": 0.58},
    {"name": "Cutter-B", "group": "cut", "power": 2.4, "wear": 0.00055, "speed": 0.74, "h0": 0.22},
    {"name": "Embroidery-1", "group": "print", "power": 3.4, "wear": 0.00050, "speed": 1.00, "h0": 0.46},
    {"name": "Lockstitch-1", "group": "sew", "power": 0.75, "wear": 0.00038, "speed": 1.05, "h0": 0.24},
    {"name": "Lockstitch-2", "group": "sew", "power": 0.75, "wear": 0.00040, "speed": 0.98, "h0": 0.52},
    {"name": "Bartack-1", "group": "sew_heavy", "power": 1.15, "wear": 0.00048, "speed": 1.00, "h0": 0.34},
    {"name": "Hardware-1", "group": "hardware", "power": 1.45, "wear": 0.00028, "speed": 1.00, "h0": 0.20},
    {"name": "QC-1", "group": "qc", "power": 0.35, "wear": 0.00008, "speed": 1.00, "h0": 0.08},
    {"name": "Pack-1", "group": "pack", "power": 0.55, "wear": 0.00008, "speed": 1.00, "h0": 0.12},
]

OPERATORS = [
    {"name": "Cut-1", "skill": {"cut": 1.00}},
    {"name": "Cut-2", "skill": {"cut": 0.82, "pack": 0.55}},
    {"name": "Print-1", "skill": {"print": 1.00, "qc": 0.45}},
    {"name": "Sew-1", "skill": {"sew": 1.06, "sew_heavy": 0.78}},
    {"name": "Sew-2", "skill": {"sew": 0.96, "sew_heavy": 1.08}},
    {"name": "Sew-3", "skill": {"sew": 1.00}},
    {"name": "Hard-1", "skill": {"hardware": 1.00, "pack": 0.70}},
    {"name": "QC-1", "skill": {"qc": 1.00, "pack": 0.95, "hardware": 0.62}},
]

COLORS = ("navy", "black", "red", "grey", "khaki")
COLOR_TOKENS = {"navy": 2, "black": 2, "red": 2, "grey": 2, "khaki": 1}
N_DIES = 2

# (name, eligible groups, base minutes at qty 80, token kind)
STYLES = {
    "backpack": [
        ("cut_main", ("cut",), 24.0, "die"),
        ("cut_lining", ("cut",), 16.0, "die"),
        ("print_logo", ("print",), 18.0, "color"),
        ("sew_body", ("sew",), 52.0, "color"),
        ("sew_strap", ("sew_heavy", "sew"), 30.0, "color"),
        ("zipper", ("hardware",), 22.0, "none"),
        ("qc", ("qc",), 14.0, "none"),
        ("pack", ("pack",), 11.0, "none"),
    ],
    "trolley": [
        ("cut_shell", ("cut",), 28.0, "die"),
        ("reinforce", ("sew_heavy", "sew"), 26.0, "color"),
        ("sew_shell", ("sew",), 58.0, "color"),
        ("frame", ("hardware",), 34.0, "none"),
        ("wheels", ("hardware",), 20.0, "none"),
        ("qc", ("qc",), 16.0, "none"),
        ("pack", ("pack",), 13.0, "none"),
    ],
    "sports": [
        ("cut_sack", ("cut",), 18.0, "die"),
        ("print_logo", ("print",), 14.0, "color"),
        ("sew_sack", ("sew",), 36.0, "color"),
        ("drawstring", ("hardware", "sew"), 16.0, "none"),
        ("qc", ("qc",), 12.0, "none"),
        ("pack", ("pack",), 9.0, "none"),
    ],
}

SIZE_FACTOR = {"S": 0.88, "M": 1.00, "L": 1.16}
H_FAIL = 0.40
REPAIR_MIN = 48.0
PM_MIN = 16.0
SETUP_MIN = 11.0
WARMUP_MIN = 4.0


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


@dataclass
class Op:
    job: int
    idx: int
    name: str
    groups: tuple
    base: float
    token: str
    pred: int


@dataclass
class Job:
    jid: int
    style: str
    color: str
    size: str
    qty: int
    release: float
    due: float
    ops: list


@dataclass
class Instance:
    jobs: list
    color_tokens: dict
    n_dies: int
    rush_time: float | None = None
    rush_jobs: list = field(default_factory=list)
    wear_scale: float = 1.0
    h0_shift: float = 0.0


def build_job(jid, style, color, size, qty, release, due) -> Job:
    spec = STYLES[style]
    ops = []
    for i, (name, groups, base, token) in enumerate(spec):
        ops.append(Op(jid, i, name, groups, base, token, i - 1))
    return Job(jid, style, color, size, qty, release, due, ops)


def make_instance(seed: int, scenario: str) -> Instance:
    rng = np.random.default_rng(seed)
    styles = ("backpack", "trolley", "sports")
    sizes = ("S", "M", "L")
    n = 16 if scenario != "large" else 24
    jobs = []
    for j in range(n):
        style = styles[int(rng.integers(0, 3))]
        color = COLORS[int(rng.integers(0, len(COLORS)))]
        size = sizes[int(rng.integers(0, 3))]
        qty = int(rng.choice([40, 60, 80, 80, 100, 120]))
        release = float(rng.choice([0.0, 0.0, 0.0, 40.0, 90.0, 150.0]))
        rough = sum(step[2] for step in STYLES[style]) * (qty / 80.0) * SIZE_FACTOR[size]
        # Due dates include queueing allowance. A pure process-time multiple
        # would make every order late once the embroidery and sewing cells queue.
        due = release + rough * float(rng.uniform(3.6, 6.4))
        jobs.append(build_job(j, style, color, size, qty, release, due))

    tokens = dict(COLOR_TOKENS)
    wear_scale = 1.0
    h0_shift = 0.0
    rush_time = None
    rush_jobs = []

    if scenario == "shortage":
        tokens = {c: (1 if c in ("khaki", "red", "navy") else v) for c, v in tokens.items()}
    elif scenario == "wear":
        wear_scale = 1.85
        h0_shift = 0.10
    elif scenario == "rush":
        rush_time = 220.0
        for k in range(4):
            style = styles[k % 3]
            color = COLORS[k % len(COLORS)]
            qty = 80
            size = "M"
            release = 220.0
            rough = sum(step[2] for step in STYLES[style]) * SIZE_FACTOR[size]
            due = release + rough * 2.15
            rush_jobs.append(build_job(100 + k, style, color, size, qty, release, due))
    elif scenario == "large":
        wear_scale = 1.15

    return Instance(jobs, tokens, N_DIES, rush_time, rush_jobs, wear_scale, h0_shift)


def processing_time(op: Op, job: Job, machine: dict, operator: dict, health: float) -> float:
    skill = operator["skill"].get(machine["group"], 0.0)
    qty_f = job.qty / 80.0
    size_f = SIZE_FACTOR[job.size]
    health_f = 1.0 + 0.85 * max(health, 0.0) ** 1.15
    return op.base * qty_f * size_f * health_f / (machine["speed"] * skill)


def failure_probability(health: float, minutes: float, wear_scale: float) -> float:
    # Weibull-like hazard integrated over the cycle, evaluated at the
    # health expected at the middle of the cycle.
    mid = health + 0.5 * 0.0004 * wear_scale * minutes
    lam = 2.4 * max(0.0, mid - H_FAIL) ** 1.35
    hours = minutes / 60.0
    return float(1.0 - np.exp(-lam * hours))


def energy_kwh(power: float, minutes: float, health: float) -> float:
    return power * (minutes / 60.0) * (1.0 + 0.45 * max(health, 0.0))


class Shop:
    def __init__(self, inst: Instance, use_ssdt: bool, epsilon: float, rng: np.random.Generator):
        self.inst = inst
        self.use_ssdt = use_ssdt
        self.epsilon = epsilon
        self.rng = rng
        self.now = 0.0
        self.jobs = list(inst.jobs)
        self.n_m = len(MACHINES)
        self.n_o = len(OPERATORS)
        self.h = np.array([m["h0"] + inst.h0_shift for m in MACHINES], dtype=float)
        self.m_free = np.zeros(self.n_m)
        self.o_free = np.zeros(self.n_o)
        self.last_color = [None] * self.n_m
        self.busy = np.zeros(self.n_m)
        self.done = [[False] * len(j.ops) for j in self.jobs]
        self.job_done_t = [None] * len(self.jobs)
        self.color_used = {c: 0 for c in COLORS}
        self.die_used = 0
        self.energy = 0.0
        self.setups = 0
        self.pms = 0
        self.expected_failures = 0.0
        self.breakdown_delay = 0.0
        self.gantt = []
        self.decisions = 0
        self.rush_inserted = inst.rush_time is None
        self.frozen = None  # (priority map, machine pref map) or None

    def _insert_rush(self):
        if self.inst.rush_time is None or self.rush_inserted:
            return
        if self.now + 1e-9 >= self.inst.rush_time:
            base = len(self.jobs)
            for job in self.inst.rush_jobs:
                self.jobs.append(job)
                self.done.append([False] * len(job.ops))
                self.job_done_t.append(None)
            self.rush_inserted = True
            if self.frozen is not None:
                pri, pref = self.frozen
                # Frozen plans keep their original keys; new work is last.
                for j in range(base, len(self.jobs)):
                    for i in range(len(self.jobs[j].ops)):
                        pri[(j, i)] = 1e6 + j * 10 + i
                        pref[(j, i)] = 0

    def _ready_ops(self):
        ready = []
        for j, job in enumerate(self.jobs):
            if job.release > self.now + 1e-9:
                continue
            for op in job.ops:
                if self.done[j][op.idx]:
                    continue
                if op.pred >= 0 and not self.done[j][op.pred]:
                    continue
                ready.append(op)
                break  # chain precedence: only the first unfinished op is ready
        return ready

    def _remaining_after(self, op: Op) -> float:
        job = self.jobs[op.job] if op.job < len(self.jobs) else None
        # jobs list index equals position; Op.job stores original jid which for
        # base jobs equals index. Rush jobs use jid 100+. Map by identity.
        return 0.0

    def _job_index(self, op: Op) -> int:
        return op.job if op.job < len(self.jobs) and self.jobs[op.job].jid == op.job else next(
            i for i, j in enumerate(self.jobs) if j.jid == op.job
        )

    def remaining_work(self, op: Op) -> float:
        j = self._job_index(op)
        job = self.jobs[j]
        total = 0.0
        for later in job.ops[op.idx + 1 :]:
            total += later.base * (job.qty / 80.0) * SIZE_FACTOR[job.size]
        return total

    def _token_ok(self, op: Op, job: Job) -> bool:
        if op.token == "die":
            return self.die_used < self.inst.n_dies
        if op.token == "color":
            return self.color_used[job.color] < self.inst.color_tokens[job.color]
        return True

    def _hold_token(self, op: Op, job: Job, delta: int):
        if op.token == "die":
            self.die_used += delta
        elif op.token == "color":
            self.color_used[job.color] += delta

    def candidates(self):
        cands = []
        ready = self._ready_ops()
        if not ready:
            return cands
        contenders_color = {}
        contenders_group = {g: 0 for g in MACHINE_GROUPS}
        cut_ready = 0
        for op in ready:
            job = self.jobs[self._job_index(op)]
            if op.token == "color":
                contenders_color[job.color] = contenders_color.get(job.color, 0) + 1
            if op.token == "die":
                cut_ready += 1
            for g in op.groups:
                contenders_group[g] += 1
        for op in ready:
            j = self._job_index(op)
            job = self.jobs[j]
            if not self._token_ok(op, job):
                continue
            for mi, machine in enumerate(MACHINES):
                if machine["group"] not in op.groups:
                    continue
                if self.m_free[mi] > self.now + 1e-9:
                    continue
                for oi, operator in enumerate(OPERATORS):
                    skill = operator["skill"].get(machine["group"], 0.0)
                    if skill <= 0.0:
                        continue
                    if self.o_free[oi] > self.now + 1e-9:
                        continue
                    pt = processing_time(op, job, machine, operator, float(self.h[mi]))
                    setup = 0.0
                    prev = self.last_color[mi]
                    if machine["group"] in ("cut", "print", "sew", "sew_heavy"):
                        if prev is None:
                            setup = WARMUP_MIN
                        elif prev != job.color:
                            setup = SETUP_MIN
                    cands.append(
                        {
                            "op": op,
                            "j": j,
                            "mi": mi,
                            "oi": oi,
                            "pt": pt,
                            "setup": setup,
                            "skill": skill,
                            "color_s": contenders_color.get(job.color, 0)
                            / max(1, self.inst.color_tokens[job.color]),
                            "die_s": cut_ready / max(1, self.inst.n_dies),
                            "group_s": contenders_group[machine["group"]] / 3.0,
                        }
                    )
        return cands

    def _apply(self, cand, pm: bool):
        op = cand["op"]
        j = cand["j"]
        mi = cand["mi"]
        oi = cand["oi"]
        job = self.jobs[j]
        machine = MACHINES[mi]
        health = float(self.h[mi])
        pm_time = PM_MIN if pm else 0.0
        if pm:
            # Preventive maintenance renews the degradation state before cutting
            # or sewing, then the cycle is recomputed at the renewed health.
            self.h[mi] = 0.30 * health + 0.04
            health = float(self.h[mi])
            self.pms += 1
            pt = processing_time(op, job, machine, OPERATORS[oi], health)
        else:
            pt = cand["pt"]
        setup = cand["setup"]
        minutes = pm_time + setup + pt
        p_fail = failure_probability(health, pt, self.inst.wear_scale)
        extra = p_fail * REPAIR_MIN
        self.expected_failures += p_fail
        self.breakdown_delay += extra
        end = self.now + minutes + extra
        self.energy += energy_kwh(machine["power"], minutes, health)
        self.energy += p_fail * energy_kwh(machine["power"] * 0.5, REPAIR_MIN, health)
        self.setups += 1 if setup >= SETUP_MIN - 1e-6 else 0
        self.busy[mi] += minutes
        self.m_free[mi] = end
        self.o_free[oi] = end
        self.last_color[mi] = job.color
        wear = machine["wear"] * self.inst.wear_scale * pt
        self.h[mi] = min(1.25, health + wear)
        self._hold_token(op, job, +1)
        self.gantt.append(
            {
                "job": int(job.jid),
                "op": op.name,
                "machine": machine["name"],
                "start": round(self.now, 2),
                "end": round(end, 2),
                "pm": bool(pm),
            }
        )
        self._occupancy.append((end, j, op.idx, mi, oi, op, job))
        self.decisions += 1

    def _release_finished(self):
        still = []
        for item in self._occupancy:
            end, j, idx, mi, oi, op, job = item
            if end <= self.now + 1e-8:
                self.done[j][idx] = True
                self._hold_token(op, job, -1)
                if all(self.done[j]):
                    self.job_done_t[j] = end
            else:
                still.append(item)
        self._occupancy = still

    def _next_time(self) -> float | None:
        times = []
        if self._occupancy:
            times.append(min(item[0] for item in self._occupancy))
        for job in self.jobs:
            if job.release > self.now + 1e-9:
                times.append(job.release)
        if self.inst.rush_time is not None and not self.rush_inserted:
            times.append(self.inst.rush_time)
        if not times:
            return None
        return min(times)

    def run(self, decide_fn, frozen=None):
        self._occupancy = []
        self.frozen = frozen
        guard = 0
        while True:
            guard += 1
            if guard > 20000:
                raise RuntimeError("simulation did not terminate")
            self._insert_rush()
            self._release_finished()
            if all(all(row) for row in self.done):
                break
            cands = self.candidates()
            if cands:
                choice, pm = decide_fn(self, cands)
                self._apply(cands[choice], pm)
                continue
            nxt = self._next_time()
            if nxt is None:
                # No candidate and no future event: deadlock (resource leak).
                raise RuntimeError("deadlock at t=%.2f" % self.now)
            self.now = nxt
        completions = [t for t in self.job_done_t if t is not None]
        tard = []
        on_time = 0
        for j, job in enumerate(self.jobs):
            c = self.job_done_t[j]
            late = max(0.0, c - job.due)
            tard.append(late)
            if late <= 1e-6:
                on_time += 1
        busy = self.busy
        imb = float(np.std(busy))
        return {
            "cmax": float(max(completions) if completions else 0.0),
            "tardiness": float(sum(tard)),
            "mean_tardiness": float(np.mean(tard) if tard else 0.0),
            "ontime": float(on_time / max(1, len(self.jobs))),
            "energy": float(self.energy),
            "setups": int(self.setups),
            "imbalance": imb,
            "failures": float(self.expected_failures),
            "bd_delay": float(self.breakdown_delay),
            "pms": int(self.pms),
            "decisions": int(self.decisions),
            "gantt": self.gantt,
        }


def _node_costs(shop: Shop, cand, w):
    job = shop.jobs[cand["j"]]
    rem = shop.remaining_work(cand["op"])
    slack = job.due - (shop.now + cand["pt"] + cand["setup"] + rem)
    pt_n = cand["pt"] / 45.0
    setup_n = cand["setup"] / SETUP_MIN
    slack_pen = max(0.0, -slack) / 180.0
    crit = (cand["pt"] + rem) / max(30.0, job.due - shop.now)
    health = float(shop.h[cand["mi"]])
    load_dev = (shop.busy[cand["mi"]] - float(np.mean(shop.busy))) / 80.0
    p_fail = failure_probability(health, cand["pt"], shop.inst.wear_scale)
    energy = energy_kwh(MACHINES[cand["mi"]]["power"], cand["pt"] + cand["setup"], health)
    # Twelve interpretable costs. Learned weights stay positive.
    raw = np.array(
        [
            pt_n,
            setup_n,
            slack_pen,
            crit,
            health,
            max(0.0, load_dev),
            energy,
            p_fail,
            cand["color_s"],
            cand["die_s"],
            cand["group_s"],
            max(0.0, 1.05 - cand["skill"]),
        ],
        dtype=float,
    )
    # Attention keys for the four heterogeneous nodes.
    k_op = raw[0] + raw[1] + 1.4 * raw[2] + raw[3]
    k_m = raw[4] + raw[5] + 0.35 * raw[6] + 1.6 * raw[7]
    k_o = raw[11] + 0.3 * raw[10]
    k_s = raw[8] + raw[9] + 0.5 * raw[10]
    return raw, np.array([k_op, k_m, k_o, k_s], dtype=float)


def rch_choose(shop: Shop, cands, params, omega, use_resource=True, use_status=True):
    """One-hop heterogeneous attention over operation, machine, operator, shop."""
    w = np.exp(params[:12])  # positive weights on the twelve costs
    a = params[12:16]  # attention logits bias
    gate = params[16:20]  # how strongly each objective opens a node
    best_i, best_s, best_pm = 0, 1e18, False
    om = np.asarray(omega, dtype=float)
    # Preference opens nodes: time/tardiness -> op, energy/health -> machine,
    # balance -> operator, due-date pressure -> resource contention.
    pref_open = np.array(
        [
            0.55 + gate[0] * (om[0] + om[1]),
            0.55 + gate[1] * (om[2] + 0.5 * om[0]),
            0.45 + gate[2] * om[3],
            0.45 + gate[3] * (om[1] + 0.25),
        ]
    )
    for i, cand in enumerate(cands):
        raw, keys = _node_costs(shop, cand, w)
        if not use_resource:
            raw = raw.copy()
            raw[8:11] = 0.0
            keys = keys.copy()
            keys[3] = 0.0
        if not use_status:
            raw = raw.copy()
            raw[4] = 0.0
            raw[7] = 0.0
            keys = keys.copy()
            keys[1] = raw[5] + 0.35 * raw[6]
        node_cost = np.array(
            [
                w[0] * raw[0] + w[1] * raw[1] + w[2] * raw[2] + w[3] * raw[3],
                w[4] * raw[4] + w[5] * raw[5] + w[6] * raw[6] + w[7] * raw[7],
                w[11] * raw[11],
                w[8] * raw[8] + w[9] * raw[9] + w[10] * raw[10],
            ]
        )
        alpha = _softmax(a + pref_open * keys)
        score = float(np.dot(alpha, node_cost * pref_open))
        health = float(shop.h[cand["mi"]])
        p_fail = failure_probability(health, cand["pt"], shop.inst.wear_scale)
        pred_h = health + MACHINES[cand["mi"]]["wear"] * shop.inst.wear_scale * cand["pt"]
        pm = False
        if shop.use_ssdt and (p_fail > shop.epsilon or pred_h > 0.90):
            pm = True
            # PM spends a short window but removes most of the failure cost.
            score += 0.15 * pref_open[1]
            score -= node_cost[1] * 0.45
        if score < best_s:
            best_s = score
            best_i = i
            best_pm = pm
    return best_i, best_pm


def rule_choose(shop: Shop, cands, rule: str):
    best_i, best_key = 0, None
    for i, cand in enumerate(cands):
        job = shop.jobs[cand["j"]]
        if rule == "FIFO":
            key = (job.release, job.jid, cand["op"].idx, cand["pt"])
        elif rule == "SPT":
            key = (cand["pt"] + cand["setup"], job.due, job.jid)
        elif rule == "EDD":
            key = (job.due, cand["pt"], job.jid)
        elif rule == "ATC":
            rem = shop.remaining_work(cand["op"])
            slack = job.due - (shop.now + cand["pt"] + rem)
            p_avg = 40.0
            atc = (1.0 / max(1.0, cand["pt"])) * np.exp(-max(0.0, slack) / (2.0 * p_avg))
            key = (-atc, cand["setup"], job.jid)
        else:
            raise ValueError(rule)
        if best_key is None or key < best_key:
            best_key = key
            best_i = i
    return best_i, False


def scalar_cost(m, ref):
    # Every term is a ratio to FIFO on the same instance, so a value below
    # one means a better balanced schedule than first-in-first-out.
    return (
        0.30 * m["cmax"] / max(1.0, ref["cmax"])
        + 0.30 * (m["tardiness"] + 5.0) / (ref["tardiness"] + 5.0)
        + 0.16 * m["energy"] / max(1.0, ref["energy"])
        + 0.12 * (m["imbalance"] + 1.0) / (ref["imbalance"] + 1.0)
        + 0.12 * (m["failures"] + 0.2) / (ref["failures"] + 0.2)
    )


def evaluate_params(params, seeds, scenario, omega, use_ssdt, epsilon, use_resource=True, use_status=True):
    scores = []
    for seed in seeds:
        inst = make_instance(seed, scenario)
        rng = np.random.default_rng(seed + 17)
        base_shop = Shop(inst, False, 1.0, rng)
        base = base_shop.run(lambda s, c: rule_choose(s, c, "FIFO"))
        rng2 = np.random.default_rng(seed + 17)

        def decide(s, c, params=params):
            return rch_choose(s, c, params, omega, use_resource, use_status)

        shop = Shop(inst, use_ssdt, epsilon, rng2)
        metrics = shop.run(decide)
        scores.append(scalar_cost(metrics, base))
    return float(np.mean(scores))


def train_rch(generations=28, pop=36, elite=8):
    rng = np.random.default_rng(2026)
    dim = 20
    mean = np.zeros(dim)
    # Start from a mild SPT / slack / health preference (log-weights).
    mean[:12] = np.log(
        np.array([1.1, 0.8, 1.3, 0.6, 0.7, 0.4, 0.5, 0.9, 0.8, 0.5, 0.6, 0.7])
    )
    mean[12:16] = np.array([0.2, 0.0, 0.0, 0.1])
    mean[16:20] = np.array([0.8, 0.8, 0.8, 0.8])
    std = np.ones(dim) * 0.35
    train_seeds = [11, 23, 37, 41]
    history = []
    best_params = mean.copy()
    best_score = 1e9
    omega = np.array([0.34, 0.33, 0.18, 0.15])
    for gen in range(generations):
        samples = []
        fits = []
        for k in range(pop):
            theta = mean + std * rng.normal(size=dim)
            fit = evaluate_params(theta, train_seeds, "nominal", omega, True, 0.10)
            samples.append(theta)
            fits.append(fit)
        order = np.argsort(fits)
        elites = np.stack([samples[i] for i in order[:elite]], axis=0)
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 0.04
        if fits[order[0]] < best_score:
            best_score = float(fits[order[0]])
            best_params = samples[order[0]].copy()
        history.append({"gen": gen, "best": float(fits[order[0]]), "mean": float(np.mean(fits))})
        print(f"gen {gen:02d} best {fits[order[0]]:.4f} mean {np.mean(fits):.4f}", flush=True)
    return best_params, history, best_score


def run_method(inst, method, params, omega, epsilon, seed):
    rng = np.random.default_rng(seed + 91)
    if method == "FIFO":
        shop = Shop(inst, False, 1.0, rng)
        return shop.run(lambda s, c: rule_choose(s, c, "FIFO"))
    if method == "SPT":
        shop = Shop(inst, False, 1.0, rng)
        return shop.run(lambda s, c: rule_choose(s, c, "SPT"))
    if method == "EDD":
        shop = Shop(inst, False, 1.0, rng)
        return shop.run(lambda s, c: rule_choose(s, c, "EDD"))
    if method == "ATC":
        shop = Shop(inst, False, 1.0, rng)
        return shop.run(lambda s, c: rule_choose(s, c, "ATC"))
    if method == "RCH":
        shop = Shop(inst, False, 1.0, rng)
        return shop.run(lambda s, c: rch_choose(s, c, params, omega, True, True))
    if method == "RCH-SSDT":
        shop = Shop(inst, True, epsilon, rng)
        return shop.run(lambda s, c: rch_choose(s, c, params, omega, True, True))
    if method == "RCH-no-resource":
        shop = Shop(inst, True, epsilon, rng)
        return shop.run(lambda s, c: rch_choose(s, c, params, omega, False, True))
    if method == "RCH-no-status":
        shop = Shop(inst, False, 1.0, rng)
        return shop.run(lambda s, c: rch_choose(s, c, params, omega, True, False))
    raise ValueError(method)


def summarise(rows):
    keys = ["cmax", "tardiness", "mean_tardiness", "ontime", "energy", "setups", "imbalance", "failures", "bd_delay", "pms"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k + "_mean"] = float(vals.mean())
        out[k + "_std"] = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
    return out


def nsga_front_points(seed: int, params_unused=None):
    """Small NSGA-II on one nominal instance, used as an evolutionary reference."""
    inst = make_instance(seed, "nominal")
    rng = np.random.default_rng(seed + 3)
    # Chromosome: priority key per base operation and a machine bias.
    ops = [(j, op.idx) for j, job in enumerate(inst.jobs) for op in job.ops]
    n = len(ops)

    def decode(prio, bias):
        order = {ops[i]: prio[i] for i in range(n)}

        def decide(shop, cands, order=order, bias=bias):
            best_i, best_key = 0, None
            for i, cand in enumerate(cands):
                key = (
                    order.get((cand["j"], cand["op"].idx), 0.0) + 0.15 * bias[cand["mi"]],
                    cand["pt"],
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_i = i
            return best_i, False

        shop = Shop(inst, False, 1.0, np.random.default_rng(seed + 5))
        return shop.run(decide)

    pop_n, gens = 18, 12
    P = rng.random((pop_n, n))
    B = rng.normal(size=(pop_n, len(MACHINES)))
    def evaluate_all(P, B):
        fits = []
        for i in range(len(P)):
            m = decode(P[i], B[i])
            fits.append((m["cmax"], m["tardiness"], m["energy"]))
        return fits

    fits = evaluate_all(P, B)

    def nondominated_sort(fits):
        ranks = []
        for i, a in enumerate(fits):
            dom = 0
            for b in fits:
                if b[0] <= a[0] and b[1] <= a[1] and b[2] <= a[2] and (
                    b[0] < a[0] or b[1] < a[1] or b[2] < a[2]
                ):
                    dom += 1
            ranks.append(dom)
        return ranks

    for _gen in range(gens):
        ranks = nondominated_sort(fits)
        kids_p = []
        kids_b = []
        while len(kids_p) < pop_n:
            i, j = rng.integers(0, pop_n, size=2)
            if ranks[i] > ranks[j]:
                i = j
            child_p = np.clip(P[i] + 0.15 * rng.normal(size=n), 0, 1)
            child_b = B[i] + 0.15 * rng.normal(size=len(MACHINES))
            mut = rng.random(n) < 0.08
            child_p[mut] = rng.random(np.sum(mut))
            kids_p.append(child_p)
            kids_b.append(child_b)
        C = np.stack(kids_p)
        CB = np.stack(kids_b)
        cfits = evaluate_all(C, CB)
        P = np.concatenate([P, C], axis=0)
        B = np.concatenate([B, CB], axis=0)
        fits = fits + cfits
        ranks = nondominated_sort(fits)
        keep = np.argsort(ranks)[:pop_n]
        P, B = P[keep], B[keep]
        fits = [fits[i] for i in keep]
    ranks = nondominated_sort(fits)
    front = [fits[i] for i in range(pop_n) if ranks[i] == 0]
    return front


def main():
    print("training RCH-GAT parameters", flush=True)
    params, history, best_score = train_rch()
    np.save("data/rch_params.npy", params)
    scenarios = ["nominal", "rush", "wear", "shortage"]
    methods = [
        "FIFO",
        "SPT",
        "EDD",
        "ATC",
        "RCH",
        "RCH-SSDT",
        "RCH-no-resource",
        "RCH-no-status",
    ]
    seeds = list(range(20))
    omega = [0.34, 0.33, 0.18, 0.15]
    epsilon = 0.10
    results = {"train": history, "train_best": best_score, "scenarios": {}, "params": params.tolist()}
    for scenario in scenarios:
        results["scenarios"][scenario] = {}
        for method in methods:
            rows = []
            gantt = None
            for seed in seeds:
                inst = make_instance(1000 + seed, scenario)
                metrics = run_method(inst, method, params, omega, epsilon, 1000 + seed)
                if gantt is None and seed == 0 and method in ("FIFO", "RCH-SSDT", "SPT"):
                    gantt = metrics["gantt"]
                row = dict(metrics)
                row.pop("gantt")
                rows.append(row)
            summary = summarise(rows)
            summary["gantt_seed0"] = gantt
            results["scenarios"][scenario][method] = summary
            print(scenario, method, {k: round(summary[k], 2) for k in ("cmax_mean", "tardiness_mean", "energy_mean", "failures_mean", "ontime_mean")}, flush=True)

    # Preference frontier on 8 nominal seeds.
    frontier = []
    grid = [
        [0.55, 0.25, 0.10, 0.10],
        [0.25, 0.55, 0.10, 0.10],
        [0.20, 0.20, 0.45, 0.15],
        [0.20, 0.20, 0.15, 0.45],
        [0.34, 0.33, 0.18, 0.15],
    ]
    labels = ["time", "due", "energy", "balance", "balanced"]
    for label, om in zip(labels, grid):
        rows = []
        for seed in range(8):
            inst = make_instance(3000 + seed, "nominal")
            metrics = run_method(inst, "RCH-SSDT", params, om, epsilon, 3000 + seed)
            rows.append(metrics)
        sm = summarise(rows)
        frontier.append({"label": label, "omega": om, **sm})
        print("frontier", label, round(sm["cmax_mean"], 1), round(sm["tardiness_mean"], 1), round(sm["energy_mean"], 2), flush=True)
    results["frontier"] = frontier

    eps_sweep = []
    for eps in (0.02, 0.05, 0.10, 0.20, 0.50):
        rows = []
        for seed in range(8):
            inst = make_instance(4000 + seed, "wear")
            metrics = run_method(inst, "RCH-SSDT", params, omega, eps, 4000 + seed)
            rows.append(metrics)
        sm = summarise(rows)
        eps_sweep.append({"epsilon": eps, **sm})
        print("eps", eps, round(sm["cmax_mean"], 1), round(sm["failures_mean"], 3), round(sm["pms_mean"], 2), flush=True)
    results["epsilon_sweep"] = eps_sweep

    print("nsga reference", flush=True)
    fronts = []
    for seed in range(4):
        front = nsga_front_points(5000 + seed)
        fronts.append([{"cmax": float(a), "tardiness": float(b), "energy": float(c)} for a, b, c in front])
        print("nsga seed", seed, "front", len(front), flush=True)
    results["nsga_fronts"] = fronts

    # Drop bulky gantt except a short prefix for the figure.
    for scenario in results["scenarios"]:
        for method in results["scenarios"][scenario]:
            g = results["scenarios"][scenario][method].get("gantt_seed0")
            if g:
                results["scenarios"][scenario][method]["gantt_seed0"] = g[:18]

    with open("data/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("wrote data/results.json", flush=True)


if __name__ == "__main__":
    main()

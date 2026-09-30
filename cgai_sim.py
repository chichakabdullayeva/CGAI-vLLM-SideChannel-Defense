#!/usr/bin/env python3
"""
CGAI v2 simulation (stdlib only): Counter-Guided Adaptive Isolation on a vLLM-style prefix cache.

Tier A  HMAC salting of prefix-cache block hashes (root or boundary).
Tier B  Watcher: 'legacy' (reconstruction of the v1 heuristic) or 'refined' (leaky bucket + template-stability gate).
Tier C  Latency injection, applied only to flagged tenants.

Nothing here is hardware-measured. Latency is calibrated to KVGov's A100 numbers
(cold 149.6 ms vs cached 32.8 ms at 2,119 tokens -> 32.8 ms + 0.0551 ms per uncached token).
Everything else (workload, rates, watcher parameters) is an assumption stated in the code.
No parameter was tuned against a target overhead. Usage:
    python cgai_sim.py [--seeds 7 11] [--duration 300] [--trials 20] [--caches 12000 4000]
"""
import argparse, copy, hashlib, hmac, itertools, math, random, statistics, sys, time
from collections import OrderedDict, defaultdict

BLOCK = 16                                   # tokens per KV block
T_FIXED = 32.8                               # ms, KVGov cached TTFT
T_PER_TOK = (149.6 - 32.8) / 2119            # ms per uncached token (~0.0551)
SIGMA = 0.03                                 # lognormal service-time noise (assumption)
PRE_N = 64                                   # public shared preamble, blocks (1,024 tokens)
PRE = list(range(PRE_N))
SHA = hashlib.sha256
ZERO = b"\0" * 32
MASTER = b"cgai-demo-master-key"
_PK = {}


def pkey(tenant):
    k = _PK.get(tenant)
    if k is None:
        k = _PK[tenant] = hmac.new(MASTER, tenant.encode(), SHA).digest()
    return k


class Req:
    __slots__ = ("t", "tenant", "kind", "blocks", "out", "insert", "pub", "tag")

    def __init__(self, t, tenant, kind, blocks, out, insert, pub, tag=None):
        self.t, self.tenant, self.kind, self.blocks = t, tenant, kind, blocks
        self.out, self.insert, self.pub, self.tag = out, insert, pub, tag


# ----------------------------------------------------------------------------------------------
# Tier B watchers
# ----------------------------------------------------------------------------------------------
class NoWatcher:
    def observe(self, ten, t, nb, out, hb):
        return False


class LegacyWatcher:
    """RECONSTRUCTION of the v1 heuristic from its description (original file unavailable):
    window of 20 requests; score = 0.4*frac(1-token) + 0.3*frac(modal block count)
    + 0.3*hit/miss flip rate. Static, window-level, no time axis, no notion of template stability."""
    def __init__(self):
        self.win = defaultdict(list)

    def observe(self, ten, t, nb, out, hb):
        w = self.win[ten]
        w.append((out <= 1, nb, hb / nb >= 0.5))
        if len(w) > 20:
            w.pop(0)
        if len(w) < 8:
            return False
        n = len(w)
        f1 = sum(x[0] for x in w) / n
        lens = [x[1] for x in w]
        f2 = lens.count(max(set(lens), key=lens.count)) / n
        st = [x[2] for x in w]
        fl = sum(1 for a, b in zip(st, st[1:]) if a != b) / (n - 1)
        return 0.4 * f1 + 0.3 * f2 + 0.3 * fl >= 0.65


class RefinedWatcher:
    """Leaky-bucket probe score with a template-stability gate.
    Evidence (+1) only when a request is a 1-token, long-prompt, *cold-prefix* request from a tenant whose
    EWMA hit fraction is low. A batch classifier re-uses one template, so its hit fraction stays high and it
    never accrues evidence. A prober must use a distinct prefix per candidate, so its hit fraction is low.
    Score leaks at LEAK points/s, so only a sustained probing rate above LEAK/s can reach THETA.
    Parameters were fixed a priori and not tuned on the outputs."""
    ALPHA, PRIOR, GATE, COLD = 0.15, 0.75, 0.6, 0.5
    MIN_BLOCKS, LEAK, THETA, THETA_LOW = 32, 0.5, 8.0, 3.0

    def __init__(self):
        self.st = {}

    def observe(self, ten, t, nb, out, hb):
        s = self.st.get(ten)
        if s is None:
            s = self.st[ten] = [0.0, t, self.PRIOR, False]
        score, last, eh, flag = s
        score = max(0.0, score - self.LEAK * (t - last))
        hf = hb / nb
        if out <= 1 and nb >= self.MIN_BLOCKS and hf < self.COLD and eh < self.GATE:
            score += 1.0
        eh += self.ALPHA * (hf - eh)
        flag = score >= (self.THETA_LOW if flag else self.THETA)
        s[0], s[1], s[2], s[3] = score, t, eh, flag
        return flag


WATCHERS = {"none": NoWatcher, "legacy": LegacyWatcher, "refined": RefinedWatcher}


# ----------------------------------------------------------------------------------------------
# Serving engine: single FIFO prefill queue + LRU block cache (decode assumed not to block prefill)
# ----------------------------------------------------------------------------------------------
class Server:
    def __init__(self, cfg, cap, seed):
        self.cfg, self.cap = cfg, cap
        self.cache = OrderedDict()
        self.free = 0.0
        self.rn = random.Random(seed * 31 + 1)   # service noise: same stream across configs (paired)
        self.rj = random.Random(seed * 31 + 2)   # jitter stream
        self.w = WATCHERS[cfg["watcher"]]()
        self.rootd = {}
        self.cpu = 0.0
        self.n = 0

    def keys(self, r):
        mode, ten, d = self.cfg["salt"], r.tenant, ZERO
        if mode == "root":
            d = self.rootd.get(ten)
            if d is None:
                d = self.rootd[ten] = hmac.new(pkey(ten), b"root", SHA).digest()
        out = []
        ap = out.append
        for i, c in enumerate(r.blocks):
            if mode == "boundary" and i == r.pub:
                d = hmac.new(pkey(ten), d, SHA).digest()   # salt everything after the public prefix
            d = SHA(d + c.to_bytes(8, "little")).digest()
            ap(d)
        return out

    def handle(self, r):
        ks = self.keys(r)
        start = max(r.t, self.free)
        cache, h = self.cache, 0
        for k in ks:
            if k in cache:
                h += 1
            else:
                break
        nb = len(ks)
        service = (T_FIXED + T_PER_TOK * (nb - h) * BLOCK) * self.rn.lognormvariate(0, SIGMA)
        self.free = start + service / 1000.0
        if r.insert:                               # insert leaf-first so LRU evicts leaves before roots
            for k in reversed(ks):
                if k in cache:
                    cache.move_to_end(k)
                else:
                    cache[k] = 1
            while len(cache) > self.cap:
                cache.popitem(last=False)
        c0 = time.perf_counter()
        flagged = self.w.observe(r.tenant, r.t, nb, r.out, h)
        self.cpu += time.perf_counter() - c0
        self.n += 1
        ttft = (start - r.t) * 1000.0 + service
        jit = self.rj.uniform(0, self.cfg["jitter"]) if flagged else 0.0
        return ttft + jit, flagged, jit


# ----------------------------------------------------------------------------------------------
# Workload (all parameters are assumptions; rates chosen for ~0.6 prefill utilisation)
# ----------------------------------------------------------------------------------------------
def gen_benign(rng, duration, ids):
    R = []

    def add(t, ten, blocks, out, pub):
        R.append(Req(t, ten, "benign", blocks, out, True, pub))

    for i in range(8):                                     # chat: multi-turn sessions, long outputs
        ten, t, hist = f"chat-{i}", rng.random() * 2, None
        while True:
            t += rng.expovariate(0.7)
            if t >= duration:
                break
            if hist is None or rng.random() < 0.25 or len(hist) > 200:
                hist = ids(rng.randint(3, 10))
            else:
                hist = hist + ids(rng.randint(12, 25))
            add(t, ten, PRE + hist, rng.randint(80, 300), PRE_N)
    for i in range(3):                                     # RAG: Zipf over 20 documents
        ten, t = f"rag-{i}", rng.random() * 2
        docs, w = [ids(90) for _ in range(20)], [1 / (k + 1) for k in range(20)]
        while True:
            t += rng.expovariate(0.8)
            if t >= duration:
                break
            d = rng.choices(range(20), w)[0]
            add(t, ten, PRE + docs[d] + ids(rng.randint(2, 4)), rng.randint(100, 250), PRE_N)

    def classifier(ten, times, tpl, item, pub):             # legitimate 1-token batch classifier, 1% retries
        for t in times:
            b = tpl + item()
            add(t, ten, b, 1, pub)
            if rng.random() < 0.01:
                add(t + rng.uniform(0.3, 1.0), ten, b, 1, pub)

    def poisson(rate):
        t, out = rng.random(), []
        while True:
            t += rng.expovariate(rate)
            if t >= duration:
                return out
            out.append(t)

    classifier("cls-0", poisson(3.0), PRE + ids(48), lambda: ids(4), PRE_N)
    bt, t0 = [], rng.random() * 25                          # cls-1: 5 s bursts at 8 req/s every 25 s
    while t0 < duration:
        t = t0
        while True:
            t += rng.expovariate(8.0)
            if t >= t0 + 5 or t >= duration:
                break
            bt.append(t)
        t0 += 25
    classifier("cls-1", bt, PRE + ids(48), lambda: ids(4), PRE_N)
    classifier("cls-2", poisson(0.4), ids(100), lambda: ids(rng.randint(2, 8)), 0)   # no public preamble
    ep, k, t = [ids(40) for _ in range(20)], 0, rng.random()   # eval harness: 20 fixed prompts, cycled
    while t < duration:
        add(t, "eval-0", ep[k % 20], 1, 0)
        k += 1
        t += rng.uniform(0.8, 1.2)
    R.sort(key=lambda r: r.t)
    return R


def make_ids():
    c = itertools.count(1000)
    return lambda n: [next(c) for _ in range(n)]


CFGS = {}


def mk(name, salt, watcher, jitter):
    CFGS[name] = dict(salt=salt, watcher=watcher, jitter=jitter)


# ----------------------------------------------------------------------------------------------
# Experiment 1: benign traffic -> TTFT overhead and false-positive rate
# ----------------------------------------------------------------------------------------------
def e1_task(a):
    cap, seed, cname, dur = a
    rng = random.Random(seed)
    reqs = gen_benign(rng, dur, make_ids())
    srv = Server(CFGS[cname], cap, seed)
    tt, cl, fl, jt, ts, ten_f = [], [], [], [], [], set()
    for r in reqs:
        t, f, j = srv.handle(r)
        tt.append(t); cl.append(r.tenant.split("-")[0]); fl.append(f); jt.append(j); ts.append(r.t)
        if f:
            ten_f.add(r.tenant)
    return (cap, seed, cname), dict(tt=tt, cl=cl, fl=fl, jt=jt, ts=ts, ten=sorted(ten_f), cpu=srv.cpu, n=srv.n)


def pct(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(p * len(x)))]


def summarize(base, r, discard):
    """Metrics over requests arriving after `discard` seconds (cold-cache transient excluded; it is still simulated)."""
    ix = [i for i, t in enumerate(r["ts"]) if t >= discard]
    bt, rt = [base["tt"][i] for i in ix], [r["tt"][i] for i in ix]
    mb, mr = statistics.fmean(bt), statistics.fmean(rt)
    jit_part = statistics.fmean(r["jt"][i] for i in ix) / mb
    d = dict(mean=mr, ovh=mr / mb - 1, p99=pct(rt, .99) / pct(bt, .99) - 1,
             jit=jit_part, cache=(mr / mb - 1) - jit_part, cpu_us=1e6 * r["cpu"] / r["n"],
             sync=r["cpu"] / r["n"] * 1000 / mb, ten=r["ten"], base_mean=mb, n=len(ix))
    for c in ("chat", "rag", "cls", "eval"):
        jx = [i for i in ix if r["cl"][i] == c]
        d["fpr_" + c] = sum(r["fl"][i] for i in jx) / len(jx)
        d["ovh_" + c] = statistics.fmean(r["tt"][i] for i in jx) / statistics.fmean(base["tt"][i] for i in jx) - 1
    return d


# ----------------------------------------------------------------------------------------------
# Experiment 2: active probing (victim secret among 64 candidates)
# ----------------------------------------------------------------------------------------------
# name: (measurement rounds R, probe rate/s, mimic requests per probe)
ATTACKERS = {"naive": (1, 3.0, 0), "adaptive_min4": (4, 3.0, 0), "mimic_poison": (1, 1.5, 3), "slow_low": (1, 0.4, 0)}


WARM = 90.0


def e2_task(a):
    """One (config, trial): warm the server with benign traffic, then fork the state once per attacker variant."""
    cname, trial, cap = a
    seed = trial * 7919 + 13
    rng, ids = random.Random(seed), make_ids()
    tmax = WARM + max(64 * R_ / rate for R_, rate, _ in ATTACKERS.values()) + 5
    bg = gen_benign(rng, tmax, ids)
    cands = [ids(100) for _ in range(64)]
    secret = rng.randrange(64)
    tpl = ids(16)
    order = list(range(64)); rng.shuffle(order)
    victim = Req(WARM, "victim", "victim", PRE + cands[secret] + ids(1), 50, True, PRE_N)
    srv = Server(CFGS[cname], cap, seed)
    for r in bg:
        if r.t < WARM:
            srv.handle(r)
    res = {}
    for atk, (R_, rate, mim) in ATTACKERS.items():
        total = WARM + 64 * R_ / rate + 5
        reqs = [r for r in bg if WARM <= r.t < total] + [victim]
        t, iv = WARM + 2.0, 1.0 / rate
        for rd in range(R_):
            for c in order:
                d = iv / (mim + 1)
                for k in range(mim):                  # mimicry: templated, cache-hitting 1-token traffic
                    reqs.append(Req(t + k * d, "atk", "mimic", PRE + tpl + ids(4), 1, True, PRE_N))
                # probe is non-inserting: generous to the attacker (assumes a perfect cache-reset oracle)
                reqs.append(Req(t + mim * d, "atk", "probe", PRE + cands[c] + ids(1), 1, False, PRE_N, (c, rd)))
                t += iv
        reqs.sort(key=lambda r: r.t)
        s2 = copy.deepcopy(srv)
        meas, np_, first, bn, bj = defaultdict(list), 0, None, 0, 0
        for r in reqs:
            ttft, f, j = s2.handle(r)
            if r.kind == "probe":
                np_ += 1
                meas[r.tag[0]].append(ttft)
                if f and first is None:
                    first = np_ - 1
            elif r.kind == "mimic":
                if f and first is None:
                    first = np_
            elif r.kind == "benign":
                bn += 1; bj += f
        guess = min(meas, key=lambda c: min(meas[c]))
        res[atk] = dict(ok=guess == secret, first=first, bn=bn, bj=bj)
    return cname, res


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, a = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - a) / d), min(1.0, (c + a) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--discard", type=float, default=60)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--caches", type=int, nargs="+", default=[12000, 4000])
    ap.add_argument("--jitter-ms", type=float, default=120)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--skip-e1", action="store_true")
    a = ap.parse_args()
    J = a.jitter_ms
    mk("baseline", "none", "none", J); mk("root_salt", "root", "none", J)
    mk("boundary_salt", "boundary", "none", J); mk("bc_only", "none", "refined", J)
    mk("cgai_legacy", "boundary", "legacy", J); mk("cgai_refined", "boundary", "refined", J)

    t0 = time.time()
    us = []
    for _ in range(20000):
        c = time.perf_counter(); hmac.new(pkey("x"), ZERO, SHA).digest(); us.append(time.perf_counter() - c)
    print(f"[micro] Python HMAC-SHA256 key derivation: {1e6*statistics.median(us):.2f} us "
          f"(= {100*statistics.median(us)*1000/T_FIXED:.4f}% of the {T_FIXED} ms cached TTFT)")

    tasks = [] if a.skip_e1 else [(cap, s, c, a.duration) for cap in a.caches for s in a.seeds for c in CFGS]
    if a.procs > 1:
        from multiprocessing import Pool
        with Pool(a.procs) as p:
            res = dict(p.map(e1_task, tasks))
    else:
        res = dict(map(e1_task, tasks))

    order = ["root_salt", "boundary_salt", "cgai_legacy", "cgai_refined"]
    print(f"\n=== EXPERIMENT 1: benign traffic, {a.duration:.0f}s, seeds {a.seeds}, jitter U(0,{J:.0f}) ms ===")
    for cap in ([] if a.skip_e1 else a.caches):
        print(f"\n--- cache = {cap} blocks ---")
        agg = defaultdict(list)
        for s in a.seeds:
            base = res[(cap, s, "baseline")]
            bs = summarize(base, base, a.discard)
            print(f"seed {s}: baseline mean TTFT {bs['mean']:.2f} ms over {bs['n']} requests (first {a.discard:.0f}s discarded)")
            for c in order:
                d = summarize(base, res[(cap, s, c)], a.discard)
                agg[c].append(d)
                print(f"  {c:14s} mean {d['mean']:7.2f} ms  overhead {100*d['ovh']:6.2f}%  "
                      f"(jitter {100*d['jit']:5.2f}% + cache/queue {100*d['cache']:5.2f}%)  p99 {100*d['p99']:6.2f}%  "
                      f"flagged tenants: {d['ten'] or '-'}")
        print("\n  MEAN over seeds [min..max]:")
        print(f"  {'config':14s} {'overhead%':>22s} {'FPR cls':>8s} {'FPR eval':>9s} {'FPR chat/rag':>13s} {'ovh cls%':>9s} {'ovh eval%':>10s} {'watcher us/req':>15s} {'+sync%':>7s}")
        for c in order:
            L = agg[c]
            ov = [100 * x["ovh"] for x in L]
            f = lambda k: statistics.fmean(x[k] for x in L)
            print(f"  {c:14s} {statistics.fmean(ov):7.2f} [{min(ov):6.2f}..{max(ov):6.2f}] "
                  f"{100*f('fpr_cls'):7.2f}% {100*f('fpr_eval'):8.2f}% {100*max(f('fpr_chat'), f('fpr_rag')):12.2f}% "
                  f"{100*f('ovh_cls'):8.2f} {100*f('ovh_eval'):9.2f} {f('cpu_us'):14.1f} {100*f('sync'):6.2f}")
        print("  FPR = fraction of that class's requests that received Tier-C jitter. '+sync%' = extra TTFT if the pure-Python watcher ran in-path.")

    cfg2 = ["bc_only", "boundary_salt", "cgai_refined"]
    mk("none_attack", "none", "none", J)
    cfg2 = ["none_attack"] + cfg2
    t2 = [(c, i, a.caches[0]) for c in cfg2 for i in range(a.trials)]
    if a.procs > 1:
        from multiprocessing import Pool
        with Pool(a.procs) as p:
            r2 = p.map(e2_task, t2)
    else:
        r2 = list(map(e2_task, t2))
    grp = defaultdict(list)
    for cn, rd in r2:
        for k, v in rd.items():
            grp[(cn, k)].append(v)
    print(f"\n=== EXPERIMENT 2: active probing, {WARM:.0f}s warm-up, {a.trials} trials per cell, cache {a.caches[0]}, chance = {100/64:.1f}% ===")
    print("attacker probes are non-inserting (generous to attacker); 'clean' = median probes sent before first jitter")
    print(f"  {'config':14s} {'attacker':14s} {'ASR':>7s} {'95% Wilson CI':>16s} {'flagged':>8s} {'clean probes':>13s} {'benign jittered':>16s}")
    for c in cfg2:
        for k in ATTACKERS:
            L = grp[(c, k)]
            n, ok = len(L), sum(x["ok"] for x in L)
            lo, hi = wilson(ok, n)
            fr = [x["first"] for x in L if x["first"] is not None]
            bj = sum(x["bj"] for x in L) / sum(x["bn"] for x in L)
            print(f"  {c:14s} {k:14s} {100*ok/n:6.1f}% [{100*lo:5.1f},{100*hi:5.1f}]% {len(fr):3d}/{n:<4d} "
                  f"{(statistics.median(fr) if fr else float('nan')):12.1f} {100*bj:15.2f}%")
    print(f"\n[done in {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

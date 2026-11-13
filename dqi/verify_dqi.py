"""Local re-implementation of the Benchmark II (DQI Reed-Solomon phase decoder) scoring rule.

Written from the published specification (arXiv:2610.23832, Sec. III.B) so that a candidate
circuit can be checked on this machine.  It is NOT the official verifier; the official one on
Kaggle remains the source of truth.  The sampling order below is chosen to match the published
procedure so that results are directly comparable for the same circuit file and salt.

Contract checked per shot (m=4095 positions, n=70 syndromes over GF(2^12)):
  * registers S0..S69 (12 qubits each) hold S_i = sum_j e_j alpha^{(i+1) j}; they must be restored
  * registers E0..E4094 (12 classical bits each) hold the X-measurement record M_j
  * every other qubit must end in |0>
  * global phase must be (-1)^{sum_j popcount(e_j & M_j)}
Score = R(eps_plus) * Q^2 * T with T = mean executed CCX/CCZ/CSWAP per shot.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import math
import multiprocessing as mp
import sys
import time

import numpy as np

import kickmix as km

M, N, B = 4095, 70, 12
T_MAX = N // 2
Q_FIELD = 1 << B
R_SET = (Q_FIELD - math.isqrt(Q_FIELD)) // 2  # 2016
SHOTS = 400 * 256
BATCH = 256
LOG2_DELTA = -128.0


def weight_distribution(m=M, t=T_MAX, b=B, r=R_SET):
    """Pr[wt(e) = k], k = 0..t: squared principal eigenvector of the DQI tridiagonal matrix."""
    q = 1 << b
    k = np.arange(t + 1, dtype=np.float64)
    off = np.sqrt(k[1:] * (m - k[1:] + 1))
    d = (q - 2 * r) / math.sqrt(r * (q - r))
    _, vecs = np.linalg.eigh(np.diag(k * d) + np.diag(off, 1) + np.diag(off, -1))
    w = np.abs(vecs[:, -1])
    w = w / np.linalg.norm(w)
    return w ** 2


def cp_upper(k, n, log2_delta=LOG2_DELTA):
    """One-sided Clopper-Pearson upper bound at confidence 1 - 2^log2_delta."""
    if k == n:
        return 1.0
    log_delta = log2_delta * math.log(2.0)
    if k == 0:
        return -math.expm1(log_delta / n)
    j = np.arange(k + 1, dtype=np.float64)
    log_comb = np.r_[0.0, np.cumsum(np.log(n - j[1:] + 1.0) - np.log(j[1:]))]

    def log_tail(p):
        terms = log_comb + j * math.log(p) + (n - j) * math.log1p(-p)
        mx = float(terms.max())
        return mx + math.log(float(np.exp(terms - mx).sum()))

    lo = k / n if log_tail(k / n) >= log_delta else 0.0
    hi = 1.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        if log_tail(mid) > log_delta:
            lo = mid
        else:
            hi = mid
    return hi


def repetition(eps):
    if eps == 0.0:
        return 1.0
    return math.log(2.0) / -math.log(0.5 + 2.0 * math.sqrt(eps))


class Tables:
    def __init__(self, b=B):
        f = km.GF2Field(b)
        self.order = (1 << b) - 1
        g = f.primitive_element
        self.antilog = np.array([f.pow(g, k) for k in range(self.order)], dtype=np.int64)
        assert len(set(self.antilog.tolist())) == self.order
        self.log = np.zeros(1 << b, dtype=np.int64)
        self.log[self.antilog] = np.arange(self.order)


def to_bits(values, b=B):
    bits = (values[:, :, None] >> np.arange(b)) & 1 == 1
    return bits.transpose(1, 2, 0).reshape(-1, len(values))


def draw_batch(tab, rng, count, wprobs, fixed_weight=None):
    """Returns (syndromes[count, N], mask[count, M], phase[count], weights[count])."""
    maxv = (1 << B) - 1
    if fixed_weight is None:
        weights = rng.choice(len(wprobs), size=count, p=wprobs)
    else:
        weights = np.full(count, fixed_weight, dtype=np.int64)
    mask = rng.integers(0, maxv, size=(count, M), endpoint=True, dtype=np.int64)
    synd = np.zeros((count, N), dtype=np.int64)
    phase = np.zeros(count, dtype=bool)
    rows = np.arange(1, N + 1, dtype=np.int64)[:, None]
    for s in np.flatnonzero(weights):
        k = int(weights[s])
        pos = rng.choice(M, size=k, replace=False)
        vals = rng.integers(1, maxv, size=k, endpoint=True, dtype=np.int64)
        ex = (tab.log[vals][None, :] + rows * pos[None, :]) % tab.order
        synd[s] = np.bitwise_xor.reduce(tab.antilog[ex], axis=1)
        phase[s] = sum(int(x).bit_count() for x in vals & mask[s, pos]) & 1 == 1
    return synd, mask, phase, weights


def circuit_digest(circuit):
    buf = io.BytesIO()
    circuit.to_file(buf, format='kmb')
    return hashlib.sha256(buf.getvalue()).hexdigest()


def registers_of(circuit):
    regs = circuit.register_data
    s_names = [f'S{i}' for i in range(N)]
    e_names = [f'E{j}' for j in range(M)]
    if set(regs) != set(s_names) | set(e_names) or len(regs) != circuit.num_registers:
        raise ValueError('registers must be exactly S0..S69 and E0..E4094')
    sq, eb = [], []
    for nm in s_names:
        reg = list(regs[nm])
        if len(reg) != B or not all(isinstance(x, km.q) for x in reg):
            raise ValueError(f'bad syndrome register {nm}')
        sq.extend(reg)
    for nm in e_names:
        reg = list(regs[nm])
        if len(reg) != B or not all(isinstance(x, km.b) for x in reg):
            raise ValueError(f'bad mask register {nm}')
        eb.extend(reg)
    if len(set(sq)) != len(sq) or len(set(eb)) != len(eb):
        raise ValueError('registers overlap')
    sset = set(sq)
    anc = km.array([km.q(k) for k in range(circuit.num_qubits) if km.q(k) not in sset])
    return km.array(sq), km.array(eb), anc


_CTX = {}


def _run(task):
    start, rng_seed, sim_seed = task
    c = _CTX
    rng = np.random.default_rng(rng_seed)
    synd, mask, phase, weights = draw_batch(c['tab'], rng, c['batch'], c['wprobs'], c['fixed'])
    sim = km.Simulator(batch_size=c['batch'], debug_print=False, counting_operations=True, seed=sim_seed)
    sim.use_same_registers_as(c['circuit'])
    sbits = to_bits(synd)
    sim.write_across_shots(c['sq'], sbits)
    sim.write_across_shots(c['eb'], to_bits(mask))
    sim.do(c['circuit'])
    flipped = sim.read_phase_flipped_across_shots()
    diff = sim.read_across_shots(c['sq']) ^ sbits
    junk = sim.read_across_shots(c['anc']) if len(c['anc']) else np.zeros((0, c['batch']), dtype=bool)
    val_err = diff.any(axis=0) | junk.any(axis=0)
    ph_err = (flipped ^ phase) & ~val_err
    st = sim.operation_statistics()
    bad = np.flatnonzero(val_err | ph_err)
    return (int(val_err.sum()), int(ph_err.sum()), int(st.num_toffoli_like),
            [(start + int(i), int(weights[i]), bool(val_err[i])) for i in bad])


def verify(circuit, *, shots=SHOTS, batch=BATCH, salt='', seed=None, workers=3, fixed_weight=None):
    stats = circuit.operation_statistics()
    if 'Z_POW' in stats:
        raise ValueError('Z_POW gates are not allowed')
    digest = circuit_digest(circuit)
    sq, eb, anc = registers_of(circuit)
    if seed is None:
        base = int(digest, 16) if not salt else int(
            hashlib.sha256(f'{digest}{salt}'.encode()).hexdigest(), 16)
    else:
        base = int(seed)
    starts = range(0, shots, batch)
    seeds = np.random.SeedSequence(base, pool_size=8).generate_state(
        8 * len(starts), dtype=np.uint64).reshape(-1, 2, 4)
    tasks = [(s, int.from_bytes(seeds[i, 0], 'little'), int.from_bytes(seeds[i, 1], 'little'))
             for i, s in enumerate(starts)]
    _CTX.update(circuit=circuit, sq=sq, eb=eb, anc=anc, tab=Tables(), batch=batch,
                wprobs=weight_distribution(), fixed=fixed_weight)
    if workers <= 1:
        res = [_run(t) for t in tasks]
    else:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
            res = list(pool.map(_run, tasks))
    ve = sum(r[0] for r in res)
    pe = sum(r[1] for r in res)
    tof = sum(r[2] for r in res) / shots
    fails = [f for r in res for f in r[3]]
    k = ve + pe
    eps = cp_upper(k, shots)
    out = dict(num_qubits=circuit.num_qubits, toffoli_mean=tof,
               toffoli_static=stats.num_toffoli_like, shots=shots, value_errors=ve,
               phase_errors=pe, eps_plus=eps, digest=digest, salt=salt,
               failed_shots=fails[:50])
    if eps >= 1 / 16:
        out.update(status='FAIL', score=None)
    else:
        rep = repetition(eps)
        out.update(status='PASS', repetition=rep, score=rep * circuit.num_qubits ** 2 * tof)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('circuit')
    ap.add_argument('--shots', type=int, default=SHOTS)
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--salt', default='')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--weight', type=int, default=None, help='stress test: fix the error weight')
    a = ap.parse_args()
    t0 = time.time()
    circ = km.Circuit.from_file(a.circuit)
    out = verify(circ, shots=a.shots, batch=a.batch, salt=a.salt, seed=a.seed, workers=a.workers,
                 fixed_weight=a.weight)
    out['seconds'] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=1))
    sys.exit(0 if out['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()

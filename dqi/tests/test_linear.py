"""Tests for circ/linear.py.  Run from the dqi directory: python3 -m pytest -q tests/test_linear.py"""
import random
import time

import numpy as np
import pytest

import kickmix as km
from circ import linear as L
from model import gf, reference


def rand_matrix(rng, m, n):
    return [rng.getrandbits(n) for _ in range(m)]


def rand_invertible(rng, n):
    while True:
        A = rand_matrix(rng, n, n)
        if L.gf2_rank(A) == n:
            return A


def simulate(circuit, qubits, X):
    """Runs `circuit` on the columns of X; returns (output bits, executed Toffoli count, phase flips)."""
    sim = km.Simulator(batch_size=X.shape[1], counting_operations=True, seed=7)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots(km.array(qubits), X)
    sim.do(circuit)
    return (sim.read_across_shots(km.array(qubits)), sim.operation_statistics().num_toffoli_like,
            sim.read_phase_flipped_across_shots())


def expected(A, X, ncols=None):
    return (L.to_numpy(A, ncols).astype(np.int64) @ X.astype(np.int64)) % 2 == 1


# ---------------------------------------------------------------- GF(2) helpers

def test_gf2_helpers():
    rng = random.Random(1)
    for m, k, n in [(1, 1, 1), (5, 9, 3), (17, 8, 40), (64, 65, 63), (100, 130, 70)]:
        A, B = rand_matrix(rng, m, k), rand_matrix(rng, k, n)
        ref = (L.to_numpy(A, k).astype(np.int64) @ L.to_numpy(B, n)) % 2
        assert L.gf2_matmul(A, B) == L.from_numpy(ref)
        assert L.from_numpy(L.to_numpy(A, k)) == A
        assert L.gf2_transpose(L.gf2_transpose(A, k), m) == A
        assert (L.to_numpy(L.gf2_transpose(A, k), m) == L.to_numpy(A, k).T).all()
        x = rng.getrandbits(n)
        assert L.gf2_apply(A, L.gf2_apply(B, x)) == L.gf2_apply(L.gf2_matmul(A, B), x)
    for n in (1, 2, 7, 33, 100):
        A = rand_invertible(rng, n)
        Ai = L.gf2_inverse(A)
        assert L.gf2_matmul(A, Ai) == L.gf2_identity(n) == L.gf2_matmul(Ai, A)
    for n, r in [(6, 0), (20, 7), (90, 31), (90, 90)]:           # P * diag(1^r, 0) * Q has rank r
        A = L.gf2_matmul(rand_invertible(rng, n), rand_invertible(rng, n)[:r] + [0] * (n - r))
        assert L.gf2_rank(A) == r
    assert L.gf2_rank(L.gf2_identity(50)) == 50 and L.gf2_rank([3, 3, 3]) == 1 and L.gf2_rank([]) == 0
    with pytest.raises(ValueError):
        L.gf2_inverse([3, 5, 6])


def test_gf2_840_is_fast():
    rng = random.Random(2)
    A = rand_invertible(rng, 840)
    t0 = time.time()
    Ai = L.gf2_inverse(A)
    assert L.gf2_matmul(A, Ai) == L.gf2_identity(840) and L.gf2_rank(Ai) == 840
    assert time.time() - t0 < 10


# ---------------------------------------------------------------- synthesis

def test_synth_program_matrix():
    rng = random.Random(3)
    for n in range(1, 41):
        A = rand_invertible(rng, n)
        for sec in (None, 1, 2, 5):
            ops = L.synth_linear(A, sec)
            assert L.program_matrix(ops, n) == A
            assert L.program_matrix(ops[::-1], n) == L.gf2_inverse(A)
    assert L.synth_linear(L.gf2_identity(30)) == []
    perm = list(range(30))
    rng.shuffle(perm)
    P = [1 << p for p in perm]
    ops = L.synth_linear(P)
    assert L.program_matrix(ops, 30) == P and all(k == L.SWAP for k, _, _ in ops) and len(ops) < 30
    with pytest.raises(ValueError):
        L.synth_linear([3, 5, 6])


def test_emit_linear_sizes_1_to_64():
    rng = random.Random(4)
    b = km.CircuitBuilder()
    regs, mats = [], []
    for n in range(1, 65):
        regs.append(list(b.create_quantum_register(n, name=f'r{n}')))
        mats.append(rand_invertible(rng, n))
        ops = L.emit_linear(b, regs[-1], mats[-1])
        assert all(k in (L.CX, L.SWAP) for k, _, _ in ops)
    perm = list(range(40))                                       # one batch of chained SWAPs
    rng.shuffle(perm)
    regs.append(list(b.create_quantum_register(40, name='perm')))
    mats.append([1 << p for p in perm])
    assert all(k == L.SWAP for k, _, _ in L.emit_linear(b, regs[-1], mats[-1]))
    with pytest.raises(ValueError):
        L.emit_linear(b, regs[-1], mats[0])
    circuit = b.finish_circuit()
    allq = [q for r in regs for q in r]
    assert circuit.num_qubits == len(allq)                       # no ancilla
    assert circuit.operation_statistics().num_toffoli_like == 0
    X = np.random.default_rng(5).integers(0, 2, size=(len(allq), 128)).astype(bool)
    Y, toffoli, flipped = simulate(circuit, allq, X)
    assert toffoli == 0 and not flipped.any()
    off = 0
    for A in mats:
        n = len(A)
        assert (Y[off:off + n] == expected(A, X[off:off + n])).all(), n
        off += n


def test_emit_linear_840():
    rng = random.Random(6)
    A = rand_invertible(rng, 840)
    b = km.CircuitBuilder()
    q = list(b.create_quantum_register(840, name='x'))
    t0 = time.time()
    ops = L.emit_linear(b, q, A)
    assert time.time() - t0 < 20
    assert len(ops) < 200_000                                    # plain Gauss-Jordan needs ~350k
    circuit = b.finish_circuit()
    assert circuit.num_qubits == 840
    assert circuit.operation_statistics().num_instructions == len(ops)
    X = np.random.default_rng(7).integers(0, 2, size=(840, 128)).astype(bool)
    Y, toffoli, flipped = simulate(circuit, q, X)
    assert toffoli == 0 and not flipped.any()
    assert (Y == expected(A, X)).all()
    b = km.CircuitBuilder()                                      # the reversed program is the inverse
    q = list(b.create_quantum_register(840, name='x'))
    L.emit_program(b, q, ops, inverse=True)
    Z, toffoli, _ = simulate(b.finish_circuit(), q, Y)
    assert toffoli == 0 and (Z == X).all()


def test_emit_xor_linear():
    rng = random.Random(8)
    m, n = 37, 53
    A = rand_matrix(rng, m, n)
    b = km.CircuitBuilder()
    src = list(b.create_quantum_register(n, name='src'))
    dst = list(b.create_quantum_register(m, name='dst'))
    assert L.emit_xor_linear(b, src, dst, A) == sum(r.bit_count() for r in A)
    assert L.emit_xor_linear(b, src, dst, [0] * m) == 0
    circuit = b.finish_circuit()
    assert circuit.num_qubits == m + n
    X = np.random.default_rng(9).integers(0, 2, size=(m + n, 64)).astype(bool)
    Y, toffoli, flipped = simulate(circuit, src + dst, X)
    assert toffoli == 0 and not flipped.any()
    assert (Y[:n] == X[:n]).all() and (Y[n:] == X[n:] ^ expected(A, X[:n], n)).all()
    with pytest.raises(ValueError):
        L.emit_xor_linear(b, src, dst, [1 << n] * m)


# ---------------------------------------------------------------- fields and matrix builders

def test_field_K():
    rng = random.Random(10)
    u = 1 << 12
    assert L.f_trace(L.K_C) == 1 and all(L.f_trace(c) == 0 for c in range(L.K_C))
    assert len(L.TRACE1) == 2048
    assert L.k_mul(u, u) == u ^ L.K_C
    for _ in range(200):
        a, b, c = (rng.getrandbits(24) for _ in range(3))
        assert L.k_mul(a, b) == L.k_mul(b, a)
        assert L.k_mul(L.k_mul(a, b), c) == L.k_mul(a, L.k_mul(b, c))
        assert L.k_mul(a, b ^ c) == L.k_mul(a, b) ^ L.k_mul(a, c)
        assert L.k_mul(a & 4095, b & 4095) == gf.mul(a & 4095, b & 4095)      # F sits in K as hi = 0
        if a:
            assert L.k_mul(a, L.k_inv(a)) == 1
            assert L.k_pow(a, (1 << 24) - 1) == 1
        assert L.k_conj(a) == L.k_pow(a, 1 << 12)
        assert L.k_pow(a, 5) == L.k_mul(a, L.k_mul(L.k_mul(a, a), L.k_mul(a, a)))
    assert L.k_pow(0, 0) == 1 and L.k_eval([], 77) == 0


def test_matrix_builders():
    rng = random.Random(11)
    field = km.GF2Field(12)
    for _ in range(20):
        c, x = rng.getrandbits(12), rng.getrandbits(12)
        assert L.gf2_apply(L.f_mul_matrix(c), x) == gf.mul(c, x) == field.mul(c, x)
        kc, kx = rng.getrandbits(24), rng.getrandbits(24)
        assert L.gf2_apply(L.k_mul_matrix(kc), kx) == L.k_mul(kc, kx)
        for e in (0, 1, 5, 11, 12):
            assert L.gf2_apply(L.f_frobenius_matrix(e), x) == gf.power(x, 1 << (e % 12))
        for e in (0, 1, 7, 12, 23, 24):
            assert L.gf2_apply(L.k_frobenius_matrix(e), kx) == L.k_pow(kx, 1 << (e % 24))
    assert L.k_frobenius_matrix(12) == L.map_matrix(L.k_conj, 24, 24)
    assert L.f_frobenius_matrix(12) == L.gf2_identity(12) == L.f_mul_matrix(1)
    # block_diag, square and rectangular blocks
    blocks = [L.f_mul_matrix(7), L.k_mul_matrix(123456), rand_matrix(rng, 3, 5)]
    D = L.block_diag(blocks, [12, 24, 5])
    assert len(D) == 39 and L.block_diag(blocks[:2]) == D[:36]
    xs = [rng.getrandbits(12), rng.getrandbits(24), rng.getrandbits(5)]
    y = L.gf2_apply(D, xs[0] | xs[1] << 12 | xs[2] << 36)
    assert y == (L.gf2_apply(blocks[0], xs[0]) | L.gf2_apply(blocks[1], xs[1]) << 12
                 | L.gf2_apply(blocks[2], xs[2]) << 36)
    # F-block matrices and evaluation of an F-polynomial at a point of K
    grid = [[rng.getrandbits(12) for _ in range(4)] for _ in range(3)]
    grid[1][2] = 0
    G = L.f_block_matrix(grid)
    v = [rng.getrandbits(12) for _ in range(4)]
    out = L.gf2_apply(G, sum(c << 12 * j for j, c in enumerate(v)))
    for i, row in enumerate(grid):
        acc = 0
        for c, x in zip(row, v):
            acc ^= gf.mul(c, x)
        assert (out >> 12 * i) & 4095 == acc
    for n in (1, 2, 9, 70):
        xi = rng.getrandbits(24)
        E = L.k_eval_matrix(xi, n)
        assert len(E) == 24 and all(r >> 12 * n == 0 for r in E)
        p = [rng.getrandbits(12) for _ in range(n)]
        assert L.gf2_apply(E, sum(c << 12 * j for j, c in enumerate(p))) == L.k_eval(p, xi)


# ---------------------------------------------------------------- Stage 0 node transform

def test_node_constants():
    cs, xis, M = L.node_quadratics(), L.node_points(), L.node_modulus()
    assert cs == L.TRACE1[:35] and cs[0] == L.K_C and len(set(cs)) == 35
    for c in cs:                                                   # m_k has no root in F
        assert all(gf.mul(a, a) ^ a ^ c for a in range(4096))
    assert xis[0] == 1 << 12
    for c, xi in zip(cs, xis):
        assert xi >> 12 == 1 and xi & 1 == 0                       # lo + u, smaller of {xi, xi + 1}
        assert L.k_mul(xi, xi) ^ xi ^ c == 0
        assert L.k_mul(xi ^ 1, xi ^ 1) ^ xi ^ 1 ^ c == 0 and L.k_conj(xi) == xi ^ 1
        assert L.k_eval(M, xi) == 0
    assert len(M) == 71 and M[70] == 1
    assert all(gf.peval(M, x) for x in range(4096))                # no root among the locators


def syndromes(err):
    """S_i = sum_l Y_l X_l^(i+1), i = 0..69, for err = {position j: value}, X_j = g^j."""
    S = [0] * 70
    for j, y in err.items():
        for i in range(70):
            S[i] ^= gf.mul(y, gf.power(gf.EXP[j], i + 1))
    return S


def test_node_transform_against_reference():
    rng = random.Random(12)
    T, M, xis = L.node_transform(), L.node_modulus(), L.node_points()
    assert len(T) == 840 and all(r >> 840 == 0 for r in T) and L.gf2_rank(T) == 840
    for w in list(range(36)) + [35, 35, 1, 2]:
        err = {j: rng.randrange(1, 4096) for j in rng.sample(range(4095), w)}
        S = syndromes(err)
        nodes = L.gf2_apply(T, sum(s << 12 * i for i, s in enumerate(S)))
        Q, R, _ = reference.decode(S, M)
        assert gf.pdeg(Q) == w and all(gf.peval(Q, gf.EXP[j]) == 0 for j in err)
        N = reference.node_poly(S, M)
        for k, xi in enumerate(xis):
            val = (nodes >> 24 * k) & 0xFFFFFF
            assert val == L.k_eval(N, xi)
            assert val == L.k_mul(L.k_eval(R, xi), L.k_inv(L.k_eval(Q, xi)))
            direct = 0                                             # H(xi) = sum_l Yt_l / (xi - X_l)
            for j, y in err.items():
                x = gf.EXP[j]
                direct ^= L.k_mul(gf.mul(y, gf.mul(x, gf.peval(M, x))), L.k_inv(xi ^ x))
            assert val == direct


def test_node_transform_circuit():
    T = L.node_transform()
    b = km.CircuitBuilder()
    q = [x for i in range(70) for x in b.create_quantum_register(12, name=f'S{i}')]
    L.emit_node_transform(b, q)
    fwd = b.finish_circuit()
    assert fwd.num_qubits == 840 and fwd.operation_statistics().num_toffoli_like == 0
    assert fwd.operation_statistics().num_instructions == len(L.node_program())
    X = np.random.default_rng(13).integers(0, 2, size=(840, 128)).astype(bool)
    Y, toffoli, flipped = simulate(fwd, q, X)
    assert toffoli == 0 and not flipped.any() and (Y == expected(T, X)).all()
    b = km.CircuitBuilder()
    q = [x for i in range(70) for x in b.create_quantum_register(12, name=f'S{i}')]
    L.emit_node_transform_inverse(b, q)
    inv = b.finish_circuit()
    assert inv.num_qubits == 840
    Z, toffoli, flipped = simulate(inv, q, Y)
    assert toffoli == 0 and not flipped.any() and (Z == X).all()
    assert (simulate(inv, q, X)[0] == expected(L.gf2_inverse(T), X)).all()


# ---------------------------------------------------------------- reviewer checks (independent code paths)
# Everything below avoids model.gf and the K helpers of circ.linear: F arithmetic goes through
# kickmix.GF2Field(12), K = F[u]/(u^2 + u + C) is re-implemented here, programs are interpreted with numpy.

FK = km.GF2Field(12)
fmul = FK.mul


def ref_trace(a):
    t = 0
    for _ in range(12):
        t, a = t ^ a, FK.square(a)
    return t


REF_C = next(a for a in range(4096) if ref_trace(a))


def ref_kmul(a, b):
    a0, a1, b0, b1 = a & 4095, a >> 12, b & 4095, b >> 12
    c0, c1, c2 = fmul(a0, b0), fmul(a0, b1) ^ fmul(a1, b0), fmul(a1, b1)     # c0 + c1 u + c2 u^2
    return (c0 ^ fmul(REF_C, c2)) | ((c1 ^ c2) << 12)


def ref_kpow(a, e):
    r = 1
    while e:
        if e & 1:
            r = ref_kmul(r, a)
        a, e = ref_kmul(a, a), e >> 1
    return r


def run_program(ops, X):
    """Numpy interpreter of a program on the rows of the bool array X (independent of program_matrix)."""
    X = X.copy()
    for kind, a, b in ops:
        if kind == L.SWAP:
            X[[a, b]] = X[[b, a]]
        else:
            X[b] ^= X[a]
    return X


def test_synth_exhaustive_up_to_4x4():
    """Every n x n matrix, n <= 4: invertible ones are synthesised exactly, singular ones are rejected."""
    done = 0
    for n in range(1, 5):
        codes = np.arange(1 << n * n)
        mats = ((codes[:, None] >> np.arange(n * n)) & 1).reshape(-1, n, n)
        odd_det = np.rint(np.linalg.det(mats.astype(float))).astype(np.int64) % 2 == 1
        eye = np.eye(n, dtype=bool)
        for code in codes:
            A = [int(code >> n * i) & ((1 << n) - 1) for i in range(n)]
            for sec in ((None, 1, 2, 3) if n < 4 else (None, 3)):
                if not odd_det[code]:
                    with pytest.raises(ValueError):
                        L.synth_linear(A, sec)
                    continue
                ops = L.synth_linear(A, sec)
                assert all(a != b and 0 <= a < n and 0 <= b < n for _, a, b in ops)
                assert (run_program(ops, eye) == mats[code].astype(bool)).all(), (A, sec)
                done += 1
    assert done == 4 * (1 + 6 + 168) + 2 * 20160                  # |GL(n, 2)| = 1, 6, 168, 20160


def test_synth_structured_and_swap_heavy():
    rng = random.Random(20)
    n = 61
    full = (1 << n) - 1
    perm = list(range(n))
    rng.shuffle(perm)
    tri = [rng.getrandbits(i) | 1 << i for i in range(n)]
    special = [L.gf2_identity(n), [(2 << i) - 1 for i in range(n)], [full ^ ((1 << i) - 1) for i in range(n)],
               [1 << (n - 1 - i) for i in range(n)], [1 << ((i + 1) % n) for i in range(n)],
               [full ^ (1 << i) for i in range(n - 1)] + [full],
               L.gf2_matmul(tri, [1 << p for p in perm]), L.gf2_matmul([1 << p for p in perm], tri)]
    for _ in range(150):                                           # permuted sparse triangular products
        m = rng.randrange(2, 30)
        lo = [sum((rng.random() < .1) << j for j in range(i)) | 1 << i for i in range(m)]
        up = L.gf2_transpose([sum((rng.random() < .1) << j for j in range(i)) | 1 << i for i in range(m)])
        p1, p2 = list(range(m)), list(range(m))
        rng.shuffle(p1), rng.shuffle(p2)
        special.append(L.gf2_matmul([1 << p for p in p1], L.gf2_matmul(L.gf2_matmul(lo, up), [1 << p for p in p2])))
    X = np.random.default_rng(21).integers(0, 2, size=(n, 80)).astype(bool)
    for A in special:
        m = len(A)
        want = expected(A, X[:m])
        for sec in (None, 1, 3, 8, 200):
            ops = L.synth_linear(A, sec)
            assert (run_program(ops, X[:m]) == want).all()
            assert (run_program(ops[::-1], want) == X[:m]).all()
    assert L.synth_linear([]) == [] and L.synth_linear([1]) == []


def test_malformed_inputs_raise_value_error():
    for bad in ([5, 2], [1, 6], [1 | 1 << 70, 2], [2]):           # bits outside the n x n square
        with pytest.raises(ValueError):
            L.synth_linear(bad)
        with pytest.raises(ValueError):
            L.gf2_inverse(bad)
    for bad in ([0], [3, 3], [0, 0, 0], [6, 5, 3]):
        for sec in (None, 1, 2):
            with pytest.raises(ValueError):
                L.synth_linear(bad, sec)
        with pytest.raises(ValueError):
            L.gf2_inverse(bad)
    assert L.gf2_inverse([]) == [] and L.gf2_matmul([], []) == [] and L.gf2_transpose([0, 0], 0) == []
    assert L.gf2_transpose([], 3) == [0, 0, 0]
    assert L.k_eval_matrix(4096, 0) == [0] * 24                    # no coefficient register: 24 x 0
    b = km.CircuitBuilder()
    q = list(b.create_quantum_register(841, name='x'))
    for emit in (L.emit_node_transform, L.emit_node_transform_inverse):
        for qs in (q, q[:839]):
            with pytest.raises(ValueError):
                emit(b, qs)
    with pytest.raises(ValueError):                                # repeated wire
        L.emit_linear(b, [q[0], q[1], q[0]], [1, 3, 4])
    with pytest.raises(ValueError):                                # dst overlaps src
        L.emit_xor_linear(b, q[:3], q[2:4], [1, 2])
    assert b.finish_circuit().operation_statistics().num_instructions == 0
    with pytest.raises(ZeroDivisionError):
        L.k_inv(0)


def test_emit_on_shuffled_qubits_and_under_condition():
    """Bit i lives on qubits[i] for any qubit order; km.array arguments work; a pushed classical
    condition gates the whole map; bystander qubits are untouched; no Toffoli, no phase, no ancilla."""
    rng = random.Random(22)
    b = km.CircuitBuilder()
    reg = b.create_quantum_register(60, name='x')
    other = b.create_quantum_register(20, name='o')
    cond = b.alloc_bits(1)
    order = list(range(60))
    rng.shuffle(order)
    sub = km.array([reg[i] for i in order[:41]])
    A, B, C = rand_invertible(rng, 41), rand_invertible(rng, 20), rand_matrix(rng, 20, 41)
    L.emit_linear(b, sub, A)
    b.push_condition(cond[0])
    L.emit_linear(b, other, B)
    b.pop_condition()
    L.emit_xor_linear(b, sub, other, C)
    circuit = b.finish_circuit()
    assert circuit.num_qubits == 80 and circuit.operation_statistics().num_toffoli_like == 0
    X = np.random.default_rng(23).integers(0, 2, size=(80, 200)).astype(bool)
    cbit = np.random.default_rng(24).integers(0, 2, size=(1, 200)).astype(bool)
    assert cbit.any() and not cbit.all()
    sim = km.Simulator(batch_size=200, counting_operations=True, seed=1)
    sim.use_same_registers_as(circuit)
    allq = km.array(list(reg) + list(other))
    sim.write_across_shots(allq, X)
    sim.write_across_shots(km.array(list(cond)), cbit)
    sim.do(circuit)
    Y = sim.read_across_shots(allq)
    idx = np.array(order[:41])
    want = X.copy()
    want[idx] = expected(A, X[idx])
    want[60:] = np.where(cbit, expected(B, X[60:]), X[60:]) ^ expected(C, want[idx], 41)
    assert (Y == want).all()
    assert sim.operation_statistics().num_toffoli_like == 0 and not sim.read_phase_flipped_across_shots().any()


def test_K_against_kickmix_field():
    rng = random.Random(25)
    assert REF_C == L.K_C == 2048
    assert L.TRACE1 == [a for a in range(4096) if ref_trace(a)] == list(range(2048, 4096))
    for i in range(24):                                            # all basis pairs: proof by bilinearity
        for j in range(24):
            assert L.k_mul(1 << i, 1 << j) == ref_kmul(1 << i, 1 << j)
    for _ in range(300):
        a, b, c = (rng.getrandbits(24) for _ in range(3))
        assert L.k_mul(a ^ c, b) == L.k_mul(a, b) ^ L.k_mul(c, b) == ref_kmul(a ^ c, b)
    for a in [1, 2, 4095, 4096, 4097, (1 << 24) - 1, 1 << 23, 1 << 11] + [rng.getrandbits(24) | 1 for _ in range(20)]:
        assert L.k_inv(a) == ref_kpow(a, (1 << 24) - 2)
        assert L.k_conj(a) == ref_kpow(a, 1 << 12) and L.k_pow(a, 54321) == ref_kpow(a, 54321)
    assert L.k_mul(0, 77) == 0 and L.k_mul(1, 0xABCDEF) == 0xABCDEF and L.k_pow(0, 3) == 0
    cs, xis = L.node_quadratics(), L.node_points()
    assert cs == list(range(2048, 2083))
    for c, xi in zip(cs, xis):                                     # both roots, the smaller one is chosen
        assert ref_kmul(xi, xi) ^ xi ^ c == 0 == ref_kmul(xi ^ 1, xi ^ 1) ^ xi ^ 1 ^ c and xi < xi ^ 1
    M = [1]                                                        # M rebuilt with kickmix arithmetic
    for c in cs:                                                   # (x^2 + x + c) M
        P = M + [0, 0]
        M = [fmul(c, P[i]) ^ (P[i - 1] if i else 0) ^ (P[i - 2] if i > 1 else 0) for i in range(len(P))]
    assert M == L.node_modulus()


def bits_of(rows, width):
    """rows[shot][i] ints -> bool [len(rows[0]) * width, shots], little-endian, as verify_dqi.to_bits."""
    arr = np.array(rows, dtype=np.int64)
    return ((arr[:, :, None] >> np.arange(width)) & 1).astype(bool).reshape(len(rows), -1).T


def test_node_circuit_against_locator_form():
    """End to end on the simulator with real syndromes.  Expected values from a third formula,
    N(x) = sum_l Y_l X_l (M(x) - M(X_l)) / (x - X_l)  (synthetic division of M per error location),
    evaluated at xi_k with the K arithmetic of this file."""
    rng = random.Random(26)
    M, xis = L.node_modulus(), L.node_points()
    gpow = [1]
    for _ in range(4094):
        gpow.append(fmul(gpow[-1], FK.primitive_element))
    errs = [{j: rng.randrange(1, 4096) for j in rng.sample(range(4095), w)} for w in list(range(36)) + [35] * 8]
    errs += [{0: 1}, {4094: 4095}, {j: 4095 for j in range(35)}, {j: 1 for j in range(4060, 4095)}]
    S_rows, H_rows = [], []
    for err in errs:
        S, N = [0] * 70, [0] * 70
        for j, y in err.items():
            x, p = gpow[j], y
            for i in range(70):
                p = fmul(p, x)                                     # y x^(i+1)
                S[i] ^= p
            quo, s = M[70], fmul(y, x)
            for k in range(69, -1, -1):                            # quo = coefficient k of M / (x - X_l)
                N[k] ^= fmul(s, quo)
                quo = M[k] ^ fmul(x, quo)
            assert quo != 0                                        # = M(X_l): no locator is a root of M
        H = []
        for xi in xis:
            r = 0
            for c in reversed(N):
                r = ref_kmul(r, xi) ^ c
            H.append(r)
        S_rows.append(S), H_rows.append(H)
    assert not any(H_rows[0]) and all(all(H) for H in H_rows[1:])
    X, want = bits_of(S_rows, 12), bits_of(H_rows, 24)
    T = L.node_transform()
    unit = np.zeros((840, 842), dtype=bool)                        # 840 unit vectors, all-zero, all-one
    unit[np.arange(840), np.arange(840)] = True
    unit[:, 841] = True
    Tn = L.to_numpy(T).astype(bool)
    Tin = L.to_numpy(L.gf2_inverse(T))
    assert ((Tn.astype(np.int64) @ Tin) % 2 == np.eye(840, dtype=np.int64)).all()
    circuits = []
    for emit in (L.emit_node_transform, L.emit_node_transform_inverse):
        b = km.CircuitBuilder()
        q = [x for i in range(70) for x in b.create_quantum_register(12, name=f'S{i}')]
        emit(b, q)
        circuits.append((b.finish_circuit(), q))
        st = circuits[-1][0].operation_statistics()
        assert circuits[-1][0].num_qubits == 840 and circuits[-1][0].num_bits == 0     # no ancilla, no bits
        assert st.num_toffoli_like == 0 and set(st.keys()) <= {'CX', 'SWAP'}
        assert st.num_instructions == len(L.node_program()) < 130_000
    (fwd, qf), (inv, qi) = circuits
    Y, toffoli, flipped = simulate(fwd, qf, X)
    assert toffoli == 0 and not flipped.any() and (Y == want).all()
    Z, toffoli, flipped = simulate(inv, qi, want)
    assert toffoli == 0 and not flipped.any() and (Z == X).all()
    Y, toffoli, flipped = simulate(fwd, qf, unit)
    assert toffoli == 0 and not flipped.any()
    assert (Y[:, :840] == Tn).all() and not Y[:, 840].any() and (Y[:, 841] == (Tn.sum(axis=1) % 2 == 1)).all()
    Z, toffoli, flipped = simulate(inv, qi, unit)
    assert toffoli == 0 and not flipped.any() and (Z[:, :840] == Tin.astype(bool)).all() and not Z[:, 840].any()


def test_conventions_agree_with_sibling_modules():
    """model/thiele.py and circ/field.py define K and the nodes on their own: same constants, same values."""
    thiele = pytest.importorskip('model.thiele')
    field = pytest.importorskip('circ.field')
    rng = random.Random(27)
    assert thiele.C0 == field.C == L.K_C
    assert thiele.GAM == L.node_quadratics() and thiele.XI == L.node_points()
    assert thiele.modulus() == L.node_modulus()
    for _ in range(200):
        a, b = rng.getrandbits(24), rng.getrandbits(24)
        assert thiele.kmul(a, b) == field.mul24(a, b) == L.k_mul(a, b)
        assert thiele.kconj(a) == field.conj24(a) == L.k_conj(a)
        if a:
            assert thiele.kinv(a) == field.inv24(a) == L.k_inv(a)
    T = L.node_transform()
    for w in (0, 1, 17, 35):
        err = {j: rng.randrange(1, 4096) for j in rng.sample(range(4095), w)}
        S = syndromes(err)
        nodes = L.gf2_apply(T, sum(s << 12 * i for i, s in enumerate(S)))
        assert [(nodes >> 24 * k) & 0xFFFFFF for k in range(35)] == list(thiele.node_values(S))

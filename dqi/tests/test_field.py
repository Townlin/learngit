"""Tests for circ.field: scheme exactness, classical helpers, and simulator checks of every emitter."""
import random

import numpy as np
import pytest

import kickmix as km
from circ import field as f
from model import gf

BATCH = 256
RNG = random.Random(20261113)


def rand(n, zero_first=True):
    """BATCH random n-bit values (the first one 0 so that the 0 -> 0 convention is exercised)."""
    v = [RNG.randrange(1 << n) for _ in range(BATCH)]
    if zero_first:
        v[0] = 0
    return v


def to_bits(vals, n):
    return np.array([[v >> i & 1 for v in vals] for i in range(n)], dtype=bool)


def from_bits(arr):
    return [sum(int(arr[i, s]) << i for i in range(arr.shape[0])) for s in range(arr.shape[1])]


def simulate(sizes, body, values, seed=7):
    """Build body(b, *registers) on fresh quantum registers, run it on values[k][shot]; returns a dict with the
    final register values, per-shot phase flips, leftover ancilla flag, Toffoli counts and the qubit count."""
    b = km.CircuitBuilder()
    regs = [b.create_quantum_register(n, name=f'r{k}') for k, n in enumerate(sizes)]
    body(b, *[list(r) for r in regs])
    assert b.num_allocated_qubits == sum(sizes), 'ancilla leak'
    circ = b.finish_circuit()
    batch = len(values[0])
    sim = km.Simulator(batch_size=batch, debug_print=False, counting_operations=True, seed=seed)
    sim.use_same_registers_as(circ)
    for r, v in zip(regs, values):
        sim.write_across_shots(r, to_bits(v, len(r)))
    sim.do(circ)
    anc = km.array([km.q(k) for k in range(sum(sizes), circ.num_qubits)])
    tof = sim.operation_statistics().num_toffoli_like
    assert tof == batch * circ.operation_statistics().num_toffoli_like          # every shot runs every Toffoli
    return dict(out=[from_bits(sim.read_across_shots(r)) for r in regs],
                flipped=bool(sim.read_phase_flipped_across_shots().any()),
                junk=bool(len(anc) and sim.read_across_shots(anc).any()),
                tof=tof // batch, extra=circ.num_qubits - sum(sizes))


def clean(res, tof, extra):
    """No phase flip, ancillas back to |0>, exact Toffoli-like count and exact transient qubit count."""
    assert not res['flipped'] and not res['junk']
    assert (res['tof'], res['extra']) == (tof, extra)


# ------------------------------------------------------------------ classical layer

def test_scheme_ranks_and_exactness():
    assert len(f.F42) == 42 and f.scheme_exact(f.F42, gf.mul, 12)
    assert len(f.K126) == 126 and f.scheme_exact(f.K126, f.mul24, 24)
    assert len(f.K99) == 99 and f.scheme_exact(f.K99, f.mul24, 24)
    assert f.F_SCHEME is f.F42 and f.K_SCHEME is f.K99
    kf = km.GF2Field(12)                                        # the scheme is in the kickmix basis
    assert kf.modulus == gf.MOD and f.scheme_exact(f.F42, kf.mul, 12)
    bad = [(al ^ 1, be, ga) for al, be, ga in f.F42]
    assert not f.scheme_exact(bad, gf.mul, 12)                  # the checker does detect a wrong scheme
    for _ in range(300):
        x, y = RNG.randrange(4096), RNG.randrange(4096)
        assert f.scheme_eval(f.F42, x, y) == gf.mul(x, y)
        x, y = RNG.randrange(1 << 24), RNG.randrange(1 << 24)
        assert f.scheme_eval(f.K99, x, y) == f.scheme_eval(f.K126, x, y) == f.mul24(x, y)


def test_small_field_schemes():
    for sch, basis, rank in ((f.S4, f.B4, 3), (f.S8, f.B8, 6), (f.S16, f.B16, 9)):
        assert len(sch) == rank
        for a in range(1 << len(basis)):
            for c in range(1 << len(basis)):
                assert f.scheme_eval(sch, a, c) == gf.mul(f.lin(basis, a), f.lin(basis, c))
    assert len(f.S64C) == 15
    for a in range(64):
        for c in range(64):
            assert f.lin(f.B64, f.scheme_eval(f.S64C, a, c)) == gf.mul(f.lin(f.B64, a), f.lin(f.B64, c))


def test_field_k():
    assert f.trace12(f.C) == 1 and not any(f.trace12(c) for c in range(1, f.C))
    assert f.mul24(f.U, f.U) == f.U ^ f.C                       # U^2 = U + C
    assert f.pow24(f.U ^ 2, (1 << 24) - 1) == 1
    for _ in range(300):
        a, c, e = (RNG.randrange(1, 1 << 24) for _ in range(3))
        assert f.mul24(a, c) == f.mul24(c, a) and f.mul24(a, 1) == a
        assert f.mul24(f.mul24(a, c), e) == f.mul24(a, f.mul24(c, e))
        assert f.mul24(a, c ^ e) == f.mul24(a, c) ^ f.mul24(a, e)
        assert f.mul24(a & 4095, c & 4095) == gf.mul(a & 4095, c & 4095)     # F sits in K as hi = 0
        assert f.conj24(a) == f.pow24(a, 1 << 12) and f.conj24(f.conj24(a)) == a
        assert f.mul24(a, f.conj24(a)) == f.norm24(a) < 4096
        assert f.mul24(a, f.inv24(a)) == 1
    assert f.inv24(0) == 0 and f.inv12(0) == 0
    try:
        from circ import linear                                 # sibling module: same K if present
    except ImportError:
        return
    if hasattr(linear, 'K_C') and hasattr(linear, 'k_mul'):
        assert linear.K_C == f.C and all(linear.k_mul(a, a ^ 77) == f.mul24(a, a ^ 77) for a in rand(24))


def ref_inv(v, n):
    """Independent reference for the inverse on an n-bit register (B8 / B64 coordinates for n = 3 / 6)."""
    if n == 24:
        return f.inv24(v)
    basis = {3: f.B8, 6: f.B64, 12: f.UNIT12}[n]
    return f.coord_fn(basis)(f.inv12(f.lin(basis, v)))


def test_quadratic_maps_and_inverse_models():
    c8 = f.coord_fn(f.B8)
    assert all(f.lin(f.B64, f.quad(x, 12)) == gf.power(x, 65) for x in range(4096))      # norm F -> GF(64)
    assert all(f.quad(v, 6) == c8(gf.power(f.lin(f.B64, v), 9)) for v in range(64))      # norm GF(64) -> GF(8)
    assert all(f.quad(x, 24) == f.norm24(x) for x in rand(24))                           # norm K -> F
    assert all(f.inv_model(v, n) == ref_inv(v, n) for n in (3, 6, 12) for v in range(1 << n))
    assert all(f.inv_model(x, 24) == f.inv24(x) for x in rand(24))
    assert all(gf.mul(x, f.inv12(x)) == 1 for x in range(1, 4096))
    assert [len(q) for q in (f.QN, f.QD, f.QJ, f.QI, f.QT, f.QK)] == [15, 6, 3, 12, 30, 84]
    for n in (6, 12, 24):                       # the identity behind the free erasures: N(x^-1) = N(x)^-1
        for x in rand(n):
            assert f.quad(ref_inv(x, n), n) == ref_inv(f.quad(x, n), n // 2)


# ------------------------------------------------------------------ circuits

@pytest.mark.parametrize('n,scheme,anc', [(12, f.F42, True), (12, f.F42, False), (24, f.K99, True),
                                          (24, f.K99, False), (24, f.K126, False), (12, None, False),
                                          (24, None, False)])
def test_mul_xor(n, scheme, anc):
    mul = gf.mul if n == 12 else f.mul24
    x, y, t = rand(n), rand(n), rand(n, zero_first=False)
    res = simulate([n, n, n], lambda b, qx, qy, qt: f.mul_xor(b, qx, qy, qt, scheme, anc=anc), [x, y, t])
    assert res['out'] == [x, y, [c ^ mul(a, d) for a, d, c in zip(x, y, t)]]
    clean(res, len(scheme or (f.F42 if n == 12 else f.K99)), int(anc))


@pytest.mark.parametrize('n,scheme', [(12, f.F42), (24, f.K99), (24, f.K126)])
def test_erase_product(n, scheme):
    mul = gf.mul if n == 12 else f.mul24
    x, y = rand(n), rand(n)
    t = [mul(a, d) for a, d in zip(x, y)]
    res = simulate([n, n, n], lambda b, qx, qy, qt: f.erase_product(b, qt, qx, qy, scheme), [x, y, t])
    assert res['out'] == [x, y, [0] * BATCH]
    clean(res, 0, 0)
    # a register that does not hold the product is detected through the phase
    res = simulate([n, n, n], lambda b, qx, qy, qt: f.erase_product(b, qt, qx, qy, scheme), [x, y, rand(n)])
    assert res['flipped']


def test_mul_then_erase_is_identity():
    x, y = rand(24), rand(24)

    def body(b, qx, qy, qt):
        f.mul_xor(b, qx, qy, qt)
        f.erase_product(b, qt, qx, qy)
    res = simulate([24, 24, 24], body, [x, y, [0] * BATCH])
    assert res['out'] == [x, y, [0] * BATCH]
    clean(res, 99, 0)


@pytest.mark.parametrize('n,tof', [(24, 42), (12, 15), (6, 6), (3, 3)])
def test_quad(n, tof):
    """quad_xor / quad_erase: norms (and the GF(8) inverse) as products of two parities of one register."""
    m = max(n // 2, 3)
    x, y = rand(n), rand(m, zero_first=False)
    want = [f.quad(a, n) for a in x]
    for anc in (True, False):
        res = simulate([n, m], lambda b, qx, qy: f.quad_xor(b, qx, qy, anc=anc), [x, y])
        assert res['out'] == [x, [c ^ w for c, w in zip(y, want)]]
        clean(res, tof, int(anc))
    res = simulate([n, m], lambda b, qx, qy: f.quad_erase(b, qx, qy), [x, want])
    assert res['out'] == [x, [0] * BATCH]
    clean(res, 0, 0)


@pytest.mark.parametrize('n,tof,extra', [(3, 3, 0), (6, 21, 3), (12, 66, 6), (24, 192, 12)])
def test_inv_init(n, tof, extra):
    x = rand(n)
    want = [ref_inv(a, n) for a in x]
    for anc in (False, True):
        res = simulate([n, n], lambda b, qx, qt: f.inv_init(b, qx, qt, anc=anc), [x, [0] * BATCH])
        assert res['out'] == [x, want]
        clean(res, tof, extra + anc)


@pytest.mark.parametrize('n,tof,extra', [(3, 0, 0), (6, 9, 6), (12, 39, 6), (24, 123, 12)])
def test_inv_del(n, tof, extra):
    x = rand(n)
    res = simulate([n, n], lambda b, qx, qt: f.inv_del(b, qx, qt), [x, [ref_inv(a, n) for a in x]])
    assert res['out'] == [x, [0] * BATCH]
    clean(res, tof, extra)
    res = simulate([n, n], lambda b, qx, qt: f.inv_del(b, qx, qt), [x, rand(n, zero_first=False)])
    assert res['flipped'] or res['junk'] or res['out'][1] != [0] * BATCH        # wrong t is not silently accepted


def test_lent_pool_is_used_and_returned_clean():
    x = rand(24)

    def body(b, qx, qt, qp):
        f.inv_init(b, qx, qt, pool=qp)
        f.inv_del(b, qx, qt, pool=qp)
    res = simulate([24, 24, 12], body, [x, [0] * BATCH, [0] * BATCH])
    assert res['out'] == [x, [0] * BATCH, [0] * BATCH]
    clean(res, 192 + 123, 0)


@pytest.mark.parametrize('n,anc,lean,tof,extra', [(12, True, False, 66, 19), (12, False, False, 66, 18),
                                                  (24, True, False, 192, 43), (24, False, False, 192, 42),
                                                  (24, True, True, 216, 31), (24, False, True, 216, 30)])
def test_inv_compute(n, anc, lean, tof, extra):
    fn = (lambda *a, **k: f.inv24_compute(*a, lean=lean, **k)) if n == 24 else f.inv12_compute
    inv = f.inv24 if n == 24 else f.inv12
    x, t = rand(n), rand(n, zero_first=False)
    x[1], x[2] = 5, (5 << 12) % (1 << n)                        # K: elements of F and of F*U
    res = simulate([n, n], lambda b, qx, qt: fn(b, qx, qt, anc=anc), [x, t])
    assert res['out'] == [x, [c ^ inv(a) for a, c in zip(x, t)]]
    clean(res, tof, extra)
    res = simulate([n, n], lambda b, qx, qt: fn(b, qx, qt, anc=anc, erase=True), [x, [inv(a) for a in x]])
    assert res['out'] == [x, [0] * BATCH]
    clean(res, tof - (84 if n == 24 else 30), extra)


@pytest.mark.parametrize('n,lean,tof,extra', [(12, False, 66, 30), (12, True, 75, 24),
                                              (24, False, 192, 54), (24, True, 231, 36)])
def test_inv_inplace(n, lean, tof, extra):
    x = rand(n)
    res = simulate([n], lambda b, qx: f.inv_inplace(b, qx, lean=lean), [x])
    assert res['out'] == [[ref_inv(a, n) for a in x]]
    clean(res, tof, extra)


def test_inverse_exhaustive_f_and_structured_k():
    """All 4096 elements of F through inv_init and inv_inplace; subfield / degenerate-looking elements of K."""
    x = list(range(4096))
    res = simulate([12, 12], lambda b, qx, qt: f.inv_init(b, qx, qt), [x, [0] * 4096])
    assert res['out'] == [x, [f.inv12(a) for a in x]]
    clean(res, 66, 6)
    res = simulate([12], lambda b, qx: f.inv_inplace(b, qx, lean=True), [x])
    assert res['out'] == [[f.inv12(a) for a in x]]
    clean(res, 75, 24)
    sub = [gf.power(gf.G, 65 * k) for k in range(63)]                       # GF(64)*
    k = [0, 1, f.U, f.U ^ 1] + sub + [s << 12 for s in sub] + [s | s << 12 for s in sub]
    k += [f.pow24(f.U ^ 2, e) for e in range(1, 64)]
    res = simulate([24, 24], lambda b, qx, qt: f.inv_init(b, qx, qt), [k, [0] * len(k)])
    assert res['out'] == [k, [f.inv24(a) for a in k]]
    clean(res, 192, 12)
    res = simulate([24], lambda b, qx: f.inv_inplace(b, qx), [k])
    assert res['out'] == [[f.inv24(a) for a in k]]
    clean(res, 192, 54)


def test_compose_and_scalar_scheme():
    conj, sq = f.linmap(f.conj24, 24), f.linmap(lambda v: f.mul24(v, v), 24)
    sch = f.compose(f.K99, lx=conj, ly=sq, lt=f.linmap(lambda v: f.mul24(v, 0xABCDEF), 24))
    assert len(sch) == 99 and len(f.FK84) == 84
    for a, c in zip(rand(24), rand(24)):
        assert f.scheme_eval(sch, a, c) == f.mul24(0xABCDEF, f.mul24(f.conj24(a), f.mul24(c, c)))
        assert f.scheme_eval(f.FK84, a, c & 4095) == f.mul24(a, c & 4095)
        assert f.scheme_eval(f.QK, a, c & 4095) == f.mul24(f.conj24(a), c & 4095)


def test_composite_circuit_on_builder_arrays():
    """Registers passed as km.array straight from the builder; every operation followed by its partner."""
    b = km.CircuitBuilder()
    x, y, t = (b.create_quantum_register(24, name=nm) for nm in 'xyt')
    f.mul_xor(b, x, y, t)
    f.erase_product(b, t, x, y)
    f.inv_init(b, x, t)
    f.inv_del(b, x, t)
    f.inv24_compute(b, x, t, lean=True)
    f.inv24_compute(b, x, t, erase=True)
    f.inv_inplace(b, x)
    f.inv_inplace(b, x, lean=True, anc=True)
    f.quad_xor(b, y, t[:12])
    f.quad_erase(b, y, t[:12])
    circ = b.finish_circuit()
    assert b.num_allocated_qubits == 72 and circ.num_qubits == 72 + 54
    assert circ.operation_statistics().num_toffoli_like == 99 + 192 + 123 + 216 + 108 + 192 + 231 + 42
    sim = km.Simulator(batch_size=BATCH, debug_print=False, seed=11)
    sim.use_same_registers_as(circ)
    xv, yv = rand(24), rand(24)
    sim.write_across_shots(x, to_bits(xv, 24))
    sim.write_across_shots(y, to_bits(yv, 24))
    sim.do(circ)
    assert from_bits(sim.read_across_shots(x)) == xv and from_bits(sim.read_across_shots(y)) == yv
    rest = km.array([km.q(k) for k in range(48, circ.num_qubits)])
    assert not sim.read_across_shots(rest).any() and not sim.read_phase_flipped_across_shots().any()


# ------------------------------------------------------------------ independent review checks
# Everything below re-derives the expected values without model.gf tables, f.mul24 or f.scheme_eval.

MOD = (1 << 12) | 0b1010011
ORD24 = (1 << 24) - 1


def m12(a, c):
    """Schoolbook carry-less product followed by reduction modulo x^12 + x^6 + x^4 + x + 1."""
    p = 0
    for i in range(12):
        p ^= (a << i) * (c >> i & 1)
    for i in range(22, 11, -1):
        p ^= (MOD << (i - 12)) * (p >> i & 1)
    return p


def m24(a, c):
    """Schoolbook product (four F-products) in F[u]/(u^2 + u + 2048)."""
    a0, a1, c0, c1 = a & 4095, a >> 12, c & 4095, c >> 12
    hh = m12(a1, c1)
    return (m12(a0, c0) ^ m12(2048, hh)) | ((m12(a0, c1) ^ m12(a1, c0) ^ hh) << 12)


def pw(mul, a, e):
    r = 1
    while e:
        if e & 1:
            r = mul(r, a)
        a, e = mul(a, a), e >> 1
    return r


def inv_pw(x, n):
    """x^(2^n - 2) by square-and-multiply (0 -> 0)."""
    return pw(m12 if n == 12 else m24, x, (1 << n) - 2) if x else 0


def edge(n, count=BATCH):
    """0, 1, all ones, single bits and their complements, subfield elements (K: also F, F*u, F*(u+1)), random."""
    full = (1 << n) - 1
    v = [0, 1, full, full ^ 1, 3] + [1 << i for i in range(n)] + [full ^ (1 << i) for i in range(n)]
    sub = [pw(m12, 2, 4095 // k * j) for k in (3, 7, 15, 63) for j in (1, 2)]
    v += sub if n == 12 else sub + [s << 12 for s in sub] + [s * 4097 for s in sub] + [4095, 4095 << 12]
    return v + [RNG.randrange(1 << n) for _ in range(count - len(v))]


def test_independent_field_arithmetic():
    kf = km.GF2Field(12)
    assert MOD == kf.modulus == gf.MOD and kf.primitive_element == gf.G == 2
    for _ in range(3000):
        a, c = RNG.randrange(4096), RNG.randrange(4096)
        assert m12(a, c) == kf.mul(a, c) == gf.mul(a, c)
    reducible = {m12(r, r) ^ r for r in range(4096)}            # c for which z^2 + z + c has a root in F
    assert min(c for c in range(4096) if c not in reducible) == f.C == 2048
    for x in range(4096):
        assert f.inv12(x) == f.inv_model(x, 12) == inv_pw(x, 12)
        assert f.lin(f.B64, f.quad(x, 12)) == pw(m12, x, 65)
    for a in edge(24, 600):
        c = RNG.randrange(1 << 24)
        assert f.mul24(a, c) == m24(a, c) and f.conj24(a) == pw(m24, a, 4096)
        assert f.norm24(a) == m24(a, pw(m24, a, 4096)) == f.quad(a, 24) < 4096
        assert f.inv24(a) == f.inv_model(a, 24) == inv_pw(a, 24)
        assert f.scheme_eval(f.FK84, a, c & 4095) == m24(a, c & 4095)
        assert f.scheme_eval(f.QK, a, c & 4095) == m24(pw(m24, a, 4096), c & 4095)
    assert pw(m24, 4096, ORD24) == 1 and all(pw(m24, 4096, ORD24 // q) != 1 for q in (3, 5, 7, 13, 17, 241))


def test_schemes_against_independent_products():
    """Own evaluator and own products: all basis pairs, edge operands and random pairs; ranks as claimed."""
    def ev(s, x, y):
        r = 0
        for al, be, ga in s:
            r ^= ga * (bin(al & x).count('1') & bin(be & y).count('1') & 1)
        return r
    for s, mul, n, rank in ((f.F42, m12, 12, 42), (f.K99, m24, 24, 99), (f.K126, m24, 24, 126)):
        assert len(s) == rank == len({(al, be) for al, be, _ in s}) and all(0 < min(t) <= max(t) < 1 << n for t in s)
        assert all(ev(s, 1 << i, 1 << j) == mul(1 << i, 1 << j) for i in range(n) for j in range(n))
        e = edge(n, 300)
        assert all(ev(s, x, y) == mul(x, y) for x, y in zip(e, reversed(e)))
        assert all(ev(s, x, y) == mul(x, y) for x in (0, 1, (1 << n) - 1) for y in e[:60])


def test_parity_plan_on_overlapping_registers():
    """_parities on registers that share qubits: right parities, exact restore, px == py iff equal qubit sets."""
    for _ in range(20000):
        nq = RNG.randrange(2, 7)
        x, y = RNG.sample(range(nq), RNG.randrange(1, nq + 1)), RNG.sample(range(nq), RNG.randrange(1, nq + 1))
        al, be = RNG.randrange(1, 1 << len(x)), RNG.randrange(1, 1 << len(y))
        sx, sy = {x[i] for i in f.bits(al)}, {y[i] for i in f.bits(be)}
        px, cx, py, cy = f._parities(x, y, al, be)
        assert (px == py) == (sx == sy) and px not in cx
        if px == py:
            assert not cy
            continue
        v = [RNG.randrange(2) for _ in range(nq)]
        w = list(v)
        for q in cx:
            w[px] ^= w[q]
        for q in cy:
            w[py] ^= w[q]
        assert w[px] == sum(v[q] for q in sx) % 2 and w[py] == sum(v[q] for q in sy) % 2
        for q in cy:
            w[py] ^= w[q]
        for q in cx:
            w[px] ^= w[q]
        assert w == v


@pytest.mark.parametrize('n', [12, 24])
def test_same_register_squaring_is_free(n):
    """mul_xor(b, x, x, t) used to emit CCX q q t (invalid circuit): each term is linear, so it costs a CX."""
    mul = m12 if n == 12 else m24
    x, t = edge(n), rand(n, zero_first=False)
    sq = [mul(a, a) for a in x]
    for anc in (False, True):
        res = simulate([n, n], lambda b, qx, qt: f.mul_xor(b, qx, qx, qt, anc=anc), [x, t])
        assert res['out'] == [x, [c ^ s for c, s in zip(t, sq)]]
        clean(res, 0, 0)
    res = simulate([n, n], lambda b, qx, qt: f.erase_product(b, qt, qx, qx), [x, sq])
    assert res['out'] == [x, [0] * len(x)]
    clean(res, 0, 0)
    assert simulate([n, n], lambda b, qx, qt: f.erase_product(b, qt, qx, qx), [x, t])['flipped']


def test_target_overlap_is_rejected_at_build_time():
    b = km.CircuitBuilder()
    x, y = (list(b.create_quantum_register(12, name=nm)) for nm in 'xy')
    for bad in (lambda: f.mul_xor(b, x, y, x), lambda: f.mul_xor(b, x, y, y[6:] + x[:6]),
                lambda: f.erase_product(b, y, x, y), lambda: f.quad_xor(b, x + y, x)):
        with pytest.raises(ValueError):
            bad()


@pytest.mark.parametrize('n,seed', [(12, 3), (12, 4), (24, 3), (24, 4)])
def test_edge_inputs_every_emitter(n, seed):
    """Edge operands, two measurement seeds, expected values from square-and-multiply; exact counts."""
    mul, comp = (m12, f.inv12_compute) if n == 12 else (m24, f.inv24_compute)
    cnt = {12: dict(mul=42, init=(66, 6), dele=(39, 6), comp=(66, 18), era=(36, 18), inp=(66, 30), lean=(75, 24)),
           24: dict(mul=99, init=(192, 12), dele=(123, 12), comp=(192, 42), era=(108, 42), inp=(192, 54),
                    lean=(231, 36))}[n]
    x = edge(n)
    y, t, zero = x[::-1], rand(n, zero_first=False), [0] * len(x)
    inv = [inv_pw(a, n) for a in x]
    assert all(mul(a, i) == (a != 0) for a, i in zip(x, inv))
    prod = [mul(a, c) for a, c in zip(x, y)]
    for anc in (False, True):
        res = simulate([n, n, n], lambda b, qx, qy, qt: f.mul_xor(b, qx, qy, qt, anc=anc), [x, y, t], seed)
        assert res['out'] == [x, y, [c ^ p for c, p in zip(t, prod)]]
        clean(res, cnt['mul'], int(anc))
    res = simulate([n, n, n], lambda b, qx, qy, qt: f.erase_product(b, qt, qx, qy), [x, y, prod], seed)
    assert res['out'] == [x, y, zero]
    clean(res, 0, 0)
    for body, start, want, key in (
            (lambda b, qx, qt: f.inv_init(b, qx, qt), zero, inv, 'init'),
            (lambda b, qx, qt: f.inv_del(b, qx, qt), inv, zero, 'dele'),
            (lambda b, qx, qt: comp(b, qx, qt), t, [c ^ i for c, i in zip(t, inv)], 'comp'),
            (lambda b, qx, qt: comp(b, qx, qt, erase=True), inv, zero, 'era')):
        res = simulate([n, n], body, [x, start], seed)
        assert res['out'] == [x, want], key
        clean(res, *cnt[key])
    for lean in (False, True):
        res = simulate([n], lambda b, qx: f.inv_inplace(b, qx, lean=lean), [x], seed)
        assert res['out'] == [inv]
        clean(res, *cnt['lean' if lean else 'inp'])
    if n == 24:
        for erase, tof in ((False, 216), (True, 132)):
            res = simulate([n, n], lambda b, qx, qt: f.inv24_compute(b, qx, qt, lean=True, erase=erase),
                           [x, inv if erase else t], seed)
            assert res['out'] == [x, zero if erase else [c ^ i for c, i in zip(t, inv)]]
            clean(res, tof, 30)


def test_measurement_outcomes_differ_between_seeds():
    """The free erasures are only tested if HMR outcomes are random: the simulator must sample them."""
    recs = []
    for seed in (1, 2):
        b = km.CircuitBuilder()
        q, m = b.create_quantum_register(24, name='q'), b.create_classical_register(24, name='m')
        b.hmr(q, m)
        circ = b.finish_circuit()
        sim = km.Simulator(batch_size=64, seed=seed)
        sim.use_same_registers_as(circ)
        sim.write_across_shots(q, np.ones((24, 64), dtype=bool))
        sim.do(circ)
        rec = sim.read_across_shots(m)
        assert not sim.read_across_shots(q).any()
        assert (sim.read_phase_flipped_across_shots() == (rec.sum(axis=0) % 2 == 1)).all()     # (-1)^(m . v)
        assert 0.3 < rec.mean() < 0.7
        recs.append(rec)
    assert (recs[0] != recs[1]).any()


@pytest.mark.parametrize('cond', [False, True])
def test_inside_classically_conditioned_block(cond):
    """push_condition around the emitters: identity and zero executed Toffolis when the bit is 0."""
    b = km.CircuitBuilder()
    x, y, t = (b.create_quantum_register(24, name=nm) for nm in 'xyt')
    cb = b.create_classical_register(1, name='c')
    b.push_condition(cb[0])
    f.mul_xor(b, x, y, t)
    f.inv24_compute(b, x, t)
    f.inv24_compute(b, y, t, lean=True)
    f.inv_inplace(b, y)
    b.pop_condition()
    circ = b.finish_circuit()
    xv, yv, tv = edge(24), edge(24)[::-1], rand(24, zero_first=False)
    sim = km.Simulator(batch_size=BATCH, counting_operations=True, seed=5)
    sim.use_same_registers_as(circ)
    for r, v in ((x, xv), (y, yv), (t, tv)):
        sim.write_across_shots(r, to_bits(v, 24))
    sim.write_across_shots(cb, np.full((1, BATCH), cond))
    sim.do(circ)
    want = [xv, [inv_pw(a, 24) for a in yv], [c ^ m24(a, d) ^ inv_pw(a, 24) ^ inv_pw(d, 24) for a, d, c in zip(xv, yv, tv)]]
    assert [from_bits(sim.read_across_shots(r)) for r in (x, y, t)] == (want if cond else [xv, yv, tv])
    rest = km.array([km.q(k) for k in range(72, circ.num_qubits)])
    assert not sim.read_across_shots(rest).any() and not sim.read_phase_flipped_across_shots().any()
    assert sim.operation_statistics().num_toffoli_like == (99 + 192 + 216 + 192) * BATCH * cond


def test_cancels_kickmix_builtin_circuits():
    """Our multiplier / inverter XORed onto kickmix's own GF(2^12) circuits leaves |0> for all 4096 x."""
    fld, yv = km.GF2Field(12), [RNG.randrange(4096) for _ in range(4096)]
    for inverse, ours in ((False, 42), (True, 66)):
        for with_ours in (False, True):             # first pass: Toffoli-like count of the built-in alone
            b = km.CircuitBuilder()
            x, y, t = (b.create_quantum_register(12, name=nm) for nm in 'xyt')
            if inverse:
                b.init_gf2_inverse(x, target=t, field=fld)
                with_ours and f.inv12_compute(b, x, t)
            else:
                b.init_gf2_mul(x, y, target=t, field=fld)
                with_ours and f.mul_xor(b, x, y, t)
            circ = b.finish_circuit()
            if not with_ours:
                builtin = circ.operation_statistics().num_toffoli_like
        assert circ.operation_statistics().num_toffoli_like == builtin + ours and builtin > ours    # 63, 315 in 1.0.0
        sim = km.Simulator(batch_size=4096, seed=1)
        sim.use_same_registers_as(circ)
        sim.write_across_shots(x, to_bits(range(4096), 12))
        sim.write_across_shots(y, to_bits(yv, 12))
        sim.do(circ)
        assert from_bits(sim.read_across_shots(x)) == list(range(4096)) and from_bits(sim.read_across_shots(y)) == yv
        rest = km.array([km.q(k) for k in range(24, circ.num_qubits)])
        assert not sim.read_across_shots(rest).any() and not sim.read_phase_flipped_across_shots().any()


@pytest.mark.parametrize('n,size', [(12, 2), (12, 9), (24, 5), (24, 20)])
def test_partial_and_oversized_pool(n, size):
    x = edge(n)

    def body(b, qx, qt, qp):
        f.inv_init(b, qx, qt, pool=qp)
        f.inv_del(b, qx, qt, pool=qp)
        f.inv_init(b, qx, qt, pool=qp)
    res = simulate([n, n, size], body, [x, [0] * BATCH, [0] * BATCH])
    assert res['out'] == [x, [inv_pw(a, n) for a in x], [0] * BATCH]
    tof, need = {12: (66 + 39 + 66, 6), 24: (192 + 123 + 192, 12)}[n]
    clean(res, tof, max(0, need - size))


def interpret(circ, state, rng):
    """Independent sparse state-vector run of a circuit's text form.  state: {basis int: amplitude}, bit k of
    the key = qubit k.  Unlike the kickmix simulator this handles superpositions, so relative phases between
    branches are checked directly.  Returns (state, executed Toffoli-like gates)."""
    bit, stack, tof = {}, [], 0
    for line in str(circ).splitlines():
        p = line.split()
        if p[0] in ('APPEND_TO_REGISTER', 'REGISTER'):
            continue
        cut = p.index('if') if 'if' in p else len(p)
        op, a, cond = p[0], [int(s[1:]) for s in p[1:cut]], [int(s[1:]) for s in p[cut + 1:]]
        if op == 'POP_CONDITION':
            stack.pop()
        elif op == 'PUSH_CONDITION':
            stack += cond
        elif not all(bit.get(c, 0) for c in stack + cond):
            pass
        elif op in ('CX', 'CCX'):
            ctl, tgt = sum(1 << q for q in a[:-1]), 1 << a[-1]
            tof += op == 'CCX'
            state = {(k ^ tgt if k & ctl == ctl else k): v for k, v in state.items()}
        elif op in ('Z', 'CZ', 'CCZ'):
            ctl = sum(1 << q for q in a)
            tof += op == 'CCZ'
            state = {k: (-v if k & ctl == ctl else v) for k, v in state.items()}
        elif op == 'SWAP':
            both = (1 << a[0]) | (1 << a[1])
            state = {(k ^ both if 0 < k & both < both else k): v for k, v in state.items()}
        elif op == 'R':
            assert not any(k >> a[0] & 1 for k in state), 'reset of a qubit that is not |0>'
        elif op == 'HMR':
            m = rng.randrange(2)
            for _ in range(2):                      # second pass only if outcome m has probability 0
                new = {}
                for k, v in state.items():
                    k0 = k & ~(1 << a[0])
                    new[k0] = new.get(k0, 0) + (-v if m and k != k0 else v)
                norm = sum(abs(v) ** 2 for v in new.values()) ** 0.5
                if norm > 1e-9:
                    break
                m ^= 1
            state, bit[a[1]] = {k: v / norm for k, v in new.items() if abs(v) > 1e-12}, m
        elif op == 'BIT_STORE0':
            bit[a[0]] = 0
        elif op == 'BIT_INVERT':
            bit[a[0]] = bit.get(a[0], 0) ^ 1
        else:
            raise ValueError('instruction not modelled: ' + line)
    return state, tof


def fidelity(sizes, body, support, expect, trials=2, seed=1):
    """Run body on a random superposition of the register tuples in `support`; returns the smallest
    |<wanted|actual>| over the trials (0 if the supports differ) and the executed Toffoli-like count."""
    rng = random.Random(seed)
    b = km.CircuitBuilder()
    regs = [list(b.create_quantum_register(n, name=f'g{k}')) for k, n in enumerate(sizes)]
    body(b, *regs)
    circ = b.finish_circuit()
    pack = lambda vals: sum(v << sum(sizes[:k]) for k, v in enumerate(vals))
    worst = 1.0
    for _ in range(trials):
        amp = [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in support]
        nrm = sum(abs(z) ** 2 for z in amp) ** 0.5
        want = {pack(expect(v)): z / nrm for v, z in zip(support, amp)}
        out, tof = interpret(circ, {pack(v): z / nrm for v, z in zip(support, amp)}, rng)
        assert len(want) == len(support)
        ov = abs(sum(want[k].conjugate() * out.get(k, 0) for k in want)) if set(out) == set(want) else 0.0
        worst = min(worst, ov)
    return worst, tof


def test_superpositions_with_independent_interpreter():
    xs = [0, 1, 4095] + RNG.sample(range(2, 4095), 61)
    ks = [0, 1, 4096, 4097, 5, 5 << 12, ORD24] + [RNG.randrange(1 << 24) for _ in range(25)]
    y12, y24 = [RNG.randrange(4096) for _ in xs], [RNG.randrange(1 << 24) for _ in ks]
    i12, i24 = (lambda v: inv_pw(v, 12)), (lambda v: inv_pw(v, 24))
    cases = [
        ([12, 12, 12], lambda b, x, y, t: f.mul_xor(b, x, y, t), list(zip(xs, y12, y12[::-1])),
         lambda v: (v[0], v[1], v[2] ^ m12(v[0], v[1])), 42),
        ([12, 12, 12], lambda b, x, y, t: f.mul_xor(b, x, y, t, anc=True), list(zip(xs, y12, y12[::-1])),
         lambda v: (v[0], v[1], v[2] ^ m12(v[0], v[1])), 42),
        ([12, 12, 12], lambda b, x, y, t: f.erase_product(b, t, x, y), [(x, y, m12(x, y)) for x, y in zip(xs, y12)],
         lambda v: (v[0], v[1], 0), 0),
        ([12, 12], lambda b, x, t: f.mul_xor(b, x, x, t), list(zip(xs, y12)), lambda v: (v[0], v[1] ^ m12(v[0], v[0])), 0),
        ([12, 12], lambda b, x, t: f.inv_init(b, x, t), [(x, 0) for x in xs], lambda v: (v[0], i12(v[0])), 66),
        ([12, 12], lambda b, x, t: f.inv_del(b, x, t), [(x, i12(x)) for x in xs], lambda v: (v[0], 0), 39),
        ([12, 12], lambda b, x, t: f.inv12_compute(b, x, t), list(zip(xs, y12)), lambda v: (v[0], v[1] ^ i12(v[0])), 66),
        ([12, 12], lambda b, x, t: f.inv12_compute(b, x, t, erase=True), [(x, i12(x)) for x in xs],
         lambda v: (v[0], 0), 36),
        ([12], lambda b, x: f.inv_inplace(b, x), [(x,) for x in xs], lambda v: (i12(v[0]),), 66),
        ([12], lambda b, x: f.inv_inplace(b, x, lean=True), [(x,) for x in xs], lambda v: (i12(v[0]),), 75),
        ([24, 24, 24], lambda b, x, y, t: f.erase_product(b, t, x, y), [(x, y, m24(x, y)) for x, y in zip(ks, y24)],
         lambda v: (v[0], v[1], 0), 0),
        ([24, 24], lambda b, x, t: f.inv_init(b, x, t), [(x, 0) for x in ks], lambda v: (v[0], i24(v[0])), 192),
        ([24, 24], lambda b, x, t: f.inv_del(b, x, t), [(x, i24(x)) for x in ks], lambda v: (v[0], 0), 123),
        ([24, 24], lambda b, x, t: f.inv24_compute(b, x, t, lean=True), list(zip(ks, y24)),
         lambda v: (v[0], v[1] ^ i24(v[0])), 216),
        ([24], lambda b, x: f.inv_inplace(b, x), [(x,) for x in ks], lambda v: (i24(v[0]),), 192),
    ]
    for k, (sizes, body, support, expect, tof) in enumerate(cases):
        fid, got = fidelity(sizes, body, support, expect)
        assert abs(fid - 1) < 1e-9 and got == tof, (k, fid, got)

    def no_fixup(b, x, y, t):                       # negative control: X-measure the product without the CZs
        f.mul_xor(b, x, y, t)
        b.hmr(t, b.alloc_bits(12))
    fid, _ = fidelity([12, 12, 12], no_fixup, [(x, y, 0) for x, y in zip(xs, y12)], lambda v: (v[0], v[1], 0), trials=3)
    assert fid < 0.9

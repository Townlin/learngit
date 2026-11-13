"""Tests for model/thiele.py.  Run from the dqi directory: python3 -m pytest -q tests/test_thiele.py"""
import random

import pytest

from model import gf, reference
from model import thiele as T

M = T.modulus()


def kpow(a, e):
    r = 1
    while e:
        if e & 1:
            r = T.kmul(r, a)
        a, e = T.kmul(a, a), e >> 1
    return r


def decode_case(rng, w):
    err = T.random_error(rng, w)
    S = T.syndromes(err)
    return err, S, T.node_values(S, M)


# ---------------------------------------------------------------- K = GF(2^24) and the nodes

def test_field_k():
    rng = random.Random(1)
    assert T.C0 == min(a for a in range(4096) if T.trace(a)) and all(T.trace(g) for g in T.GAM)
    assert T.GAM == sorted(T.GAM) == T.TR1[:35] and len(T.TR1) == 2048
    assert T.kmul(T.U, T.U) == T.U ^ T.C0                      # u^2 = u + C0
    for _ in range(300):
        a, b, c = (rng.randrange(1, 1 << 24) for _ in range(3))
        assert T.kmul(a, b) == T.kmul(b, a) and T.kmul(a, 1) == a
        assert T.kmul(T.kmul(a, b), c) == T.kmul(a, T.kmul(b, c))
        assert T.kmul(a, b ^ c) == T.kmul(a, b) ^ T.kmul(a, c)
        assert T.kmul(a, T.kinv(a)) == 1
        assert T.kconj(a) == kpow(a, 1 << 12) and T.knorm(a) == T.kmul(a, T.kconj(a))
        x, y = a & 4095, b & 4095                              # F embeds as hi = 0
        assert T.kmul(x, y) == gf.mul(x, y) and T.kscale(x, b) == T.kmul(x, b)
    assert T.kinv(0) == 0 and kpow(rng.randrange(1, 1 << 24), (1 << 24) - 1) == 1


def test_nodes():
    assert len(M) == 71 and M[-1] == 1 and all(gf.peval(M, x) for x in range(4096))
    for k in range(T.NODES):
        xi, other = T.XI[k], T.kconj(T.XI[k])
        assert T.keval(T.m_poly(k), xi) == 0 == T.keval(T.m_poly(k), other) == T.keval(M, xi)
        assert other == xi ^ 1 and xi < other and xi == T.U | T.TAU[k]
        for g in range(T.NODES):                               # the two F-valued chain constants
            assert T.keval(T.m_poly(k), T.XI[g]) == T.GAM[k] ^ T.GAM[g]
            assert T.XI[k] ^ T.XI[g] == T.TAU[k] ^ T.TAU[g]
    rng = random.Random(2)
    for _ in range(50):                                        # (a_k, b_k) coordinates of a word
        k, g, word = rng.randrange(35), rng.randrange(35), rng.randrange(1 << 24)
        nu = T.nu_poly(word, k)
        assert len(nu) <= 2 and T.keval(nu, T.XI[k]) == word
        assert T.keval(nu, T.XI[g]) == T.nu_at(word, k, g)


def test_node_values_identity():
    """h_k = N(xi_k) (from syndromes) = H(xi_k) (definition) = R(xi_k)/Q(xi_k) (reference decode)."""
    rng = random.Random(3)
    for w in (0, 1, 2, 17, 34, 35, 35, 35):
        err, S, h = decode_case(rng, w)
        assert h == T.node_values_direct(err, M)
        Q, R, _ = reference.decode(S, M)
        assert gf.pdeg(Q) == w and Q == gf.from_roots([gf.EXP[j] for j in sorted(err)])
        for k in range(T.NODES):
            assert T.kmul(T.keval(Q, T.XI[k]), h[k]) == T.keval(R, T.XI[k])


# ---------------------------------------------------------------- extract / inverse

def eager_words(Q, R):
    """The definition of the CF with explicit rational functions phi_k = A/B (no lazy chains)."""
    A, B, words = Q, R, []
    while B:
        k = len(words)
        c = T.kmul(T.keval(A, T.XI[k]), T.kinv(T.keval(B, T.XI[k])))
        quo, rem = gf.pdivmod(gf.padd(A, gf.pmul(T.nu_poly(c, k), B)), T.m_poly(k))
        assert not rem                                         # phi_(k-1) - nu_k vanishes at xi_k
        A, B = B, quo
        words.append(c)
    return words, A


def test_extract_matches_definition_and_counts():
    rng = random.Random(4)
    for w in (35, 35, 35, 34, 29, 16):
        err, S, h = decode_case(rng, w)
        Q, R, _ = reference.decode(S, M)
        st, st2 = {}, {}
        c, pole = T.extract(h, st)
        words, last = eager_words(Q, R)
        assert len(words) == w and gf.pdeg(last) == 0          # phi_w = infinity after w levels
        assert c[:w] == words and not any(c[w:])
        assert (st['mul'], st['inv']) == (629, 35)             # 595 chain steps + 34 final R/D
        assert T.inverse(c, pole, st2) == h and (st2['mul'], st2['inv']) == (629, 35)


def test_roundtrip_on_arbitrary_words():
    """extract is total on K^35 (any 35 words, zeros included) and inverse undoes it."""
    rng = random.Random(5)
    for trial in range(40):
        h = [rng.randrange(1 << 24) for _ in range(T.NODES)]
        for g in rng.sample(range(T.NODES), trial % 6):
            h[g] = 0
        c, pole = T.extract(h)
        assert T.inverse(c, pole) == h
        assert all(not (p and x) for p, x in zip(pole, c))     # a pole always stores the word 0
    with pytest.raises(ValueError):                            # phi_0(xi_1) = 0 is not a node value
        T.inverse([0] * T.NODES, [0] * T.NODES)


def test_generic_words_solve_key_equation():
    """Any 35 nonzero words (b_k = 0 or a_k = 0 included: no 2^-12 special cases) describe the pair
    (N_0, D_0) with D_0 = h N_0 at every node, deg N_0 <= 35, deg D_0 <= 34."""
    rng = random.Random(12)
    for _ in range(20):
        c = [rng.randrange(1, 1 << 24) for _ in range(T.NODES)]
        for g in rng.sample(range(T.NODES), 6):
            c[g] &= 4095                                       # b_k = 0: nu_k constant
        for g in rng.sample(range(T.NODES), 3):
            c[g] = (c[g] >> 12 << 12) or 1                     # lo(c_k) = 0
        h = T.inverse(c, [0] * T.NODES)
        assert T.extract(h) == (c, [0] * T.NODES)
        N0, D0 = T.synthesize(c)
        assert gf.pdeg(N0) <= 35 and gf.pdeg(D0) <= 34
        assert all(T.keval(D0, x) == T.kmul(v, T.keval(N0, x)) for x, v in zip(T.XI, h))
        lead, prev = c[-1] >> 12, 1                            # x^35 coefficient of N_0 is the
        for k in reversed(range(T.NODES - 1)):                 # continuant of b_1..b_35
            lead, prev = gf.mul(c[k] >> 12, lead) ^ prev, lead
        assert lead == (N0 + [0] * 36)[35]


# ---------------------------------------------------------------- synthesis

def test_synthesize_full_weight_vs_reference():
    rng = random.Random(6)
    for _ in range(80):
        err, S, h = decode_case(rng, 35)
        c, pole = T.extract(h)
        assert not any(pole) and all(c)
        N0, D0 = T.synthesize(c)
        Q, R, _ = reference.decode(S, M)
        lam = gf.inv(N0[-1])
        assert gf.pdeg(N0) == 35 and gf.pscale(N0, lam) == Q and gf.pscale(D0, lam) == R
        assert T.error_values(N0, D0, err, M) == err


def test_variant_h_does_not_terminate():
    """Starting from phi_0 = H instead of 1/H needs 36 levels: 35 words do not reproduce R/Q."""
    rng = random.Random(7)
    err, S, h = decode_case(rng, 35)
    Q, R, _ = reference.decode(S, M)
    c, pole = T.extract([T.kinv(x) for x in h])                # chain start (1 : 1/h) = (h : 1)
    A, B = T.synthesize(c, pole[-1])                           # would have to be ~ (R, Q)
    assert not any(pole) and gf.padd(gf.pmul(A, Q), gf.pmul(B, R))
    c, pole = T.extract(h)
    A, B = T.synthesize(c, pole[-1])
    assert not gf.padd(gf.pmul(A, R), gf.pmul(B, Q))           # the chosen variant: A/B = Q/R


# ---------------------------------------------------------------- degenerate cases

def test_low_weight_conventions():
    """Zero pattern, flags, round trip, common factor and error values for all weights."""
    res = T.experiment(weights=range(16, 36), trials=8, seed=8)
    res.update(T.experiment(weights=range(0, 16), trials=2, seed=9))
    assert all(ok == n and acc == 0 for n, ok, acc in res.values()), res


def test_low_weight_pattern_explicit():
    rng = random.Random(10)
    for w in (33, 32):
        err, S, h = decode_case(rng, w)
        st = {}
        c, pole = T.extract(h, st)
        ends = {g: (r, d) for g, s, r, d in st['zeros'] if s == g}
        for g in range(T.NODES):                               # 1-based node G = g + 1
            if g + 1 <= w:
                assert g not in ends and c[g] and not pole[g]
            else:                                              # G - w odd: D = 0;  even: R = 0
                assert ends[g] == ((g + 1 - w) % 2 == 0, (g + 1 - w) % 2 == 1)
                assert c[g] == 0 and pole[g] == (g + 1 - w) % 2
        assert pole[-1] == (35 - w) % 2
        N0, D0 = T.synthesize(c, pole[-1])
        Q, R, _ = reference.decode(S, M)
        f = T.common_factor(pole)
        assert gf.pdeg(f) == 2 * ((35 - w) // 2) and all(gf.peval(f, x) for x in range(4096))
        assert gf.pscale(N0, gf.inv(N0[-1])) == gf.pmul(Q, f)
        assert gf.pscale(D0, gf.inv(N0[-1])) == gf.pmul(R, f)
        assert T.synthesize(c, 1 ^ pole[-1])[0] != N0          # the tail bit matters


def planted(rng, k, d, kind):
    """A coprime pair (N_0, D_0) whose CF has random words at nodes 1..k-1 and, at node k (1-based),
    an accidental pole (kind 'pole': m_k | denominator) or an accidental zero word (kind 'zero')."""
    while True:
        rnd = lambda deg: [rng.randrange(4096) for _ in range(deg)] + [rng.randrange(1, 4096)]
        if kind == 'pole':
            N, D = rnd(d), gf.pmul(T.m_poly(k - 1), rnd(d - 3))
        else:
            N, D = gf.pmul(T.m_poly(k - 1), rnd(d - 2)), rnd(d - 1)
        words = [rng.randrange(1, 1 << 24) for _ in range(k - 1)]
        for j in reversed(range(k - 1)):
            N, D = gf.padd(gf.pmul(T.nu_poly(words[j], j), N), gf.pmul(T.m_poly(j), D)), N
        if gf.pdeg(gf.pxgcd(N, D)[0]) == 0 and all(T.keval(N, x) for x in T.XI):
            return N, D, words


def test_accidental_degeneracies():
    rng = random.Random(11)
    # accidental pole at node 10, d(phi_0) = 29: absorbed, the CF ends at level 31 = 29 + 2
    N, D, words = planted(rng, 10, 20, 'pole')
    h = [T.kmul(T.keval(D, x), T.kinv(T.keval(N, x))) for x in T.XI]
    c, pole = T.extract(h)
    assert c[:9] == words and [g + 1 for g in range(35) if pole[g]] == [10, 32, 34]
    assert all(c[10:31]) and not any(c[31:]) and T.inverse(c, pole) == h
    N0, D0 = T.synthesize(c, pole[-1])
    f = gf.pmul(T.m_poly(9), gf.pmul(T.m_poly(31), T.m_poly(33)))
    assert f == T.common_factor(pole) and gf.pdeg(N0) == 35
    assert gf.pscale(N0, gf.inv(N0[-1])) == gf.pscale(gf.pmul(N, f), gf.inv(N[-1]))
    assert T.normalise(N0, D0, pole) == (gf.pscale(N, gf.inv(N[-1])), gf.pscale(D, gf.inv(N[-1])))
    assert T.poles_from_tail(c, pole[-1]) != pole              # one bit cannot describe an accident
    # accidental zero word at node 10: an ordinary word, nothing special happens
    N, D, words = planted(rng, 10, 26, 'zero')
    h = [T.kmul(T.keval(D, x), T.kinv(T.keval(N, x))) for x in T.XI]
    c, pole = T.extract(h)
    assert c[:9] == words and c[9] == 0 and not any(pole) and all(c[10:])
    N0, D0 = T.synthesize(c)
    assert gf.pscale(N0, gf.inv(N0[-1])) == gf.pscale(N, gf.inv(N[-1])) and T.inverse(c, pole) == h
    # accidental pole at d(phi_0) = 35: would need 37 levels -> the 35 words do not synthesize
    N, D, words = planted(rng, 10, 26, 'pole')
    h = [T.kmul(T.keval(D, x), T.kinv(T.keval(N, x))) for x in T.XI]
    c, pole = T.extract(h)
    assert [g + 1 for g in range(35) if pole[g]] == [10] and T.inverse(c, pole) == h
    N0, D0 = T.synthesize(c, pole[-1])
    assert gf.padd(gf.pmul(N0, D), gf.pmul(D0, N))             # not proportional: decode failure


# ================================================================ independent review checks
# Second code paths for everything above: K on the bitwise F multiplication, the CF by its definition
# in F[x] only, (Q, R) straight from the error, synthesis as a forward matrix product, Euclid on the
# interpolated node polynomial, accidents on real errors, and the numpy census.

def kmul_ref(a, b):
    """Schoolbook product in F[u]/(u^2 + u + C0) on the table-free F multiplication."""
    a0, a1, b0, b1, m = a & 4095, a >> 12, b & 4095, b >> 12, gf._mul_slow
    hh = m(a1, b1)
    return (m(a0, b0) ^ m(hh, T.C0)) | ((m(a0, b1) ^ m(a1, b0) ^ hh) << 12)


def direct_QR(err):
    """Q = prod (x - X_l) and R = sum_l Yt_l prod_(l' != l) (x - X_l'), straight from the error."""
    Xs, R = [gf.EXP[j] for j in err], []
    for j, v in err.items():
        X = gf.EXP[j]
        yt = gf.mul(gf.mul(v, X), gf.peval(M, X))
        R = gf.padd(R, gf.pscale(gf.from_roots([x for x in Xs if x != X]), yt))
    return gf.from_roots(Xs), R


def cf_in_F(Q, R):
    """The CF by its definition with F[x] arithmetic only (no K): nu_k = A * B^-1 mod m_k for
    phi_(k-1) = A/B, and nu_k = 0 at a pole (m_k | B).  Returns (words, pole flags)."""
    A, B, words, pole = list(Q), list(R), [], []
    for k in range(T.NODES):
        m = T.m_poly(k)
        Bm = gf.pmod(B, m)
        nu = gf.pmod(gf.pmul(gf.pmod(A, m), gf.pinv_mod(Bm, m)), m) if Bm else []
        a, b = (nu + [0, 0])[:2]
        words.append((a ^ gf.mul(b, T.TAU[k])) | b << 12)      # a + b xi_k with xi_k = u + tau_k
        pole.append(int(not Bm))
        if Bm:
            quo, rem = gf.pdivmod(gf.padd(A, gf.pmul(nu, B)), m)
            assert not rem
            A, B = B, quo                                      # phi_k = m B / (A - nu B)
        else:
            A, B = gf.pmul(m, B), A                            # phi_k = m / phi_(k-1)
    return words, pole


def forward_product(c):
    """prod_k [[nu_k, m_k], [1, 0]] accumulated left to right (the opposite order of synthesize)."""
    P = [[[1], []], [[], [1]]]
    for k in range(T.NODES):
        nu, m = T.nu_poly(c[k], k), T.m_poly(k)
        P = [[gf.padd(gf.pmul(r[0], nu), r[1]), gf.pmul(r[0], m)] for r in P]
    return P


def trailing_run(c):
    return next((i for i, x in enumerate(reversed(c)) if x), len(c))


def solve(A, b):
    """Gaussian elimination over F (square system); None if singular."""
    n, A = len(A), [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        p = next((r for r in range(i, n) if A[r][i]), None)
        if p is None:
            return None
        A[i], A[p] = A[p], A[i]
        A[i] = [gf.div(x, A[i][i]) for x in A[i]]
        for r in range(n):
            if r != i and A[r][i]:
                A[r] = [x ^ gf.mul(A[r][i], y) for x, y in zip(A[r], A[i])]
    return [row[n] for row in A]


def planted_error(rng, w, lin):
    """A weight-w error whose scaled values Yt_l = Y_l X_l M(X_l) satisfy the F-linear constraints
    sum_l Yt_l coef(X_l) = rhs for every (coef, rhs) in lin: all but len(lin) values are random."""
    while True:
        Xs = [gf.EXP[j] for j in rng.sample(range(gf.ORDER), w)]
        yt = [rng.randrange(1, 4096) for _ in range(w - len(lin))]
        tail = solve([[cf(X) for X in Xs[len(yt):]] for cf, _ in lin],
                     [rhs ^ T._xor(gf.mul(y, cf(X)) for y, X in zip(yt, Xs)) for cf, rhs in lin])
        if tail and all(tail):
            return {gf.LOG[X]: gf.div(y, gf.mul(X, gf.peval(M, X))) for X, y in zip(Xs, yt + tail)}


def node_value_is(g, value):
    """H(xi_g) = value (g 0-based) as two F-linear constraints on the Yt (halves of 1/(xi_g - X))."""
    return [(lambda X: T.kinv(T.XI[g] ^ X) & 4095, value & 4095),
            (lambda X: T.kinv(T.XI[g] ^ X) >> 12, value >> 12)]


def full_decode(err):
    S = T.syndromes(err)
    h = T.node_values(S, M)
    (c, pole), (Q, R) = T.extract(h), reference.decode(S, M)[:2]
    return h, c, pole, Q, R


def test_k_second_path_and_edge_values():
    def pw(a, e):
        r = 1
        while e:
            if e & 1:
                r = kmul_ref(r, a)
            a, e = kmul_ref(a, a), e >> 1
        return r
    rng = random.Random(21)
    vals = [0, 1, 2, 4095, 4096, 4097, T.U | 1, T.C0, T.C0 << 12, 4095 << 12, (1 << 24) - 1]
    vals += [rng.randrange(1 << 24) for _ in range(40)]
    for a in vals:
        assert all(T.kmul(a, b) == kmul_ref(a, b) for b in vals)
        assert T.kinv(a) == pw(a, (1 << 24) - 2) and T.kconj(a) == pw(a, 1 << 12)   # Fermat; 0 -> 0
        assert T.knorm(a) == kmul_ref(a, T.kconj(a)) < 4096
        assert all(T.kscale(s, a) == kmul_ref(s, a) for s in (0, 1, 4095, a & 4095))
    km = pytest.importorskip('kickmix')                        # trace convention of the contest field
    f, tr = km.GF2Field(12), list(range(4096))
    for _ in range(11):
        tr = [f.mul(t, t) ^ a for a, t in enumerate(tr)]
    assert [a for a in range(4096) if tr[a]] == list(range(2048, 4096)) == T.TR1 and set(tr) == {0, 1}
    assert all(gf.mul(t, t) ^ t ^ T.C0 for t in range(4096))   # u^2 + u + C0 has no root in F
    for k in range(T.NODES):                                   # x = a + b u is a root of m_k iff b^2 = b
        roots = [a | b << 12 for b in (0, 1) for a in range(4096)     # and a^2 + a + b C0 + gam_k = 0
                 if gf.mul(a, a) ^ a ^ b * T.C0 ^ T.GAM[k] == 0]
        assert len(roots) == 2 and min(roots) == T.XI[k]


def test_every_weight_against_definition_in_F():
    """Weights 0..35: words and flags equal the F[x]-only definition applied to (Q, R) built straight
    from the error; the low-weight pattern; flags and tail from the words alone; both synthesis orders."""
    rng = random.Random(22)
    for w in range(36):
        err = T.random_error(rng, w)
        h, c, pole, Q, R = full_decode(err)
        assert (Q, R) == direct_QR(err) and (c, pole) == cf_in_F(Q, R)
        assert all(c[:w]) and not any(c[w:]) and trailing_run(c) == 35 - w
        assert pole == [int(G > w and (G - w) % 2) for G in range(1, 36)]          # G = 1-based node
        tail = (35 - w) % 2
        assert T.tail_from_words(c) == tail == pole[-1]
        assert T.poles_from_tail(c) == pole == T.poles_from_tail(c, tail) and T.inverse(c, pole) == h
        N0, D0 = T.synthesize(c)
        assert (N0, D0) == T.synthesize(c, tail) and gf.pdeg(N0) == 35 - tail
        P = forward_product(c)                                 # tail vector (1; 0), or (0; 1) = phi_35
        m35 = T.m_poly(34)                                     # when phi_34 = infinity (tail = 1)
        assert (P[0][tail], P[1][tail]) == ((N0, D0) if not tail else (gf.pmul(m35, N0), gf.pmul(m35, D0)))
        f = [1]
        for G in range(w + 1, 35, 2):
            f = gf.pmul(f, T.m_poly(G - 1))
        assert (N0, D0) == (gf.pscale(gf.pmul(f, Q), N0[-1]), gf.pscale(gf.pmul(f, R), N0[-1]))
        assert T.normalise(N0, D0, pole) == (Q, R) and T.error_values(N0, D0, err, M) == err
        assert not gf.pmod(gf.padd(gf.pmul(N0, reference.node_poly(T.syndromes(err), M)), D0), M)   # (KEY)
        for g, x in enumerate(T.XI):                           # f(xi_g) = 0 at the poles g <= 34: there
            n, d = T.keval(N0, x), T.keval(D0, x)              # h_g is NOT D_0(xi_g)/N_0(xi_g) (0/0)
            assert (n == 0 == d) if pole[g] and g < 34 else (n != 0 and d == T.kmul(h[g], n))
        if w in (0, 7, 34, 35):                                # roots of N_0 in F* = the error positions
            assert [j for j in range(gf.ORDER) if not gf.peval(N0, gf.EXP[j])] == sorted(err)


# A random error (random.Random(31337) stream of the review run, weight 32) with a natural accidental
# pole at node 9: B_8(xi_9) = 0.  Found in about 16,000 scalar decodes (a 2^-24 event per node).
NATURAL_POLE = {522: 3877, 3279: 2863, 1443: 3370, 2850: 2848, 2656: 496, 948: 1269, 1500: 1053,
                1730: 1366, 1334: 2968, 1602: 2866, 3176: 3797, 258: 541, 2101: 2880, 421: 447,
                432: 4040, 716: 2845, 148: 293, 2589: 2104, 3379: 17, 4027: 2913, 2116: 864, 1540: 1896,
                3920: 1804, 2511: 1497, 1336: 765, 3172: 2774, 2354: 1526, 2573: 2781, 3121: 2935,
                3885: 2815, 1674: 33, 80: 886}


def check_absorbed_pole(err, g):
    """An accidental pole at the 1-based node g of a real error with w <= 33: the forward direction
    needs no flag, the inverse needs pole_g unless g = 1.  Returns the nodes mis-restored without it."""
    w = len(err)
    h, c, pole, Q, R = full_decode(err)
    assert (c, pole) == cf_in_F(Q, R) and T.inverse(c, pole) == h
    assert [G for G in range(1, 36) if pole[G - 1]] == [g] + list(range(w + 3, 36, 2))
    assert [G for G in range(1, 36) if not c[G - 1]] == [g] + list(range(w + 3, 36))   # c_(w+2) != 0
    assert T.tail_from_words(c) == pole[-1] == (35 - w) % 2
    N0, D0 = T.synthesize(c)
    f = gf.pmul(T.m_poly(g - 1), T.common_factor([0] * (w + 2) + pole[w + 2:]))
    assert N0 == gf.pscale(gf.pmul(f, Q), N0[-1]) and gf.pdeg(N0) == 35 - pole[-1]
    assert T.normalise(N0, D0, pole) == (Q, R) and T.error_values(N0, D0, err, M) == err
    guess = T.poles_from_tail(c)
    assert [G for G in range(1, 36) if guess[G - 1] != pole[G - 1]] == [g] * (g > 1)
    return [G for G, (x, y) in enumerate(zip(T.inverse(c, guess), h), 1) if x != y]


def test_accidental_poles_on_real_errors():
    rng = random.Random(23)
    assert len(NATURAL_POLE) == 32 and check_absorbed_pole(NATURAL_POLE, 9) == [9]
    assert T.conforms(NATURAL_POLE, M) == (True, 1)            # one extra zero state: D = 0 at (9, 8)
    for w in (16, 33):                                         # pole at node 1  <=>  h_1 = 0
        err = planted_error(rng, w, node_value_is(0, 0))
        assert check_absorbed_pole(err, 1) == []               # c_1 = 0 can only be a pole: exact
    for w in (17, 33, 34, 35):                                 # pole at node 2  <=>  nu_1(xi_2) = 1/h_2
        h1 = rng.randrange(1, 1 << 24)
        want = T.kinv(T.nu_at(T.kinv(h1), 0, 1))
        err = planted_error(rng, w, node_value_is(0, h1) + node_value_is(1, want))
        if w <= 33:
            assert check_absorbed_pole(err, 2) == [2]          # only h_2 is mis-restored without flag
            continue
        h, c, pole, Q, R = full_decode(err)                    # w >= 34: 35 words are not enough
        assert pole[:2] == [0, 1] and T.inverse(c, pole) == h and (c, pole) == cf_in_F(Q, R)
        N0, D0 = T.synthesize(c)
        assert gf.padd(gf.pmul(N0, R), gf.pmul(D0, Q))         # not proportional: decode failure
        ok, extra = T.conforms(err, M)                         # still reversible with its flags
        assert ok and extra >= 1
    for w in (34, 35):                                         # same failure with the pole at node 1
        h, c, pole, Q, R = full_decode(planted_error(rng, w, node_value_is(0, 0)))
        N0, D0 = T.synthesize(c)
        assert pole[0] and T.inverse(c, pole) == h and gf.padd(gf.pmul(N0, R), gf.pmul(D0, Q))


def test_harmless_degeneracies_on_real_errors():
    """Accidental zero word (h_2 = 0 => phi_1(xi_2) = 0) and deg R < w - 1 on real errors."""
    rng = random.Random(24)
    cases = [(w, node_value_is(1, 0)) for w in (16, 30, 35)]
    cases += [(w, [((lambda X, i=i: gf.power(X, i)), 0) for i in range(t)])        # R loses t leading
              for w, t in ((35, 1), (35, 4), (34, 2), (16, 3))]                    # coefficients
    for n, (w, lin) in enumerate(cases):
        err = planted_error(rng, w, lin)
        h, c, pole, Q, R = full_decode(err)
        if n < 3:
            assert h[1] == 0 and c[1] == 0 and all(c[:1] + c[2:w])
        else:
            assert gf.pdeg(R) < w - 1 and all(c[:w])
        assert pole == [int(G > w and (G - w) % 2) for G in range(1, 36)] == T.poles_from_tail(c)
        assert (c, pole) == cf_in_F(Q, R) and T.inverse(c, pole) == h
        N0, D0 = T.synthesize(c)
        assert T.normalise(N0, D0, pole) == (Q, R) and T.error_values(N0, D0, err, M) == err


def test_arbitrary_words_against_euclid():
    """words -> inverse -> node values -> interpolated N (CRT over the m_k) -> reference Euclid: the
    same rational function as synthesize(words), with b_k = 0 everywhere and interior zero words."""
    rng = random.Random(25)
    idem = []
    for k in range(T.NODES):                                   # e_k = 1 mod m_k, 0 mod the other m_j
        cof = gf.pdivmod(M, T.m_poly(k))[0]
        idem.append(gf.pmul(cof, gf.pinv_mod(cof, T.m_poly(k))))
    for trial in range(12):
        c = [rng.randrange(1, 1 << 24) for _ in range(T.NODES)]
        if trial % 3 == 1:
            c = [x & 4095 if rng.random() < 0.3 else x for x in c]     # some b_k = 0
        if trial % 3 == 2:
            c = [x & 4095 or 1 for x in c]                             # every b_k = 0: deg N_0 < 35
        if trial % 4 == 3:
            for g in rng.sample(range(1, 34), 3):                      # interior ordinary zero words
                c[g] = 0                                               # (node 1: 0 would be a pole)
        h = T.inverse(c, [0] * T.NODES)
        assert T.extract(h) == (c, [0] * T.NODES)
        N = []
        for k in range(T.NODES):                               # h_k = alpha + beta xi_k
            lin = T.nu_poly(h[k], k)
            N = gf.padd(N, gf.pmod(gf.pmul(lin, idem[k]), M))
        assert [T.keval(N, x) for x in T.XI] == h
        N0, D0 = T.synthesize(c)
        Q, R = reference.solve_key(N, M)
        assert not gf.padd(gf.pmul(N0, R), gf.pmul(D0, Q)) and gf.pdeg(N0) <= 35 and gf.pdeg(D0) <= 34
        assert (gf.pdeg(N0) < 35) == (trial % 3 == 2)


def test_edge_node_values():
    for h in ([0] * 35, [1] * 35, [(1 << 24) - 1] * 35, [0] + [1] * 34, [1, 0] * 17 + [T.U]):
        c, pole = T.extract(h)
        assert T.inverse(c, pole) == h and not any(p and x for p, x in zip(pole, c))
    c, pole = T.extract([0] * 35)                              # weight 0: Q = 1, R = 0
    assert c == [0] * 35 and pole == [1, 0] * 17 + [1] == T.poles_from_tail(c)
    N0, D0 = T.synthesize(c)
    assert D0 == [] and T.normalise(N0, D0, pole) == ([1], []) and gf.pdeg(N0) == 34
    assert pole[0] == T.poles_from_tail([0] + [5] * 34)[0] == 1    # c_1 = 0 is always a pole


def test_numpy_census_matches_scalar_model():
    np = pytest.importorskip('numpy')
    rng = random.Random(26)
    hs = [T.node_values(T.syndromes(e), M) for e in
          [NATURAL_POLE] + [T.random_error(rng, w) for w in (0, 1, 2, 16, 33, 34, 35, 35)]]
    hs += [[rng.choice((0, 1, (1 << 24) - 1, rng.randrange(1 << 24))) for _ in range(35)]
           for _ in range(20)]
    words, pole, zc = T.extract_np(np.array(hs).T)
    zeros = []
    for i, h in enumerate(hs):
        st = {}
        assert T.extract(h, st) == (words[:, i].tolist(), pole[:, i].astype(int).tolist())
        zeros += [(g, s) for g, s, _, d in st['zeros'] if d]
    assert zc.sum() == len(zeros) and all(zc[g, s] == zeros.count((g, s)) for g, s in set(zeros))
    res = T.zero_census(35, 300, seed=1)
    assert 200 < res['decodes'] <= 300 and res['states'] == 627 == sum(35 - s for s in range(33))
    assert (res['dzero'], res['poles'], res['zero_words'], res['errors']) == (0, 0, 0, [])
    assert T.zero_census(0, 50)['states'] == 0 == T.zero_census(2, 50)['dzero']

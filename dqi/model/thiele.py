"""Classical model of in-place rational reconstruction by a Thiele-type continued fraction (CF)
with quadratic nodes.  No circuits here: the purpose is to pin down exact conventions.

Fields.  F = GF(2^12) (model/gf.py).  K = GF(2^24) = F[u]/(u^2 + u + C0), C0 = smallest element of
F with absolute trace 1 (= 2048); a K element is the int  lo | hi << 12  = lo + hi*u.

Nodes (text: k = 1..35; code: index k-1).  m_k(x) = x^2 + x + GAM[k], GAM = the 35 smallest
trace-1 elements of F; xi_k = root of m_k in K with the smaller integer value = u + TAU[k] with
TAU[k] in F even.  M = prod m_k.  Two identities make every chain constant an F-linear map:
    m_k(xi_g) = GAM[k] + GAM[g]  (in F*, g != k),      xi_k + xi_g = TAU[k] + TAU[g]  (in F).

Data.  h_g = H(xi_g) = N(xi_g), H = R/Q of DESIGN.md (Q monic, deg Q = w = error weight).

CF variant.  phi_0 := 1/H = Q/R, and for k >= 1
    nu_k(x) = a_k + b_k x in F[x]  with  nu_k(xi_k) = phi_(k-1)(xi_k) =: c_k  (the stored word),
    phi_k   = m_k / (phi_(k-1) - nu_k),          i.e.  phi_(k-1) = nu_k + m_k / phi_k.
(a_k, b_k) are the coordinates of c_k in the F-basis (1, xi_k):  b_k = hi(c_k),
a_k = lo(c_k) + hi(c_k)*TAU[k];  nu_k(xi_g) = c_k + b_k*(TAU[k] + TAU[g]).
Why 1/H: with phi = A/B in lowest terms put d(phi) = max(deg A, deg B + 1).  A regular level
lowers d by exactly 1 and d(Q/R) = w, so phi_w = infinity identically: weight 35 uses exactly the
35 nodes and the tail is the constant (1 : 0).  d(R/Q) = w + 1 would need a 36th node.

Lazy extraction.  Node g starts from the projective point (R : D) = (1 : h_g) = phi_0(xi_g) and
applies steps k = 1..g-1,  (R, D) <- (m_k(xi_g)*D, R + nu_k(xi_g)*D)  (one K-multiplication each),
then c_g = R/D.  Each step is an invertible projective map, so (R, D) != (0, 0) always and zeros
in the middle of a chain are harmless.  Only the end of the chain matters:
    D = 0  ("pole", phi_(g-1)(xi_g) = infinity):  store c_g = 0 and set the flag pole_g = 1;
    R = 0:  c_g = 0 is an ordinary word, pole_g = 0.
A zero word means nu_k = 0, so later chains apply the uniform step (R, D) <- (m_k(xi_g)*D, R) for
both kinds; no step is ever conditioned on a flag.  kinv(0) = 0 makes "store 0" automatic.

Low weight w < 35 (no accident): phi_w = infinity, hence at node g > w the chain ends with
D = 0, R != 0 if g - w is odd (pole) and with R = 0, D != 0 if g - w is even; all words beyond w
are 0 and pole_g = [g > w and g - w odd].  The last regular word is never 0 (phi_(w-1) = A/B with
B a nonzero constant and A != 0 of degree <= 1 has no root outside F), so the trailing run of zero
words is exactly w+1..35 and NO stored flag is needed: tail := pole_35 = (35 - w) mod 2 is the
parity of the length of that run (tail_from_words), and the flags follow (poles_from_tail).

Synthesis.  Tail recurrence (N_(k-1), D_(k-1)) = (nu_k N_k + m_k D_k, N_k), k = 34..1, started
from (N_34, D_34) = (nu_35 + tail, 1 + tail)   [tail = 0: phi_35 = infinity; tail = 1: phi_34 =
infinity].  Result (N_0, D_0) = lam * f * (Q, R) with lam = lc(N_0) in F* and
f = prod of m_k over the poles k <= 34 (for weight w: m_(w+1) m_(w+3) ... with index <= 34), so
deg N_0 = 35 - tail.  f has no root in F and R/Q' = (fR)/(fQ)' at the roots of Q, so the
unnormalised pair can be scanned directly.  The x^35 coefficient of N_0 is the continuant of
(b_1..b_35): l_(k-1) = b_k l_k + l_(k+1), l_35 = 1 + tail, l_34 = b_35.

Accidents (2^-24 per chain state, measured with zero_census; one natural and several planted
real errors are in the tests).  A pole at a node g <= w (possible for g <= w - 2 only: it needs
m_g | B, deg B <= w - g) raises d by 1 instead of lowering it, so the CF ends at level L = w + 2:
for w <= 33 nothing breaks in the forward direction (f just gains the factor m_g, c_L != 0, and
tail is still the parity of the trailing zero run), for w >= 34 the 35 words no longer synthesize
(decode failure).  The words alone cannot tell such a pole from a zero word, except at node 1
where c_1 = 1/h_1 = 0 is always a pole: inverse() with flags rebuilt from the words mis-restores
exactly the node value h_g of an accidental pole at g >= 2.  An accidental zero word (R = 0 at the
end of a chain) is an ordinary word.  b_k = 0 or deg R < w - 1 (2^-12) are not special at all.
"""
from __future__ import annotations

import random

from . import gf, reference

NODES = 35
LOW = (1 << gf.B) - 1


def trace(a: int) -> int:
    """Absolute trace F -> GF(2)."""
    t = 0
    for _ in range(gf.B):
        t ^= a
        a = gf.mul(a, a)
    return t


TR1 = [a for a in range(1 << gf.B) if trace(a)]
C0 = TR1[0]                 # K = F[u]/(u^2 + u + C0)
GAM = TR1[:NODES]           # m_k = x^2 + x + GAM[k]
U = 1 << gf.B               # the K element u


# ---- K = GF(2^24) ----

def kmul(a: int, b: int) -> int:
    """Product in K (3 F-multiplications, Karatsuba)."""
    a0, a1, b0, b1 = a & LOW, a >> gf.B, b & LOW, b >> gf.B
    p0, p1 = gf.mul(a0, b0), gf.mul(a1, b1)
    return (p0 ^ gf.mul(C0, p1)) | ((gf.mul(a0 ^ a1, b0 ^ b1) ^ p0) << gf.B)


def kscale(s: int, a: int) -> int:
    """s*a for s in F, a in K (an F-linear map when s is classical)."""
    return gf.mul(s, a & LOW) | (gf.mul(s, a >> gf.B) << gf.B)


def kconj(a: int) -> int:
    """Frobenius a -> a^(2^12): u -> u + 1."""
    return (a ^ (a >> gf.B)) & LOW | (a >> gf.B << gf.B)


def knorm(a: int) -> int:
    """a * conj(a), an element of F."""
    a0, a1 = a & LOW, a >> gf.B
    return gf.mul(a0, a0 ^ a1) ^ gf.mul(C0, gf.mul(a1, a1))


def kinv(a: int) -> int:
    """Inverse in K, with the total convention kinv(0) = 0."""
    return kscale(gf.inv(knorm(a)), kconj(a)) if a else 0


def keval(p, x: int) -> int:
    """Evaluates a polynomial with coefficients in F at a point of K."""
    r = 0
    for c in reversed(p):
        r = kmul(r, x) ^ c
    return r


# ---- nodes ----

_HALF = {gf.mul(t, t) ^ t: t for t in range(0, 1 << gf.B, 2)}    # t^2 + t -> the even solution t
TAU = [_HALF[C0 ^ g] for g in GAM]                                 # xi_k = u + TAU[k]
XI = [U | t for t in TAU]


def m_poly(k: int):
    return [GAM[k], 1, 1]


def modulus():
    """M = prod m_k (monic, degree 70, no root in F)."""
    M = [1]
    for k in range(NODES):
        M = gf.pmul(M, m_poly(k))
    return M


def nu_poly(word: int, k: int):
    """nu_k = a_k + b_k x, the degree <= 1 polynomial over F with nu_k(xi_k) = word."""
    b = word >> gf.B
    return gf.ptrim([(word & LOW) ^ gf.mul(b, TAU[k]), b])


def nu_at(word: int, k: int, g: int) -> int:
    """nu_k(xi_g) in K: an F-linear function of the word (classical constant TAU[k] + TAU[g])."""
    return word ^ gf.mul(word >> gf.B, TAU[k] ^ TAU[g])


# ---- ground truth helpers (errors are dicts {position: nonzero value}) ----

def _xor(it):
    r = 0
    for v in it:
        r ^= v
    return r


def random_error(rng: random.Random, w: int):
    return {j: rng.randrange(1, 1 << gf.B) for j in rng.sample(range(gf.ORDER), w)}


def syndromes(err):
    """S_i = sum_j e_j X_j^(i+1), i = 0..69, X_j = gf.EXP[j]."""
    return [_xor(gf.mul(v, gf.power(gf.EXP[j], i + 1)) for j, v in err.items())
            for i in range(reference.NS)]


def node_values(S, M=None):
    """h_k = N(xi_k) with N = reference.node_poly(S, M): the node values as computed from syndromes."""
    N = reference.node_poly(S, modulus() if M is None else M)
    return [keval(N, x) for x in XI]


def node_values_direct(err, M=None):
    """h_k = H(xi_k) = sum_l Yt_l / (xi_k - X_l), Yt_l = Y_l X_l M(X_l): the definition of H."""
    M = modulus() if M is None else M
    yt = [(gf.EXP[j], gf.mul(gf.mul(v, gf.EXP[j]), gf.peval(M, gf.EXP[j]))) for j, v in err.items()]
    return [_xor(kscale(y, kinv(x ^ X)) for X, y in yt) for x in XI]


# ---- the continued fraction ----

def extract(h, stats=None):
    """Node values h[35] -> (words c[35], pole flags[35]) by the lazy chains of the module docstring.

    Total on K^35 and inverted by inverse().  If `stats` is a dict it receives the operation counts
    'mul' / 'inv' (K-multiplications by data, K-inversions; scaling by classical F constants is
    free) and 'zeros': tuples (g, s, R == 0, D == 0) for every chain state with a zero coordinate,
    g = 0-based node, s = number of steps already applied (s = g is the end of the chain).
    """
    c, pole, zeros, mul = [0] * NODES, [0] * NODES, [], 0
    for g in range(NODES):
        R, D = 1, h[g]                                   # phi_0(xi_g) = 1/h_g
        for k in range(g + 1):
            if not (R and D):
                zeros.append((g, k, R == 0, D == 0))
            if k < g:                                    # step k: one K-multiplication
                R, D = kscale(GAM[k] ^ GAM[g], D), R ^ kmul(nu_at(c[k], k, g), D)
                mul += 1
        pole[g] = int(D == 0)
        c[g] = kmul(R, kinv(D))                          # pole -> word 0
        mul += g > 0                                     # at node 1, R is the classical constant 1
    if stats is not None:
        stats.update(mul=mul, inv=NODES, zeros=zeros)
    return c, pole


def inverse(c, pole, stats=None):
    """(words, pole flags) -> node values h[35]; nodes are undone in the order 35..1 (node g needs
    only c_1..c_g).  Chain steps are undone by (R, D) <- (nu_k(xi_g)*R + m_k(xi_g)*D, R)."""
    h, mul = [0] * NODES, 0
    for g in reversed(range(NODES)):
        R, D = (1, 0) if pole[g] else (c[g], 1)          # phi_(g-1)(xi_g)
        for k in reversed(range(g)):
            R, D = kmul(nu_at(c[k], k, g), R) ^ kscale(GAM[k] ^ GAM[g], D), R
            mul += 1
        if R == 0:
            raise ValueError('not in the image of extract: phi_0(xi_g) = 0')
        h[g] = kmul(D, kinv(R))                          # h_g = 1/phi_0(xi_g)
        mul += g > 0                                     # at node 1, D is the classical constant 1
    if stats is not None:
        stats.update(mul=mul, inv=NODES)
    return h


def tail_from_words(c) -> int:
    """tail = pole_35 from the words alone: the parity of the length of the trailing run of zero
    words.  Exact whenever the CF ends within the 35 nodes, i.e. on every decodable input (low
    weight and accidental poles with w <= 33 included): the last level L has d(phi_(L-1)) = 1,
    hence c_L != 0, and the nodes L+1..35 alternate pole / zero word starting with a pole."""
    run = 0
    while run < NODES and not c[NODES - 1 - run]:
        run += 1
    return run & 1


def poles_from_tail(c, tail=None):
    """Rebuilds the pole flags from the words (tail = None: tail_from_words) or from the words and
    the bit tail = pole_35: inside the trailing run of zero words the flags alternate backwards
    from `tail`; a zero word at node 1 is always a pole (c_1 = 1/h_1); elsewhere they are 0.
    Exact unless an accidental pole occurred at a node 2 <= g <= w (see DESIGN.md section 5)."""
    tail = tail_from_words(c) if tail is None else tail
    pole = [0] * NODES
    for g in reversed(range(NODES)):
        if c[g]:
            break
        pole[g] = tail ^ ((NODES - 1 - g) & 1)
    pole[0] |= int(not c[0])
    return pole


def synthesize(c, tail=None):
    """Words -> (N_0, D_0) = lam * f * (Q, R); see module docstring.  The bit tail = pole_35 may be
    passed explicitly; by default it is recomputed from the words (tail_from_words)."""
    tail = tail_from_words(c) if tail is None else tail
    N, D = gf.padd(nu_poly(c[-1], NODES - 1), [tail]), gf.ptrim([1 ^ tail])
    for k in reversed(range(NODES - 1)):
        N, D = gf.padd(gf.pmul(nu_poly(c[k], k), N), gf.pmul(m_poly(k), D)), N
    return N, D


def common_factor(pole):
    """f = prod of m_k over the poles k <= 34 (a pole at node 35 contributes nothing)."""
    f = [1]
    for k in range(NODES - 1):
        if pole[k]:
            f = gf.pmul(f, m_poly(k))
    return f


def normalise(N0, D0, pole):
    """Divides out f and the leading coefficient: (N_0, D_0) -> the exact (Q, R), Q monic."""
    f = gf.pscale(common_factor(pole), N0[-1])
    (Q, r0), (R, r1) = gf.pdivmod(N0, f), gf.pdivmod(D0, f)
    assert not r0 and not r1, 'pair is not a multiple of the common factor'
    return Q, R


def error_values(N0, D0, positions, M=None):
    """Forney on the unnormalised pair: e_j = D_0(X_j) / (X_j M(X_j) N_0'(X_j)) at the given roots."""
    M = modulus() if M is None else M
    dN = gf.pderiv(N0)
    out = {}
    for j in positions:
        x = gf.EXP[j]
        assert gf.peval(N0, x) == 0, 'not a root of N_0'
        out[j] = gf.div(gf.peval(D0, x), gf.mul(gf.mul(x, gf.peval(M, x)), gf.peval(dN, x)))
    return out


# ---- experiment of DESIGN.md section 5 ----

def expected_zeros(w: int):
    """Zero pattern of the chains for weight w without accidents: the state after s steps at
    node g has D = 0 iff s >= w and s - w even, R = 0 iff s >= w and s - w odd."""
    return [(g, s, (s - w) & 1 == 1, (s - w) & 1 == 0)
            for g in range(NODES) for s in range(g + 1) if s >= w]


def conforms(err, M=None):
    """Checks every documented convention on one error against reference.decode.  Returns
    (ok, number of accidental zero states).  Without accident the exact low-weight pattern must
    appear, with the flags rebuilt from the words; with an accident the round trip with the true
    flags must still hold; whenever the CF ends within the 35 nodes the words must decode."""
    M = modulus() if M is None else M
    w, S, st = len(err), syndromes(err), {}
    h = node_values(S, M)
    c, pole = extract(h, st)
    exp = expected_zeros(w)
    extra = set(st['zeros']) - set(exp)
    ok = inverse(c, pole) == h == node_values_direct(err, M)
    if not extra:
        ok = (ok and st['zeros'] == exp and all(c[:w]) and not any(c[w:])
              and pole == [int(g >= w and (g - w) % 2 == 0) for g in range(NODES)]
              and pole == poles_from_tail(c) == poles_from_tail(c, pole[-1]))
    last = max((g + 1 for g in range(NODES) if c[g]), default=0)    # last nonzero word (1-based)
    if last == w + 2 * sum(pole[:last]):                            # the CF ended at level `last`
        Q, R, _ = reference.decode(S, M)
        N0, D0 = synthesize(c)
        ok = (ok and pole[-1] == tail_from_words(c) and (N0, D0) == synthesize(c, pole[-1])
              and gf.pdeg(N0) == NODES - pole[-1] and gf.pdeg(Q) == w
              and normalise(N0, D0, pole) == (Q, R) and error_values(N0, D0, err, M) == err)
    return ok, len(extra)


def experiment(weights=range(16, 36), trials=200, seed=2026):
    """conforms() on `trials` random errors per weight; returns
    {w: (trials, conforming, accidental zero states)}."""
    rng, M, out = random.Random(seed), modulus(), {}
    for w in weights:
        res = [conforms(random_error(rng, w), M) for _ in range(trials)]
        out[w] = (trials, sum(ok for ok, _ in res), sum(n for _, n in res))
    return out


# ---- vectorised census of accidental zeros (numpy; the scalar model above is the specification) ----

def _np_field():
    """(numpy, mul, inv): table arithmetic of F on integer arrays, with x * 0 = 0 and inv[0] = 0."""
    import numpy as np
    exp = np.zeros(4 * gf.ORDER + 1, dtype=np.int64)
    exp[:2 * gf.ORDER] = gf.EXP
    log = np.array(gf.LOG, dtype=np.int64)
    log[0] = 2 * gf.ORDER                               # products with 0 land in the zero tail of exp
    inv = np.array([0] + [gf.inv(a) for a in range(1, gf.ORDER + 1)], dtype=np.int64)
    return np, (lambda a, b: exp[log[a] + log[b]]), inv


def extract_np(h):
    """extract() on an integer array h[35, shots] of node values.  Returns (words[35, shots],
    pole[35, shots] (bool), zc[35, 35]) with zc[g, s] = number of shots whose chain state (g, s)
    has D = 0 (indices as in extract's 'zeros'; R = 0 at (g, s) is the same event as D = 0 at
    (g, s - 1), so only D is counted)."""
    np, mul, inv = _np_field()
    h = np.asarray(h, dtype=np.int64)

    def kmulv(a, b):
        p0, p1 = mul(a[0], b[0]), mul(a[1], b[1])
        return p0 ^ mul(C0, p1), mul(a[0] ^ a[1], b[0] ^ b[1]) ^ p0

    c, pole, zc = [], [], np.zeros((NODES, NODES), dtype=np.int64)
    for g in range(NODES):
        R, D = (np.ones_like(h[g]), np.zeros_like(h[g])), (h[g] & LOW, h[g] >> gf.B)
        for k in range(g):
            zc[g, k] = ((D[0] | D[1]) == 0).sum()
            p = kmulv((c[k][0] ^ mul(c[k][1], TAU[k] ^ TAU[g]), c[k][1]), D)
            s = GAM[k] ^ GAM[g]
            R, D = (mul(s, D[0]), mul(s, D[1])), (R[0] ^ p[0], R[1] ^ p[1])
        pole.append((D[0] | D[1]) == 0)
        zc[g, g] = pole[g].sum()
        n = inv[mul(D[0], D[0] ^ D[1]) ^ mul(C0, mul(D[1], D[1]))]          # 1/norm(D); 0 for D = 0
        c.append(kmulv(R, (mul(n, D[0] ^ D[1]), mul(n, D[1]))))
    return np.array([a | b << gf.B for a, b in c]), np.array(pole), zc


def zero_census(w=NODES, shots=100000, seed=0, keep=3):
    """Accidental zeros over random weight-w errors (locators drawn with replacement, shots with a
    repeated locator dropped; scaled values Yt uniform in F*), by extract_np.  Returns a dict:
      decodes    number of errors decoded,
      states     chain states (g, s) per decode in which an accidental D = 0 is possible
                 (s <= g and s <= w - 3: it needs m_g | B_s with deg B_s <= w - s - 1),
      dzero      D = 0 events in those states (expected: decodes * states * 2^-24),
      poles      of these, at the end of a chain: accidental poles (nodes g <= w - 2),
      zero_words accidental zero words (R = 0 at the end of a chain g <= w),
      errors     the first `keep` errors with an accidental pole, as {position: value}."""
    np, mul, _ = _np_field()
    rng = np.random.default_rng(seed)
    X = rng.integers(1, gf.ORDER + 1, size=(w, shots))
    srt = np.sort(X, axis=0)
    X = X[:, (srt[1:] != srt[:-1]).all(axis=0)]
    Y = rng.integers(1, gf.ORDER + 1, size=X.shape)
    tab = np.array([[kinv(x ^ a) for a in range(gf.ORDER + 1)] for x in XI])   # 1/(xi_g - X)
    h = np.array([np.bitwise_xor.reduce(mul(Y, tab[g][X] & LOW) | mul(Y, tab[g][X] >> gf.B) << gf.B,
                                        axis=0) for g in range(NODES)])
    _, pole, zc = extract_np(h)
    reg = np.array([[s <= g and s < w for s in range(NODES)] for g in range(NODES)])
    M = modulus()
    errs = [{gf.LOG[x]: gf.div(y, gf.mul(x, gf.peval(M, x)))            # Y_l = Yt_l / (X_l M(X_l))
             for x, y in zip(X[:, i].tolist(), Y[:, i].tolist())}
            for i in np.flatnonzero(pole[:w].any(axis=0))[:keep]]
    return dict(decodes=X.shape[1], states=sum(NODES - s for s in range(max(w - 2, 0))),
                dzero=int(zc[reg].sum()), poles=int(pole[:w].sum()),
                zero_words=int(sum(zc[g, g - 1] for g in range(1, w))), errors=errs)


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['census']:             # python3 -m model.thiele census <weight> <decodes> [seed]
        w, total, tot = int(sys.argv[2]), int(sys.argv[3]), dict(decodes=0, dzero=0, poles=0, zero_words=0)
        base = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        while tot['decodes'] < total:
            res = zero_census(w, 100000, seed=base + tot['decodes'])
            tot = {k: v + res[k] for k, v in tot.items()}
        print(f"weight {w}: {tot}, expected dzero {tot['decodes'] * res['states'] / 2 ** 24:.1f}, "
              f"expected poles {tot['decodes'] * max(w - 2, 0) / 2 ** 24:.1f}")
    else:                                       # python3 -m model.thiele <trials per weight>
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
        for w, (t, ok, acc) in experiment(trials=n).items():
            print(f'weight {w}: {ok}/{t} conform, {acc} accidental zero states')

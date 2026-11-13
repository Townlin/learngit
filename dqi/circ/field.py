"""Finite-field gate emitters for kickmix registers (little-endian: bit i of the integer = qubit i).

F = GF(2^12) is model.gf (kickmix.GF2Field(12)); K = GF(2^24) = F[U]/(U^2 + U + C), ints lo + (hi << 12).

A *scheme* is a list of mask triples (alpha, beta, gamma) standing for the map
    (x, y) -> XOR_k gamma_k * (parity(alpha_k & x) AND parity(beta_k & y)).
mul_xor spends exactly one Toffoli-like gate per triple and needs no ancilla; everything GF(2)-linear is
CX-only, and a register that holds scheme(x, y) of live registers is erased for free by erase_product.
Value preconditions (t clean for inv_init, t == x^-1 for inv_del / erase=True, t == scheme(x, y) for
erase_product) cannot be checked at build time: a violation shows up as a phase flip or left-over state.
The emitters take no control argument, but they may be wrapped in push_condition / pop_condition.

Toffoli-like counts / scratch qubits (besides the operands; +1 with anc=True):
    multiply   F 42 / 0        K 99 / 0                      (mul_xor, erase_product is free)
    inverse    F 66 / 6        K 192 / 12    into a clean t  (inv_init)
               F 66 / 18       K 192 / 42    t ^= x^-1       (inv12_compute, inv24_compute; K lean: 216 / 30)
               F 66 / 30       K 192 / 54    in place        (inv_inplace; lean: 75 / 24 and 231 / 36)
    un-invert  F 39 / 6        K 123 / 12    t = x^-1 -> 0   (inv_del;  erase=True of *_compute: 36 / 18, 108 / 42)
Inversion uses the tower K > F > GF(64) > GF(8): x^-1 = conj(x) N(x)^-1 with the norm N a quadratic map
(one half-size product + linear terms) and 3 ANDs for the inverse in GF(8).  Besides "product of live
registers", the free erasures use N(x^-1) = N(x)^-1: the inverse of a norm is the norm of the inverse.
"""
from __future__ import annotations

import kickmix as km
from model import gf

mul12, G = gf.mul, gf.G
UNIT12 = [1 << i for i in range(12)]
UNIT24 = [1 << i for i in range(24)]


# ------------------------------------------------------------------ bits and GF(2)-linear maps

def bits(m):
    """Indices of the set bits of m, ascending."""
    out = []
    while m:
        low = m & -m
        out.append(low.bit_length() - 1)
        m ^= low
    return out


def par(v):
    return v.bit_count() & 1


def lin(cols, v):
    """Apply the linear map with columns cols (cols[j] = image of bit j) to the vector v."""
    r = 0
    for j in bits(v):
        r ^= cols[j]
    return r


def pull(cols, mask):
    """Pull a functional back through a map: parity(mask & lin(cols, v)) == parity(pull(cols, mask) & v)."""
    return sum(par(mask & c) << j for j, c in enumerate(cols))


def coord_fn(basis):
    """crd(e) = mask c with XOR of basis[i] over bits(c) == e (KeyError outside the span; basis must be free)."""
    piv = {}
    for i, v in enumerate(basis):
        c = 1 << i
        while v and v.bit_length() in piv:
            pv, pc = piv[v.bit_length()]
            v, c = v ^ pv, c ^ pc
        if not v:
            raise ValueError('dependent basis')
        piv[v.bit_length()] = (v, c)

    def crd(e):
        c = 0
        while e:
            pv, pc = piv[e.bit_length()]
            e, c = e ^ pv, c ^ pc
        return c
    return crd


# ------------------------------------------------------------------ classical field helpers

def inv12(a):
    """a^-1 in F with 0 -> 0 (what the circuits compute)."""
    return gf.inv(a) if a else 0


def trace12(a):
    """Absolute trace F -> GF(2)."""
    r = 0
    for _ in range(12):
        r, a = r ^ a, mul12(a, a)
    return r


C = next(c for c in range(1, 4096) if trace12(c))     # U^2 = U + C, smallest trace-1 element of F
U = 1 << 12


def mul24(a, b):
    a0, a1, b0, b1 = a & 4095, a >> 12, b & 4095, b >> 12
    p0, p1 = mul12(a0, b0), mul12(a1, b1)
    return (p0 ^ mul12(C, p1)) | ((mul12(a0 ^ a1, b0 ^ b1) ^ p0) << 12)


def conj24(a):
    """Frobenius a -> a^(2^12): U -> U + 1."""
    return a ^ (a >> 12)


def norm24(a):
    """a * conj24(a) = lo^2 + lo*hi + C*hi^2, an element of F."""
    a0, a1 = a & 4095, a >> 12
    return mul12(a0, a0) ^ mul12(a0, a1) ^ mul12(C, mul12(a1, a1))


def inv24(a):
    """a^-1 in K with 0 -> 0: conj(a) / norm(a)."""
    n, c = inv12(norm24(a)), conj24(a)
    return mul12(c & 4095, n) | (mul12(c >> 12, n) << 12)


def pow24(a, e):
    r = 1
    while e:
        if e & 1:
            r = mul24(r, a)
        a, e = mul24(a, a), e >> 1
    return r


# ------------------------------------------------------------------ bilinear schemes

def scheme_eval(scheme, x, y):
    """Classical value of a scheme."""
    r = 0
    for al, be, ga in scheme:
        if par(al & x) & par(be & y):
            r ^= ga
    return r


def scheme_exact(scheme, mul, n):
    """Exact test on all n*n basis pairs (bilinearity makes this a proof for all inputs)."""
    return all(al and be and ga for al, be, ga in scheme) and all(
        scheme_eval(scheme, 1 << i, 1 << j) == mul(1 << i, 1 << j) for i in range(n) for j in range(n))


def tower_basis(mul, sb, t, d):
    """[s * t^j for j < d for s in sb]: GF(2)-basis of S[t] from a basis sb of the subfield S."""
    out, p = [], 1
    for _ in range(d):
        out += [mul(s, p) for s in sb]
        p = mul(p, t)
    return out


def eval_rows(mul, inv, t, d, pts):
    """Rows (u, W) of evaluation/interpolation at the 2d-2 points pts and at infinity: for polynomials
    x, y of degree < d over S,  x(t) y(t) = sum over rows of W * (sum_j u_j x_j) * (sum_j u_j y_j)."""
    assert len(set(pts)) == 2 * d - 2
    rows, w_inf = [], 1
    for p in pts:
        u, w = [1], 1
        for _ in range(d - 1):
            u.append(mul(u[-1], p))
        for q in pts:
            if q != p:
                w = mul(w, mul(t ^ q, inv(p ^ q)))
        rows.append((u, w))
        w_inf = mul(w_inf, t ^ p)
    return rows + [([0] * (d - 1) + [1], w_inf)]


def kar3_rows(mul, t):
    """Karatsuba rows for polynomials of degree < 3 over GF(2): 6 products."""
    p = [1]
    for _ in range(4):
        p.append(mul(p[-1], t))
    return [([1, 0, 0], p[0] ^ p[1] ^ p[2]), ([0, 1, 0], p[1] ^ p[2] ^ p[3]), ([0, 0, 1], p[2] ^ p[3] ^ p[4]),
            ([1, 1, 0], p[1]), ([1, 0, 1], p[2]), ([0, 1, 1], p[3])]


def lift(mul, sb, sub, rows):
    """Scheme of S[t] from a scheme `sub` of S (alpha/beta in sb-coordinates, gamma an element of the big
    field) and product rows (u, W).  Result: alpha/beta in tower_basis coordinates, gamma in the big field."""
    crd, out = coord_fn(sb), []
    for u, w in rows:
        ev = [crd(mul(uj, s)) for uj in u for s in sb]      # S-coordinates of sum_j u_j x_j, per input bit
        out += [(pull(ev, lam), pull(ev, mu), mul(w, nu)) for lam, mu, nu in sub]
    return out


def rebase(scheme, old, new, out=None):
    """Re-express alpha/beta from `old`-basis to `new`-basis coordinates; gamma (a field element) is turned
    into `out`-basis coordinates when out is given."""
    co, go = coord_fn(old), coord_fn(out) if out else (lambda e: e)
    cols = [co(e) for e in new]
    return [(pull(cols, al), pull(cols, be), go(ga)) for al, be, ga in scheme]


def linmap(fn, n):
    """Columns of a GF(2)-linear function on n-bit ints (for lin / pull / lin_xor / compose)."""
    return [fn(1 << i) for i in range(n)]


def compose(scheme, lx=None, ly=None, lt=None):
    """Scheme of (x, y) -> lt(scheme(lx(x), ly(y))) for linear maps given by columns: folding Frobenius maps,
    conjugations or constant factors into a product costs nothing."""
    return [(pull(lx, al) if lx else al, pull(ly, be) if ly else be, lin(lt, ga) if lt else ga)
            for al, be, ga in scheme]


ONE = [(1, 1, 1)]                                                      # GF(2)
W4, W8, H16, Z64 = (gf.power(G, 4095 // k) for k in (3, 7, 15, 63))    # generators of GF(4)*, GF(8)*, ...
B4, B8 = [1, W4], [1, W8, mul12(W8, W8)]
B16 = tower_basis(mul12, B4, H16, 2)
S4 = lift(mul12, [1], ONE, eval_rows(mul12, gf.inv, W4, 2, [0, 1]))    # rank 3, B4 coordinates
S8 = lift(mul12, [1], ONE, kar3_rows(mul12, W8))                       # rank 6, B8 coordinates
S16 = lift(mul12, B4, S4, eval_rows(mul12, gf.inv, H16, 2, [0, 1]))    # rank 9, B16 coordinates

# F = GF(8)[G]/(deg 4): 6 points of GF(8) + infinity, 7 x 6 = 42 products.
F42 = rebase(lift(mul12, B8, S8, eval_rows(mul12, gf.inv, G, 4, [0] + [gf.power(W8, i) for i in range(5)])),
             tower_basis(mul12, B8, G, 4), UNIT12)
assert len(F42) == 42 and scheme_exact(F42, mul12, 12)

# K = F[U]: Karatsuba (points 0, 1, infinity), 3 x 42 = 126 products.
K126 = lift(mul24, UNIT12, F42, eval_rows(mul24, inv24, U, 2, [0, 1]))
assert len(K126) == 126 and scheme_exact(K126, mul24, 24)


def _k99():
    """K = GF(16)[tau]/(deg 6): 10 points of GF(16) + infinity, 11 x 9 = 99 products."""
    for tau in range(U, 1 << 24):
        tb = tower_basis(mul24, B16, tau, 6)
        try:
            coord_fn(tb)
        except ValueError:
            continue                                                   # tau does not generate K over GF(16)
        pts = [0] + [gf.power(H16, i) for i in range(9)]
        return rebase(lift(mul24, B16, S16, eval_rows(mul24, inv24, tau, 6, pts)), tb, UNIT24)


K99 = _k99()
assert len(K99) == 99 and scheme_exact(K99, mul24, 24)
F_SCHEME, K_SCHEME = F42, min(K99, K126, key=len)


# ------------------------------------------------------------------ tower data for inversion
# F = GF(64)[U6], U6^2 = U6 + C6;  GF(64) = GF(8)[V], V^2 = V + C3.  With x = a + b U6 (a, b in GF(64)):
#   n = N(x) = a^2 + a b + C6 b^2 in GF(64);  n = n0 + n1 V;  d = N(n) = n0^2 + n0 n1 + C3 n1^2 in GF(8);
#   j = d^-1 (3 AND: every bit is one product of parities plus a parity);  i = n^-1 = ((n0+n1) j, n1 j);
#   x^-1 = conj(x) i = (a+b) i + (b i) U6.            Toffoli-like: 15 + 6 + 3 + 12 + 30 = 66.
# Registers of 6 / 3 qubits hold elements of GF(64) / GF(8) in B64 / B8 coordinates.

V = next(v for v in range(4096) if gf.power(v, 8) == v ^ 1)
U6 = next(u for u in range(4096) if gf.power(u, 64) == u ^ 1)
B64 = B8 + [mul12(s, V) for s in B8]
_c8, _c64 = coord_fn(B8), coord_fn(B64)
S8C = rebase(S8, B8, B8, B8)                                           # GF(8) product, all in B8 coordinates
S64C = rebase(lift(mul12, B4, S4, eval_rows(mul12, gf.inv, Z64, 3, [0, 1, W4, W4 ^ 1])),
              tower_basis(mul12, B4, Z64, 3), B64, B64)                # GF(64) product (rank 15), B64 coords
_ct = coord_fn(B64 + [mul12(s, U6) for s in B64])
_A, _B = [_ct(e) & 63 for e in UNIT12], [_ct(e) >> 6 for e in UNIT12]  # x -> coordinates of a, b


def _quad(f, n):
    """(L1, L2, L3) with f(d) == parity(L1 & d) parity(L2 & d) ^ parity(L3 & d) for all d < 2^n."""
    return next((l1, l2, l3) for l1 in range(1, 1 << n) for l2 in range(l1 + 1, 1 << n) for l3 in range(1 << n)
                if all(par(l1 & d) & par(l2 & d) ^ par(l3 & d) == f(d) for d in range(1 << n)))


_JQ = [_quad(lambda d, k=k: _c8(inv12(lin(B8, d))) >> k & 1, 3) for k in range(3)]

QN = [(pull(_A, al), pull(_B, be), ga) for al, be, ga in S64C]                         # n ^= a b       (x, x)
LN = [_c64(gf.power(e, 65) ^ mul12(lin(B64, a), lin(B64, b))) for e, a, b in zip(UNIT12, _A, _B)]
QD = [(al, be << 3, ga) for al, be, ga in S8C]                                         # d ^= n0 n1     (n, n)
LD = [_c8(gf.power(e, 9)) for e in B64]
QJ = [(l1, l2, 1 << k) for k, (l1, l2, _) in enumerate(_JQ)]                           # j ^= d^-1      (d, d)
LJ = [sum((l3 >> q & 1) << k for k, (_, _, l3) in enumerate(_JQ)) for q in range(3)]
QI = [(al | al << 3, be, ga) for al, be, ga in S8C] + [(al << 3, be, ga << 3) for al, be, ga in S8C]   # (n, j)
QT = ([(pull(_A, al) ^ pull(_B, al), be, lin(B64, ga)) for al, be, ga in S64C] +       # t ^= conj(x) i (x, i)
      [(pull(_B, al), be, mul12(lin(B64, ga), U6)) for al, be, ga in S64C])
# K level: nk = N(x) = lo hi + LK(x);  x^-1 = conj(x) m with m = nk^-1:  (lo+hi) m + (hi m) U.   42 + 66 + 84 = 192.
LK = [mul12(e, e) for e in UNIT12] + [mul12(C, mul12(e, e)) for e in UNIT12]
FK84 = [(al << k, be, ga << k) for k in (0, 12) for al, be, ga in F42]      # (x in K, m in F) -> x m, rank 84
QK = compose(FK84, lx=linmap(conj24, 24))                                   # (x, m) -> conj(x) m


QUAD = {24: ([(al, be << 12, ga) for al, be, ga in F42], LK), 12: (QN, LN), 6: (QD, LD), 3: (QJ, LJ)}
CONJMUL = {12: QK, 6: QT, 3: QI}        # by len(s): t ^= conj(x) * s for x, t of 2 len(s) (qu)bits


def quad(v, n):
    """Classical quadratic map of an n-bit register: the norm to the half-size subfield for n = 24, 12, 6
    (registers of 6 / 3 bits are in B64 / B8 coordinates), the inverse in GF(8) for n = 3."""
    q, l = QUAD[n]
    return scheme_eval(q, v, v) ^ lin(l, v)


def inv_model(x, n):
    """Classical replay of the inversion circuits on an n-bit register, n = 3, 6, 12, 24 (0 -> 0)."""
    return quad(x, 3) if n == 3 else scheme_eval(CONJMUL[n // 2], x, inv_model(quad(x, n), n // 2))


# ------------------------------------------------------------------ gate emitters: products

def _fan(b, ctrls, piv):
    """CX fan-in of the qubits ctrls onto piv."""
    if ctrls:
        b.cx(km.array(ctrls), piv)


def _fan_out(b, src, dsts):
    if dsts:
        b.cx(src, km.array(dsts))


def _parities(x, y, al, be):
    """Fan-in plan (px, cx, py, cy): after CX(cx -> px) then CX(cy -> py) the pivots hold parity(al & x) and
    parity(be & y).  x and y may share qubits (quadratic forms); the plan is its own inverse in reverse order.
    py == px exactly when both parities run over the same set of qubits: the term p AND p = p is then linear
    and the callers emit a CX / conditioned Z instead of a Toffoli-like gate."""
    a, bb = [x[i] for i in bits(al)], [y[i] for i in bits(be)]
    px, a = a[0], a[1:]
    if px in bb:            # px now holds the x-parity: re-express the y-parity over the modified register
        bb = [q for q in a if q not in bb] + [q for q in bb if q != px and q not in a] + [px]
    return px, a, bb[0], bb[1:]


def _disjoint(t, *regs):
    """Build-time guard: the target register must not share a qubit with an operand."""
    if set(t) & {q for r in regs for q in r}:
        raise ValueError('target register overlaps an operand register')


def lin_xor(b, x, t, cols):
    """t ^= lin(cols, x) (free)."""
    for q, c in zip(x, cols):
        _fan_out(b, q, [t[i] for i in bits(c)])


def mul_xor(b, x, y, t, scheme=None, anc=False):
    """t ^= scheme(x, y) with len(scheme) Toffoli-like gates; x, y restored.  t must be disjoint from x and y
    (checked).  x and y may share qubits; a term whose two parities then coincide is linear and costs a CX
    instead (so mul_xor(b, x, x, t) with a multiplication scheme is the free squaring t ^= x^2).
    anc=False: no ancilla at all (the product bit is CCX-ed onto a pivot of t between two CX fan-outs);
    anc=True: one transient AND ancilla (init_and / del_and).  Default scheme: F_SCHEME / K_SCHEME by len(x)."""
    scheme = scheme or (F_SCHEME if len(x) == 12 else K_SCHEME)
    _disjoint(t, x, y)
    a = b.alloc_qubits(1)[0] if anc else None
    for al, be, ga in scheme:
        px, cx, py, cy = _parities(x, y, al, be)
        tg = [t[i] for i in bits(ga)]
        _fan(b, cx, px)
        _fan(b, cy, py)
        if px == py:                                # same parity twice: p AND p = p
            _fan_out(b, px, tg)
        elif anc:
            b.init_and(px, py, a)
            _fan_out(b, a, tg)
            b.del_and(px, py, a)
        else:
            _fan_out(b, tg[0], tg[1:])
            b.ccx(px, py, tg[0])
            _fan_out(b, tg[0], tg[1:])
        _fan(b, cy, py)
        _fan(b, cx, px)
    if anc:
        b.free(a)


def erase_product(b, t, x, y, scheme=None):
    """Given t == scheme(x, y): return t to |0> with no Toffoli-like gate.  t is X-measured (HMR); the phase
    (-1)^(m . t) is undone by one CZ per term between the two parity pivots, conditioned on parity(m & gamma)
    (a conditioned Z when x and y share qubits and the two parities coincide).  t must be disjoint from x, y."""
    scheme = scheme or (F_SCHEME if len(x) == 12 else K_SCHEME)
    _disjoint(t, x, y)
    m, c = b.alloc_bits(len(t)), b.alloc_bits(1)[0]
    b.hmr(t, m)
    for al, be, ga in scheme:
        px, cx, py, cy = _parities(x, y, al, be)
        b.bit_store(c, False)
        b.cx(km.array([m[i] for i in bits(ga)]), c)
        _fan(b, cx, px)
        _fan(b, cy, py)
        b.cz(c, px) if px == py else b.ccz(c, px, py)
        _fan(b, cy, py)
        _fan(b, cx, px)
    b.free(m)
    b.free(c)


def quad_xor(b, src, dst, anc=False):
    """dst ^= quad(src): norm to the half-size subfield (len(src) = 24, 12, 6: 42, 15, 6 Toffoli-like) or the
    inverse in GF(8) (len 3: 3 Toffoli-like)."""
    q, l = QUAD[len(src)]
    mul_xor(b, src, src, dst, q, anc)
    lin_xor(b, src, dst, l)


def quad_erase(b, src, dst):
    """Given dst == quad(src): dst -> |0> for free."""
    q, l = QUAD[len(src)]
    lin_xor(b, src, dst, l)
    erase_product(b, dst, src, src, q)


# ------------------------------------------------------------------ gate emitters: inversion

def _grab(b, pool, k):
    """k clean qubits, from the lent pool first: (qubits, rest of the pool, freshly allocated part)."""
    own = list(b.alloc_qubits(max(0, k - len(pool))))
    return list(pool[:k]) + own, list(pool[k:]), own


def inv_init(b, x, t, anc=False, pool=()):
    """t: |0> -> x^-1 (0 -> 0) for len(x) = 3, 6, 12, 24 with 3 / 21 / 66 / 192 Toffoli-like gates.  The norm
    chain lives in the still-empty t, so only len(x)/2 scratch qubits are used (0 / 3 / 6 / 12), taken from
    `pool` (clean qubits lent by the caller) before allocating."""
    x, t, h = list(x), list(t), len(x) // 2
    if h == 1:
        return quad_xor(b, x, t, anc)
    s, pool, own = _grab(b, pool, h)
    quad_xor(b, x, t[:h], anc)                      # n = N(x), parked in t
    inv_init(b, t[:h], s, anc, pool + t[h:])        # s = n^-1
    quad_erase(b, x, t[:h])
    mul_xor(b, x, s, t, CONJMUL[h], anc)            # t = conj(x) s
    quad_erase(b, t, s)                             # s = N(t)
    b.free(own)


def inv_del(b, x, t, anc=False, pool=()):
    """Given t == x^-1: t -> |0>, for len(x) = 3, 6, 12, 24 with 0 / 9 / 39 / 123 Toffoli-like gates and
    0 / 6 / 6 / 12 scratch qubits (from `pool` first)."""
    x, t, h = list(x), list(t), len(x) // 2
    if h == 1:
        return quad_erase(b, x, t)
    s, pool, own = _grab(b, pool, max(h, 6))
    if h == 3:                                      # d = N(x), j = d^-1, then t = conj(x) j
        d, j = s[:3], s[3:]
        quad_xor(b, x, d, anc)
        quad_xor(b, d, j, anc)
        erase_product(b, t, x, j, QI)
        quad_erase(b, d, j)
        quad_erase(b, x, d)
    else:
        quad_xor(b, t, s, anc)                      # s = N(t) = N(x)^-1
        erase_product(b, t, x, s, CONJMUL[h])       # t = conj(x) s
        quad_xor(b, x, t[:h], anc)                  # n = N(x), parked in t
        inv_del(b, t[:h], s, anc, pool + t[h:])     # s = n^-1
        quad_erase(b, x, t[:h])
    b.free(own)


def tower(b, x, anc):
    """All norms of x and their inverses, kept alive: norms = [N(x), N(N(x)), ..] down to GF(8) and
    invs[k] = norms[k]^-1.  36 Toffoli-like / 18 qubits for 12-bit x, 108 / 42 for 24-bit x."""
    norms, src = [], x
    while len(src) > 3:
        norms.append(list(b.alloc_qubits(len(src) // 2)))
        quad_xor(b, src, norms[-1], anc)
        src = norms[-1]
    invs = [list(b.alloc_qubits(3))]
    quad_xor(b, src, invs[0], anc)
    for n in norms[-2::-1]:                         # n^-1 = conj(n) N(n)^-1
        invs.insert(0, list(b.alloc_qubits(len(n))))
        mul_xor(b, n, invs[1], invs[0], CONJMUL[len(invs[1])], anc)
    return norms, invs


def untower(b, x, norms, invs):
    """Free erasure of the tower of x.  The relations are symmetric under x <-> x^-1, norms <-> invs, so
    untower(b, y, invs, norms) erases the same tower from y = x^-1 when x itself is gone."""
    for k in range(len(norms) - 1):
        erase_product(b, invs[k], norms[k], invs[k + 1], CONJMUL[len(invs[k + 1])])
    quad_erase(b, norms[-1], invs[-1])
    for src, n in reversed(list(zip([x] + norms, norms))):
        quad_erase(b, src, n)
    for r in norms + invs:
        b.free(r)


def inv_xor(b, x, t, anc=False, erase=False):
    """t ^= x^-1 (0 -> 0) for any t, len(x) = 12 / 24: 66 / 192 Toffoli-like, 18 / 42 scratch qubits.
    erase=True: t == x^-1 goes to |0> instead (36 / 108 Toffoli-like).  For a clean t prefer inv_init."""
    x, t = list(x), list(t)
    norms, invs = tower(b, x, anc)
    if erase:
        erase_product(b, t, x, invs[0], CONJMUL[len(invs[0])])
    else:
        mul_xor(b, x, invs[0], t, CONJMUL[len(invs[0])], anc)
    untower(b, x, norms, invs)


def inv12_compute(b, x, t, anc=False, erase=False):
    """t ^= x^-1 in F: 66 Toffoli-like (15 + 6 + 3 + 12 + 30), 18 scratch qubits; see inv_xor."""
    inv_xor(b, x, t, anc, erase)


def inv24_compute(b, x, t, anc=False, erase=False, lean=False):
    """t ^= x^-1 in K: norm 42 + F-tower 36 + nk^-1 30 + conj(x) m 84 = 192 Toffoli-like, 42 scratch qubits.
    lean=True: only nk and i = N(nk)^-1 survive next to m = nk^-1; the rest of the F-tower is rebuilt
    afterwards (216 Toffoli-like, 30 scratch qubits).  erase as in inv_xor (108, lean 132)."""
    if not lean:
        return inv_xor(b, x, t, anc, erase)
    x, t, nk = list(x), list(t), list(b.alloc_qubits(12))
    quad_xor(b, x, nk, anc)
    (n, d), (i, j) = tower(b, nk, anc)
    quad_erase(b, d, j)
    quad_erase(b, n, d)
    quad_erase(b, nk, n)
    m = n + d + j                                   # m = nk^-1 lives in the 12 qubits of n, d, j
    mul_xor(b, nk, i, m, QT, anc)
    if erase:
        erase_product(b, t, x, m, QK)
    else:
        mul_xor(b, x, m, t, QK, anc)
    erase_product(b, m, nk, i, QT)
    quad_xor(b, nk, n, anc)
    quad_xor(b, n, d, anc)
    quad_xor(b, d, j, anc)
    untower(b, nk, [n, d], [i, j])
    quad_erase(b, x, nk)
    b.free(nk)


def inv_inplace(b, x, anc=False, lean=False):
    """x <- x^-1 in the same qubits, len(x) = 12 / 24.  The new value p = conj(x) s (s = N(x)^-1) is built in a
    12-qubit buffer (K: half by half) and the old one is erased as conj(p) n (n = N(x)), then swapped.
    Default: whole tower kept, 66 / 192 Toffoli-like, 30 / 54 scratch qubits; lean=True re-derives it:
    75 / 231 Toffoli-like, 24 / 36 scratch qubits."""
    x, h, p = list(x), len(x) // 2, list(b.alloc_qubits(12))
    if lean:
        n, s = list(b.alloc_qubits(h)), list(b.alloc_qubits(h))
        quad_xor(b, x, n, anc)
        inv_init(b, n, s, anc, p)
    else:
        norms, invs = tower(b, x, anc)
        n, s = norms[0], invs[0]
    if h == 12:
        b.cx(km.array(x[12:]), km.array(x[:12]))    # x -> conj(x)
    for part, q in ([(x[:12], F42), (x[12:], F42)] if h == 12 else [(x, QT)]):
        mul_xor(b, part, s, p, q, anc)
        erase_product(b, part, p, n, q)
        b.swap(km.array(part), km.array(p))
    if lean:                                        # now s = N(x) and n = s^-1 for the new x
        inv_del(b, s, n, anc, p)
        quad_erase(b, x, s)
        b.free(n + s)
    else:
        untower(b, x, invs, norms)
    b.free(p)

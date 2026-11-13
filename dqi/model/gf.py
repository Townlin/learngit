"""GF(2^12) arithmetic matching kickmix.GF2Field(12), plus dense polynomial helpers over it.

Elements are ints in [0, 4096); polynomials are lists of coefficients, low degree first.
"""
from __future__ import annotations

MOD = 4179          # x^12 + x^6 + x^4 + x + 1
B = 12
ORDER = (1 << B) - 1


def _mul_slow(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> B:
            a ^= MOD
    return r


G = 2               # primitive element (checked against kickmix in tests)
EXP = [1] * (2 * ORDER)
LOG = [0] * (1 << B)
for _i in range(1, 2 * ORDER):
    EXP[_i] = _mul_slow(EXP[_i - 1], G)
for _i in range(ORDER):
    LOG[EXP[_i]] = _i
assert len(set(EXP[:ORDER])) == ORDER, 'G is not primitive'


def mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]


def inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError
    return EXP[ORDER - LOG[a]]


def div(a: int, b: int) -> int:
    return mul(a, inv(b))


def power(a: int, e: int) -> int:
    if a == 0:
        return 0 if e else 1
    return EXP[(LOG[a] * e) % ORDER]


def sqrt(a: int) -> int:
    return power(a, 1 << (B - 1))


# ---- polynomials (lists, low degree first) ----

def ptrim(p):
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return p


def pdeg(p):
    p = ptrim(p)
    return len(p) - 1


def padd(p, q):
    n = max(len(p), len(q))
    return ptrim([(p[i] if i < len(p) else 0) ^ (q[i] if i < len(q) else 0) for i in range(n)])


def pmul(p, q):
    if not p or not q:
        return []
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                if b:
                    out[i + j] ^= mul(a, b)
    return ptrim(out)


def pscale(p, c):
    return ptrim([mul(a, c) for a in p])


def pdivmod(a, b):
    a = ptrim(a)
    b = ptrim(b)
    if not b:
        raise ZeroDivisionError
    q = [0] * max(0, len(a) - len(b) + 1)
    r = list(a)
    ib = inv(b[-1])
    while len(r) >= len(b):
        c = mul(r[-1], ib)
        s = len(r) - len(b)
        q[s] = c
        for i, bc in enumerate(b):
            r[s + i] ^= mul(c, bc)
        r = ptrim(r)
    return ptrim(q), r


def pmod(a, b):
    return pdivmod(a, b)[1]


def peval(p, x):
    r = 0
    for c in reversed(p):
        r = mul(r, x) ^ c
    return r


def pderiv(p):
    return ptrim([p[i] if i & 1 else 0 for i in range(1, len(p))])


def pxgcd(a, b):
    """Returns (g, u, v) with u*a + v*b = g."""
    r0, r1 = ptrim(a), ptrim(b)
    u0, u1 = [1], []
    v0, v1 = [], [1]
    while r1:
        q, r = pdivmod(r0, r1)
        r0, r1 = r1, r
        u0, u1 = u1, padd(u0, pmul(q, u1))
        v0, v1 = v1, padd(v0, pmul(q, v1))
    return r0, u0, v0


def pinv_mod(a, m):
    g, u, _ = pxgcd(pmod(a, m), m)
    if pdeg(g) != 0:
        raise ZeroDivisionError('not invertible')
    return pmod(pscale(u, inv(g[0])), m)


def from_roots(roots):
    p = [1]
    for r in roots:
        p = pmul(p, [r, 1])
    return p

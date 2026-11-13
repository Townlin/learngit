"""Classical reference decoder in the rational-function form of DESIGN.md.

decode(S, M) -> (Q, R, Phi)   with   Q N = R (mod M),  Phi = R / Q' mod Q
phase(S, masks, M)            -> the required phase bit
"""
from __future__ import annotations

from . import gf

T = 35
NS = 70
NPOS = gf.ORDER


def default_modulus():
    """A degree-70 monic polynomial with no root in F*: here simply x^70 (classical key equation)."""
    return [0] * NS + [1]


def node_poly(S, M):
    """N(x) = polynomial part of G(x) M(x), G = sum_{n>=1} p_n x^-n, p_n = S[n-1]."""
    N = [0] * NS
    for k in range(NS):
        acc = 0
        for n in range(1, NS - k + 1):
            m = M[k + n] if k + n < len(M) else 0
            if m and S[n - 1]:
                acc ^= gf.mul(S[n - 1], m)
        N[k] = acc
    return gf.ptrim(N)


def solve_key(N, M, t=T):
    """Extended Euclid on (M, N): returns (Q, R) with Q N = R mod M, deg R < t, Q monic."""
    r0, r1 = gf.ptrim(M), gf.ptrim(N)
    q0, q1 = [], [1]
    while gf.pdeg(r1) >= t:
        quo, rem = gf.pdivmod(r0, r1)
        r0, r1 = r1, rem
        q0, q1 = q1, gf.padd(q0, gf.pmul(quo, q1))
    if not q1:
        return [1], []
    lc = gf.inv(q1[-1])
    return gf.pscale(q1, lc), gf.pscale(r1, lc)


def decode(S, M=None):
    M = default_modulus() if M is None else M
    N = node_poly(S, M)
    Q, R = solve_key(N, M)
    if gf.pdeg(Q) == 0:
        return Q, R, []
    Phi = gf.pmod(gf.pmul(R, gf.pinv_mod(gf.pderiv(Q), Q)), Q)
    return Q, R, Phi


def error_word(S, M=None):
    """Returns {position: value} recovered through (Q, Phi)."""
    M = default_modulus() if M is None else M
    Q, R, Phi = decode(S, M)
    out = {}
    if gf.pdeg(Q) == 0:
        return out
    for j in range(NPOS):
        x = gf.EXP[j]
        if gf.peval(Q, x) == 0:
            kappa = gf.inv(gf.mul(x, gf.peval(M, x)))
            out[j] = gf.mul(kappa, gf.peval(Phi, x))
    return out


def phase(S, masks, M=None):
    e = error_word(S, M)
    return sum(bin(v & masks[j]).count('1') for j, v in e.items()) & 1

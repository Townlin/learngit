"""In-place GF(2)-linear maps from free gates (CX / SWAP), and the Stage-0 matrices of DESIGN.md.

Representation used throughout: a matrix is a list of Python ints, one per row, bit j of row i
being the entry (i, j); a vector is an int with bit i = coordinate i.  The column count is implicit,
so the few functions that need it take `ncols` (default: square).  to_numpy / from_numpy convert.

Fields: F = GF(2^12) is model.gf.  K = GF(2^24) = F[u]/(u^2 + u + K_C) is defined here; the int
lo + (hi << 12) stands for lo + hi*u.  K_C is the smallest element of F with absolute trace 1.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

import kickmix as km
from model import gf

CX, SWAP = 0, 1          # gate kinds of a "program": list of (kind, a, b); CX has control a, target b
FB, KB = 12, 24          # bits of an F / K element
FMASK = (1 << FB) - 1
NODES = 35               # number of quadratic factors of the Stage-0 modulus M


# ---------------------------------------------------------------- GF(2) matrices

def gf2_identity(n):
    return [1 << i for i in range(n)]


def to_numpy(A, ncols=None):
    """uint8 array [len(A), ncols] of the matrix A."""
    ncols = len(A) if ncols is None else ncols
    nb = (ncols + 7) // 8
    buf = np.frombuffer(b''.join(r.to_bytes(nb, 'little') for r in A), dtype=np.uint8)
    return np.unpackbits(buf.reshape(len(A), nb), axis=1, bitorder='little')[:, :ncols]


def from_numpy(M):
    """Inverse of to_numpy (any integer / bool 2-D array; entries are taken mod 2)."""
    packed = np.packbits(np.asarray(M).astype(np.uint8) & 1, axis=1, bitorder='little')
    return [int.from_bytes(row.tobytes(), 'little') for row in packed]


def gf2_transpose(A, ncols=None):
    return from_numpy(to_numpy(A, ncols).T) if A else [0] * (ncols or 0)


def gf2_apply(A, x):
    """A x for a vector x (int)."""
    return sum(((r & x).bit_count() & 1) << i for i, r in enumerate(A))


def gf2_matmul(A, B):
    """A B for A (m x k) and B (k x n); byte-wise tables of row combinations of B."""
    tabs = []
    for s in range(0, len(B), 8):
        t = [0]
        for row in B[s:s + 8]:
            t += [v ^ row for v in t]
        tabs.append(t)
    out = []
    for r in A:
        acc = 0
        for t, byte in zip(tabs, r.to_bytes(len(tabs), 'little')):
            acc ^= t[byte]
        out.append(acc)
    return out


def gf2_rank(A):
    rows, rank = [r for r in A if r], 0
    while rows:
        p = rows.pop()
        low = p & -p
        rows = [x for x in (r ^ p if r & low else r for r in rows) if x]
        rank += 1
    return rank


def gf2_inverse(A):
    """Inverse of a square matrix (Gauss-Jordan on [A | I]); ValueError if singular or not square."""
    n = len(A)
    _check_square(A)
    rows = [r | (1 << (n + i)) for i, r in enumerate(A)]
    for c in range(n):
        bit = 1 << c
        p = next((i for i in range(c, n) if rows[i] & bit), None)
        if p is None:
            raise ValueError('singular matrix')
        piv, rows[p] = rows[p], rows[c]
        rows = [r ^ piv if r & bit else r for r in rows]
        rows[c] = piv
    return [r >> n for r in rows]


# ---------------------------------------------------------------- synthesis

def _check_square(A):
    """ValueError unless every row of A fits in len(A) columns."""
    if any(r >> len(A) for r in A):
        raise ValueError('matrix is not square: a row has a bit beyond column len(A) - 1')


def _clear_lower(rows, sec):
    """Patel-Markov-Hayes style pass over `rows` (modified in place): row operations that zero
    every entry below the diagonal.  Columns are taken `sec` at a time; rows repeating the section
    pattern of an earlier row are cleared with one operation, and the few survivors are handled by
    halving the section recursively.  Returns [(kind, src, dst)] in the order applied:
    rows[dst] ^= rows[src] for CX, exchange of the two rows for SWAP."""
    n, ops = len(rows), []

    def clear(cand, s, e):
        """Columns [s, e); cand = ascending rows >= s that may be nonzero there."""
        mask = (1 << (e - s)) - 1
        first = {}                                   # pattern -> first row showing it
        for r in cand:
            p = (rows[r] >> s) & mask
            if p and first.setdefault(p, r) != r:
                rows[r] ^= rows[first[p]]
                ops.append((CX, first[p], r))
        live = list(first.values())                  # ascending; all other rows are now zero here
        if e - s > 1:
            mid = (s + e + 1) // 2
            clear(live, s, mid)
            clear([r for r in live if r >= mid], mid, e)
        elif not live:
            raise ValueError('singular matrix')
        elif live[0] != s:                           # single column: bring its only 1 to the diagonal
            rows[s], rows[live[0]] = rows[live[0]], rows[s]
            ops.append((SWAP, s, live[0]))

    for s in range(0, n, sec):
        clear(range(s, n), s, min(n, s + sec))
    return ops


def program_matrix(ops, n):
    """Matrix of the map a program computes on n wires."""
    rows = gf2_identity(n)
    for kind, a, b in ops:
        if kind == SWAP:
            rows[a], rows[b] = rows[b], rows[a]
        else:
            rows[b] ^= rows[a]
    return rows


def synth_linear(A, sec=None):
    """Program (list of (kind, a, b)) mapping x to A x for an invertible A; no ancilla.

    Lower part by row operations, upper part by the same pass on the transpose (PMH).  `sec` is
    the section size (1 = plain Gaussian elimination); None tries a few and keeps the shortest
    program.  The reversed program computes the inverse map.  The result is checked against A.
    ValueError if A is singular or has entries outside the n x n square."""
    n = len(A)
    _check_square(A)
    if sec is None:
        return min((synth_linear(A, s) for s in (7, 8, 12)), key=len)
    rows = list(A)
    low = _clear_lower(rows, sec)                    # R_a .. R_1 A = U  (upper triangular)
    up = _clear_lower(gf2_transpose(rows), sec)      # P_b .. P_1 U^T = I
    ops = [(k, t, s) for k, s, t in up] + low[::-1]  # A = R_1 .. R_a P_b^T .. P_1^T
    if program_matrix(ops, n) != list(A):            # not an assert: must survive python -O
        raise AssertionError('synthesis self-check failed')
    return ops


def emit_program(b, qubits, ops, inverse=False):
    """Appends a program to builder b on `qubits` (wire i = qubits[i], any order, no repeats); runs of
    equal gate kind go out as one batched call, which kickmix expands in order.  inverse=True emits
    the inverse map.  A classical condition pushed on b (push_condition) gates the whole program."""
    qs = list(qubits)
    if len(set(qs)) != len(qs):
        raise ValueError('repeated qubit')
    for kind, grp in itertools.groupby(reversed(ops) if inverse else ops, key=lambda o: o[0]):
        grp = list(grp)
        first, second = (km.array([qs[o[i]] for o in grp]) for i in (1, 2))
        (b.swap if kind == SWAP else b.cx)(first, second)


def emit_linear(b, qubits, A, sec=None):
    """In place: register value x (bit i on qubits[i]) becomes A x.  Returns the program used."""
    if len(qubits) != len(A):
        raise ValueError('matrix size does not match the register')
    ops = synth_linear(A, sec)
    emit_program(b, qubits, ops)
    return ops


def emit_xor_linear(b, src_qubits, dst_qubits, A):
    """dst ^= A src for any len(dst) x len(src) matrix A; one CX per nonzero entry."""
    src, dst = list(src_qubits), list(dst_qubits)
    if len(A) != len(dst) or any(r >> len(src) for r in A):
        raise ValueError('matrix shape does not match the registers')
    if set(src) & set(dst):
        raise ValueError('source and destination registers overlap')
    pairs = [(src[j], dst[i]) for i, r in enumerate(A) for j in range(len(src)) if r >> j & 1]
    if pairs:
        b.cx(km.array([p[0] for p in pairs]), km.array([p[1] for p in pairs]))
    return len(pairs)


# ---------------------------------------------------------------- the fields F and K

def f_trace(a):
    """Absolute trace F -> GF(2)."""
    t = 0
    for _ in range(FB):
        t ^= a
        a = gf.mul(a, a)
    return t


TRACE1 = [c for c in range(1 << FB) if f_trace(c) == 1]   # x^2 + x + c is irreducible over F
K_C = TRACE1[0]                                           # u^2 = u + K_C in K


def k_mul(a, b):
    a0, a1, b0, b1 = a & FMASK, a >> FB, b & FMASK, b >> FB
    hh = gf.mul(a1, b1)
    return (gf.mul(a0, b0) ^ gf.mul(K_C, hh)) | ((gf.mul(a0, b1) ^ gf.mul(a1, b0) ^ hh) << FB)


def k_conj(a):
    """a^(2^12): the automorphism of K over F (u -> u + 1)."""
    return a ^ (a >> FB)


def k_inv(a):
    """1/a in K; ZeroDivisionError for a = 0 (model.thiele.kinv uses the convention kinv(0) = 0)."""
    a0, a1 = a & FMASK, a >> FB
    norm = gf.mul(a0, a0 ^ a1) ^ gf.mul(K_C, gf.mul(a1, a1))      # a * conj(a), an element of F
    ninv, c = gf.inv(norm), k_conj(a)
    return gf.mul(c & FMASK, ninv) | (gf.mul(c >> FB, ninv) << FB)


def k_pow(a, e):
    r = 1
    while e:
        if e & 1:
            r = k_mul(r, a)
        a, e = k_mul(a, a), e >> 1
    return r


def k_eval(p, x):
    """Value at x in K of a polynomial with coefficients in F (or K), low degree first."""
    r = 0
    for c in reversed(p):
        r = k_mul(r, x) ^ c
    return r


# ---------------------------------------------------------------- matrix builders

def map_matrix(f, nin, nout):
    """Matrix of a GF(2)-linear function f: int -> int with nin input and nout output bits."""
    return gf2_transpose([f(1 << j) for j in range(nin)], nout)


@lru_cache(maxsize=None)
def _f_mul(c):
    return tuple(map_matrix(lambda x: gf.mul(c, x), FB, FB))


def f_mul_matrix(c):
    """12 x 12 matrix of x -> c x in F."""
    return list(_f_mul(c))


def k_mul_matrix(c):
    """24 x 24 matrix of x -> c x in K."""
    return map_matrix(lambda x: k_mul(c, x), KB, KB)


def f_frobenius_matrix(e=1):
    """12 x 12 matrix of x -> x^(2^e) in F."""
    return map_matrix(lambda x: gf.power(x, 1 << e % FB), FB, FB)


def k_frobenius_matrix(e=1):
    """24 x 24 matrix of x -> x^(2^e) in K (e = 12 is k_conj)."""
    return map_matrix(lambda x: k_pow(x, 1 << e % KB), KB, KB)


def block_diag(blocks, ncols=None):
    """Block-diagonal matrix; ncols[i] is the column count of blocks[i] (default: square blocks)."""
    out, off = [], 0
    for i, blk in enumerate(blocks):
        out += [r << off for r in blk]
        off += len(blk) if ncols is None else ncols[i]
    return out


def f_block_matrix(grid):
    """GF(2) matrix of the F-linear map whose matrix over F is `grid` (12 x 12 block per entry)."""
    out = []
    for row in grid:
        acc = [0] * FB
        for j, c in enumerate(row):
            if c:
                acc = [a | (r << FB * j) for a, r in zip(acc, _f_mul(c))]
        out += acc
    return out


def k_eval_matrix(xi, n):
    """24 x 12n matrix: n coefficient registers (c_0..c_(n-1)) in F -> sum_i c_i xi^i in K."""
    pw = [1][:n]                                     # n = 0: 24 x 0, i.e. 24 zero rows
    for _ in range(n - 1):
        pw.append(k_mul(pw[-1], xi))
    return f_block_matrix([[p & FMASK for p in pw], [p >> FB for p in pw]])


# ---------------------------------------------------------------- Stage 0: syndromes -> node values

def node_quadratics():
    """c_1..c_35: the 35 smallest c in F with trace 1; m_k(x) = x^2 + x + c_k (c_1 = K_C)."""
    return TRACE1[:NODES]


def node_points():
    """xi_1..xi_35 in K.  The roots of m_k are a + u and a + 1 + u with a^2 + a = c_k + K_C; xi_k is
    the one with the smaller integer value lo + (hi << 12), i.e. the smaller a (xi_1 = u = 4096)."""
    sol = {}
    for a in range((1 << FB) - 1, -1, -1):
        sol[gf.mul(a, a) ^ a] = a                    # descending a: the smaller root is kept
    return [sol[c ^ K_C] | (1 << FB) for c in node_quadratics()]


def node_modulus():
    """M(x) = prod_k m_k(x): 71 coefficients in F, low degree first (monic, degree 70)."""
    M = [1]
    for c in node_quadratics():
        M = gf.pmul(M, [c, 1, 1])
    return M


@lru_cache(maxsize=None)
def _node_data():
    M = node_modulus()
    n = len(M) - 1
    # N_k = sum_{j=0}^{n-1-k} S_j M_(k+j+1)   (DESIGN.md section 2 with p_(j+1) = S_j)
    to_poly = f_block_matrix([[M[k + j + 1] if k + j < n else 0 for j in range(n)] for k in range(n)])
    T = gf2_matmul([r for xi in node_points() for r in k_eval_matrix(xi, n)], to_poly)
    if gf2_rank(T) != FB * n:
        raise AssertionError('node transform is singular')
    return tuple(T), tuple(synth_linear(T))


def node_transform():
    """840 x 840 invertible matrix: syndrome bits (S_i at bits 12i..12i+11) -> node values
    (N(xi_k) at bits 24(k-1)..24(k-1)+23, as lo + (hi << 12)), N as in DESIGN.md section 2."""
    return list(_node_data()[0])


def node_program():
    """The cached free-gate program of node_transform()."""
    return list(_node_data()[1])


def _emit_node(b, qubits, inverse):
    if len(qubits) != 2 * NODES * FB:
        raise ValueError('the node transform acts on exactly 840 qubits')
    emit_program(b, qubits, _node_data()[1], inverse=inverse)


def emit_node_transform(b, qubits):
    """In place on the 840 syndrome qubits: S -> (N(xi_1), .., N(xi_35))."""
    _emit_node(b, qubits, False)


def emit_node_transform_inverse(b, qubits):
    """In place: node values -> syndromes (the forward program reversed)."""
    _emit_node(b, qubits, True)

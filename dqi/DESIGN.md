# DQI decoder — mathematical specification for our own pipeline

All arithmetic is over F = GF(2^12) as defined by `kickmix.GF2Field(12)` (little-endian polynomial
basis, modulus x^12 + x^6 + x^4 + x + 1 = 4179, primitive element `g = field.primitive_element`).
Characteristic 2: addition = XOR, minus = plus.

## 1. Task (from the verifier)

* positions j = 0..4094 with locators X_j = g^j (all nonzero field elements);
* error word e (weight w <= 35), syndromes S_i = sum_j e_j X_j^(i+1), i = 0..69;
  write p_n = S_(n-1) = sum_l Y_l X_l^n for n = 1..70 (Y_l = error value at locator X_l);
* classical masks c_j (12 bits per position);
* required phase bit: sum_j popcount(e_j & c_j) mod 2.

## 2. Rational-function form of the key equation

G(x) := sum_l Y_l X_l / (x - X_l) = sum_{n>=1} p_n x^(-n)   (Laurent series at infinity).

Pick any classical monic M(x) in F[x] of degree 70 with no root among the locators.
N(x) := polynomial part of G(x) M(x):   N_k = sum_{n=1}^{70-k} p_n M_(k+n),  k = 0..69   (linear in S).

Then with  Yt_l := Y_l X_l M(X_l)  and  Q(x) := prod_l (x - X_l)  (monic, degree w):

    H(x) := sum_l Yt_l / (x - X_l) = R(x)/Q(x),   deg R < w,
    Q(x) N(x) = R(x)  (mod M(x)),   deg Q <= 35,  deg R <= 34.          (KEY)

(Q, R) is the unique minimal solution of (KEY); classically: extended Euclid on (M, N), stop at the
first remainder of degree < 35.  At every root xi of M:  N(xi) = H(xi) = R(xi)/Q(xi), so with
M = product of 35 irreducible quadratics the data are 35 values in GF(2^24) ("node values").

Error values:  Yt_l = R(X_l)/Q'(X_l)  (Q' formal derivative; in char 2  Q' = sum over odd i of q_i x^(i-1)).
Scan-ready form:  Phi := R * (Q')^(-1) mod Q  (degree < w), so that  Phi(X_l) = Yt_l  and

    e_j = kappa_j * Phi(X_j)  if Q(X_j) = 0, else 0,     kappa_j := 1/(X_j M(X_j))  (classical).

The phase is  sum over roots j of  <c_j, kappa_j Phi(X_j)>,  i.e. per position a 12-bit zero test of
Q(X_j) and one parity that is GF(2)-linear in the coefficients of Phi.

## 3. Cost model reminders

* CX / CZ / SWAP / X / Z / measurement / reset are free; every GF(2)-linear map is free.
* CCX / CCZ / CSWAP cost 1 each (counted when executed).
* A value that is a product of two live registers can be erased for free (X-measure + CZ fix-up).
* A register that will be restored later can be X-measured now and its phase fixed by Z gates
  once the same value is back ("deferred fix").
* Score = R(eps+) * Q^2 * T; 0 failures -> R = 1.19126; each failure in 102,400 shots ~ +0.46 %.

## 4. Reference numbers

* GF(2^12) multiply: 42 AND (degree-3 polynomials over GF(8), 7 evaluation points x 6 AND).
* GF(2^24) multiply: 99 AND (degree-5 polynomials over GF(16), 11 points x 9 AND).
* Public frontier circuit: Q = 892, T = 395,394, score 3.7477e11 (see REPORT.md).

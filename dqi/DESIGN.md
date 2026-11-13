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

## 5. Thiele continued fraction: conventions

Model `model/thiele.py`, tests `tests/test_thiele.py`; experiments `python3 -m model.thiele 250` (scalar,
45 s) and `python3 -m model.thiele census 35 4000000 [seed]` (numpy, 3.5 min).  1-based indices here.

* **K** = GF(2^24) = F[u]/(u^2 + u + C0), C0 = 2048 (the trace-1 elements of F are exactly 2048..4095).
  A word is `lo | hi << 12` = lo + hi u; Frobenius: u -> u + 1; `kinv(0) := 0`.
* **Nodes** k = 1..35: m_k = x^2 + x + gam_k, gam_k = 2047 + k; xi_k = u + tau_k with tau_k in F even,
  tau_k^2 + tau_k = C0 + gam_k (xi_1 = u); M = prod m_k.  Every chain constant lies in F:
  m_k(xi_g) = gam_k + gam_g,   xi_k + xi_g = tau_k + tau_g.
* **Data**: h_g = N(xi_g) = H(xi_g) = R(xi_g)/Q(xi_g)   (identity checked numerically, weights 0..35).
* **Variant**: phi_0 := 1/H = Q/R;  c_k := phi_(k-1)(xi_k) (stored word);  nu_k = a_k + b_k x with
  b_k = hi(c_k), a_k = lo(c_k) + hi(c_k) tau_k;  phi_k = m_k/(phi_(k-1) - nu_k);
  nu_k(xi_g) = c_k + b_k (tau_k + tau_g)  (F-linear in the word).
  Reason: for phi = A/B put d = max(deg A, deg B + 1); a regular level lowers d by exactly 1 and
  d(Q/R) = w, so phi_w = infinity identically: weight 35 uses exactly the 35 nodes, tail (1 : 0).
  d(R/Q) = w + 1 would need a 36th node (tested).  b_k = 0 or deg R < w - 1 (2^-12) need no special handling.
* **Lazy extract**, nodes g = 1..35: start (R : D) = (1 : h_g); steps k = 1..g-1:
  (R, D) <- ((gam_k + gam_g) D, R + nu_k(xi_g) D);  then c_g = R * kinv(D).
  Cost: 595 chain multiplications + 34 final ones (R = 1 at node 1) = **629 K-mult, 35 K-inv**.
  **Inverse**, nodes g = 35..1: start (c_g : 1), or (1 : 0) at a pole; steps k = g-1..1:
  (R, D) <- (nu_k(xi_g) R + (gam_k + gam_g) D, R);  h_g = D * kinv(R).  Same cost: 629 mult, 35 inv.
* **Zero convention** (total on K^35, exactly reversible with the flags).  Steps are invertible
  projective maps, so (R, D) != (0, 0) and zeros inside a chain are harmless.  At the end of a chain:
  D = 0 ("pole"): store c_g = 0, flag pole_g = 1;   R = 0: c_g = 0 is an ordinary word, pole_g = 0.
  Later chains use the uniform step with nu_g = 0, i.e. (R, D) <- ((gam_k + gam_g) D, R); no step is
  ever conditioned on a flag, and kinv(0) = 0 produces the stored 0 automatically.
* **Low weight w < 35** (phi_w = infinity).  State of chain g after s steps:  s < w: R, D both nonzero;
  s >= w: D = 0, R != 0 if s - w is even;  R = 0, D != 0 if s - w is odd.  Hence at the end of chain g > w:
  g - w odd -> D = 0 (pole);  g - w even -> R = 0.  Words: c_g != 0 for g <= w, c_g = 0 for g > w;
  pole_g = [g > w and g - w odd].  **No stored flag is needed**: c_w != 0 always (phi_(w-1) = A/B, B a
  nonzero constant, 0 != A of degree <= 1), so the trailing run of zero words is exactly w+1..35 and
  **tail := pole_35 = (35 - w) mod 2 = parity of its length** (`tail_from_words`); `poles_from_tail(c)`
  rebuilds every flag (alternating backwards inside the run; c_1 = 1/h_1 = 0 is always a pole).
* **Synthesis**: (N_34, D_34) = (nu_35 + tail, 1 + tail), then (N_(k-1), D_(k-1)) = (nu_k N_k + m_k D_k, N_k)
  for k = 34..1 (tail enters by two free XORs).  Result:

      (N_0, D_0) = lam * f * (Q, R),   lam = lc(N_0) in F*,   f = prod of m_k over poles k <= 34
                 = m_(w+1) m_(w+3) ... (indices <= 34),   deg f = 2 floor((35 - w)/2),   deg N_0 = 35 - tail.

  f is a classical function of w only (w is in superposition, so f is not one fixed polynomial); it has
  no root in F, N_0 is squarefree, and Forney is invariant: e_j = D_0(X_j) / (X_j M(X_j) N_0'(X_j)) at
  the roots of N_0.  So scan/convert may use (N_0, D_0) as is; `normalise` returns the exact monic (Q, R).
  The x^35 coefficient of N_0 is the continuant of (b_1..b_35) (nonzero iff tail = 0).
  Caveat: N_0(xi_g) = D_0(xi_g) = 0 at every pole g <= 34, so h_g != D_0(xi_g)/N_0(xi_g) (0/0) there.
* **Experiments** (seeded).  Scalar: 250 random errors per weight 16..35 (5,000 decodes) and 30 per
  weight 0..15: 100 % conform to every statement above (all 630 chain states, words, flags rebuilt
  from the words, round trip, N_0 = lam f Q and D_0 = lam f R against `reference.decode`, Forney).
  Accident census (`zero_census`, numpy, cross-checked with the scalar model), weight 35, 16.25e6 decodes:
  613 chain states with D = 0 (expected 607 = 2^-24 x 627 possible states), of which 34 accidental poles
  (expected 32); 36 accidental zero words (32).  Weights 34 / 31, 2.0e6 decodes each: 73 / 90 (75 / 75).
* **Accidents** (one natural and several planted real errors, `test_accidental_poles_on_real_errors`):
  - a zero word at a node g <= w (R = 0 at the end of the chain) is an ordinary word;
  - a pole at a node g <= w - 2 raises d by 1, so the fraction ends at level w + 2.  For w <= 33 the
    forward direction absorbs it with no extra logic (f gains m_g, c_(w+2) != 0, tail is still the run
    parity, deg N_0 = 35 - tail); for w >= 34 the 35 words do not synthesize: decode failure,
    2.3e-7 per shot under the DQI weight law (0.024 expected failures per 102,400 shots);
  - the words cannot tell a pole at 2 <= g <= w - 2 from a zero word: an inverse CF with flags rebuilt from
    them mis-restores exactly that h_g: E[w - 3] 2^-24 = 1.7e-6 per shot (0.17 per 102,400 shots).

# 🚀 TokenVector.Numerics v1.1.1 Release Notes

[🇻🇳 Xem bản Tiếng Việt](RELEASE_NOTES_VI.md)

---

**Release Version:** `v1.1.1`  
**Release Date:** September 25, 2026  
**Target Framework:** .NET 8.0 LTS + TokenVector stdlib (`.tkv`, spec TV-1001)  
**License:** [MIT License](LICENSE)  
**Repository:** [https://github.com/nguyenhungtran18/TokenVector.Numerics](https://github.com/nguyenhungtran18/TokenVector.Numerics)  
**NuGet Package:** `TokenVector.Numerics` (v1.1.1)

---

## 🌟 Overview & What's New in v1.1.1

A **documentation and correctness** release. It adds a new exact-arithmetic `mathlib/` to the repository, records the verification work done on the math modules, and republishes the NuGet package with the refreshed README.

> **Binary note — read this before upgrading.** `TokenVector.Numerics.dll` in this release is **byte-identical to v1.1.0**. The C# engine sources were removed from the repository in the v1.1.0 release (`feat!: remove runtime engine sources`), so the assembly can no longer be recompiled from source. **v1.1.1 is a documentation-only bump** — the `mathlib/` additions are `.tkv` source, not part of the assembly. If you need a rebuilt binary, the engine sources must be restored first.

---

## 🔬 New: Exact-Arithmetic `mathlib/` (`.tkv`)

Two pure-TokenVector modules with no `tv` import, so they compile and run standalone with `tkvc`:

* **`mathlib/bf_bigfloat.tkv`** — base-$10^4$ bignum arithmetic, digit-by-digit $\sqrt{n}$ long division, $\pi$ via Chudnovsky binary splitting (Gourdon's formula), $e$ by series spigot, and full big-num division.
* **`mathlib/nt_number_theory.tkv`** — gcd/lcm, integer square root, trial division, `iroot`, overflow-safe `pow_mod` / Miller–Rabin primality / Pollard–Rho factorisation, and AKS primality.

## 🐛 Correctness Fixes (5 defects)

| Module | Defect | Impact | Fix |
| :--- | :--- | :--- | :--- |
| `bf_bigfloat.tkv` | Chudnovsky constant `10939058825628000` | $\pi$ wrong from ~14 significant digits on | corrected to `10939058860032000` |
| `bf_bigfloat.tkv` | `bignum_neg` dropped a valid limb | wrong results for negative bignums | complement path removed |
| `bf_bigfloat.tkv` | `pi_chudnovsky` under-fetched precision, mishandled leading zero | low-order digit errors at high precision | extra guard digits, leading zero stripped, then normalised |
| `nt_number_theory.tkv` | modular multiply/add wrapped silently on `i64` | wrong primality and factorisation near $2^{63}$ | added `mul_mod_i64` / `add_mod_i64` |
| `nt_number_theory.tkv` | `isqrt_i`, `trial_prime`, `factorize`, `iroot` overflowed at boundaries | failures at the top of the `i64` range | all intermediates range-guarded |

## 🧪 Verification Recorded

| Suite | Result | Type |
| :--- | :---: | :--- |
| `mathlib/bf_bigfloat.tkv` | **8 / 8** | native run (`tkvc build` + execute) |
| `mathlib/nt_number_theory.tkv` | **9 / 9** | native run (`tkvc build` + execute) |
| `linalg` / `linalg_functions` / `fft` / `crypto_graph` | **36 / 36** | source-level review vs independent oracles |

Four regression tests were added: `t7_bignum_neg`, `t8_precision_borders`, `t8_i64_boundaries`, `t9_large_factorization`.

## 📖 Documentation Corrections

The 175-check `.tkv` smoke command quoted throughout the documentation **does not reproduce** on current `tkvc` builds — the compiler resolves `import tv` against its own bundled directory and exposes no runtime-path flag, so the build aborts with `File khong co ham top-level nao co annotation kieu DSL`. The README, test report, `llms.txt` and these notes now state plainly that **175/175 is the v1.1.0 release record** and name the `mathlib` suites as the reproducible check. The benchmark table (compiled `tkvc` kernels vs NumPy 2.5.2) is unchanged and remains attributed to the v1.1.0 measurement run.

## 📦 Packaging

`packages/TokenVector.Numerics.1.1.1.nupkg` — same `lib/net8.0/TokenVector.Numerics.dll` and XML docs as v1.1.0, with this release's README embedded.

---

# 🚀 TokenVector.Numerics v1.1.0 Release Notes

[🇻🇳 Xem bản Tiếng Việt](RELEASE_NOTES_VI.md)

---

**Release Version:** `v1.1.0`  
**Release Date:** September 24, 2026  
**Target Framework:** .NET 8.0 LTS + TokenVector stdlib (`.tkv`, spec TV-1001)  
**License:** [MIT License](LICENSE)  
**Repository:** [https://github.com/nguyenhungtran18/TokenVector.Numerics](https://github.com/nguyenhungtran18/TokenVector.Numerics)  
**NuGet Package:** `TokenVector.Numerics` (v1.1.0)

---

## 🌟 Overview & What's New in v1.1.0

The **v1.1.0 release** ships the complete **TokenVector language translation** of the entire library (27 `.tkv` modules, ~12.1k lines, spec `TKV-SPEC-SYNTAX-2026-V1`), reaching **100% coverage of the audited numeric-library function surface** (354/354 names, 0 missing) with verified numeric parity against an independent reference, plus a cache-tiled matmul hot path, reference-parity fancy indexing, and a reproducible verification toolchain (syntax gate + runtime harness + coverage audit + benchmark).

---

## 🚀 Key Additions in v1.1.0

### 1. Full TokenVector Language Translation (`src/tokenvector/`, TV-1001)
* **27 `.tkv` modules** (~12.1k lines) — the complete TokenVector standard library, every mapping documented in each file's "Source of truth" header.
* **TV-1001 conventions:** snake_case module functions, class names preserved (`NDArray`, `Tensor`, `QState`, `KDTree`, …), dict/`isinstance`/nested-def constructs replaced by parallel lists and structural checks, Parallel.For/AVX2 collapsed to scalar loops (`tkvc -O parallel -O simd` restores them at the CIL level).
* **Acyclic import graph:** broadcast-shape helpers hosted in `tv.core`, re-exported by `tv.engine`.

### 2. 100% Numeric Library-Surface Coverage
* **354/354 audited numeric-library functions now have a `.tkv` counterpart** — audited against an independent reference library's entire public surface (595 names), verifying methods against the real `NDArray`/`BoolNDArray`/`Tensor` classes.
* **~80 new functions** added across: elementwise/scalar math (`abs`, `pow`, `maximum/minimum`, `gcd/lcm`, `frexp/ldexp/modf/divmod`, `nextafter/spacing`, `nan_to_num`, `fmax/fmin`, `heaviside`, `signbit`, `ptp`), array ops (`array_split`, `delete`, `argwhere`, `trim_zeros`, `broadcast_arrays`, `indices`, `unique_all` family, `sort_complex`, `packbits/unpackbits`), binning/indexing (`digitize`, `bincount`, `diagflat`, `fill_diagonal`, `tri`, `tril/triu_indices`, `mask_indices`, `vander`, `partition`, `searchsorted`, `choose`), complete FFT family (`fftn/ifftn`, `rfft2/irfft2`, `rfftn/irfftn`, `fftfreq/rfftfreq`, `fftshift/ifftshift`, `ihfft/hfft`), creation (`empty`, `logspace`, `geomspace`, `ravel`, `copy`, `astype`, `real/imag`, `ndim/size`), `histogram`/`histogram2d`/`histogramdd`, `euler_gamma`, `kaiser`, `bitwise_count`.
* The remaining 241 audited public names (dtype objects, constants, RNG machinery, errstate/printing, packaging) belong to the language/compiler layer in TokenVector (`tv.f64`, runtime constants, `tkvc`) by design.

### 3. Performance & Numeric Parity (compiled via `tkvc.exe`)
* **Cache-tiled matmul** (32×32 blocks) rewritten with raw flat-index arithmetic and unit-stride inner loops (`matmul_2d`, `batch_matmul`) — same API, identical results.
* **Compiled benchmark vs NumPy 2.5.2** (pure-TokenVector kernels compiled to standalone exes via `tkvc.exe`, same machine, best of 3, startup-subtracted): numeric parity confirmed per kernel — matmul/add checksums to float64 limits, SVD singular values **2.4e-14** vs LAPACK, FFT **~1e-12**. Ratios vs NumPy: add ~7×, FFT radix-2 ~17×, matmul 365–1308×, SVD ~1880×. The earlier interpreter-on-interpreter harness figures (400–3000×) are obsolete — the harness was removed from the repo.

### 4. Reference-Parity Fancy Indexing (`grid.tkv`)
* `boolean_select` / `boolean_assign` — `arr[mask]` and `arr[mask] = values` with right-aligned broadcastable boolean masks (scalar or exact-length value arrays, standard error semantics).
* `flat_index_select` / `flat_index_assign` — `np.take`/`np.put` with negative indices and `mode='raise'` bounds checking.

### 5. Verification Toolchain (`tests/tokenvector/`)
* **`smoke_tests.tkv`** — 175 checks across all 27 modules, compiled and executed natively: `tkvc.exe build smoke_tests.tkv --entry main --out smoke.exe`.
* Coverage audit result: **354/354** audited numeric-library names matched (audited against an independent reference; audit script removed together with the Python harness).

---

## 📊 Verification & Test Metrics (v1.1.0)

| Suite | Result |
| :--- | :--- |
| Runtime xUnit (`TokenVector.Numerics.Tests`) | **84 / 84 passed (100%)** |
| `.tkv` smoke suite (27 modules) | **175 / 175 passed (100%)** |
| Numeric library-surface coverage | **354 / 354 = 100%, 0 missing** |
| Numeric parity vs reference implementation (benchmark kernels) | max\|diff\| 0.0 → 1.1e-14 |
| `.tkv` syntax gate (TV-1001 grammar) | 27/27 modules clean |

---

# 🚀 TokenVector.Numerics v1.0.1 Release Notes (Previous Release)

**Release Version:** `v1.0.1`  
**Release Date:** September 13, 2026  
**Target Framework:** .NET 8.0 LTS  
**License:** [MIT License](LICENSE)  
**Repository:** [https://github.com/nguyenhungtran18/TokenVector.Numerics](https://github.com/nguyenhungtran18/TokenVector.Numerics)  
**NuGet Package:** `TokenVector.Numerics` (v1.0.1)

---

## 🌟 Overview & What's New in v1.0.1

The **v1.0.1 release** introduces critical high-order mathematical, optimization, and scientific signal processing capabilities to `TokenVector.Numerics`. This update expands linear algebra decomposition, advanced special transcendental functions, discrete wavelet analysis, quadratic programming solvers for drone/robotics control, and symplectic Hamiltonian integrators.

---

## 🚀 Key Additions in v1.0.1

### 1. Advanced Special Functions (`SpecialFunctions`)
* **Lambert W Function (`LambertW`):** Computes $W_k(x)$ where $W(x) e^{W(x)} = x$ using Halley's 3rd-order root-finding method. Supports both principal branch $k=0$ (for $x \ge -1/e$) and secondary branch $k=-1$ (for $-1/e \le x < 0$) with full scalar and tensor support.
* **Bessel Functions of the Second Kind (`BesselY0`, `BesselK0`):**
  * $Y_0(x)$ (Neumann function of order 0).
  * $K_0(x)$ (Modified Bessel function of the second kind of order 0).
* **Airy Functions (`AiryAi`, `AiryBi`):** Computes solutions to the differential equation $y'' - x y = 0$ using Maclaurin series around zero and asymptotic expansions for large $|x|$.

### 2. Matrix Decompositions & Matrix Equations (`LinAlg`)
* **Real Schur Decomposition (`Decomposition.Schur`):** Decomposes square matrix $A = Q T Q^T$, where $Q$ is an orthogonal matrix and $T$ is quasi-upper triangular, via Hessenberg reduction and shifted QR iteration.
* **Sylvester Equation Solver (`Decomposition.SolveSylvester`):** Solves the continuous Sylvester matrix equation $A X + X B = C$ (and Lyapunov equations when $B = A^T$) using Kronecker vectorization.

### 3. Wavelet & Analytic Signal Processing (`SignalProcessing`)
* **Discrete Wavelet Transform (`DWT` & `IDWT`):** 1D single-level forward and inverse wavelet transform supporting Haar and Daubechies-4 (`db4`) filter banks with exact signal reconstruction.
* **Hilbert Transform & Analytic Signal (`Hilbert`, `AnalyticSignal`):** Computes the analytic signal $x_a(t) = x(t) + i \mathcal{H}[x(t)]$ in $O(N \log N)$ via FFT.

### 4. Quadratic Programming Solver (`Optimize.QPSolve`)
* **Convex QP Solver:** Solves general convex quadratic programs:
  $$\min_x \frac{1}{2} x^T P x + q^T x \quad \text{subject to} \quad G x \le h, \quad A x = b, \quad lb \le x \le ub$$
* **ADMM Operator-Splitting Engine:** Matrix factorization with proximal operator projections, optimized for real-time model predictive control (NMPC), robotics trajectory optimization, and drone swarms.

### 5. Symplectic Hamiltonian Integrator (`PhysicsODEAndFields.SolveSymplecticVerlet`)
* **Energy-Preserving Symplectic Leapfrog / Velocity-Verlet:** Integrates Hamiltonian dynamics $H(q, p) = \frac{1}{2m} p^T p + V(q)$ preserving phase-space volume and total energy over long trajectories without secular energy drift.

---

## 📊 Verification & Test Metrics

* **Total Tests:** **84 / 84 Tests Passed (100% Pass)**
* **Failures:** **0**
* **Execution Time:** **~138 ms**
* **Verification Suite:** `TokenVector.Numerics.Tests` (xUnit, Release x64)

---

## 📦 Distribution Packages (`dist/`)

| Artifact | Path | Description |
| :--- | :--- | :--- |
| **Release Zip** | `dist/TokenVector.Numerics-v1.0.1-Release.zip` | Standalone v1.0.1 release bundle |
| **NuGet Package** | `dist/nuget/TokenVector.Numerics.1.0.1.nupkg` | Official v1.0.1 NuGet package |
| **Binary DLL** | `dist/bin/TokenVector.Numerics.dll` | Optimized standalone .NET 8 assembly |

---

## 🤝 Contributing & Community

Contributions, issue reports, and discussions are welcome on our [GitHub Repository](https://github.com/nguyenhungtran18/TokenVector.Numerics).

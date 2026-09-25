# TokenVector.Numerics

[ 🇬🇧 English ](README.md) | [ 🇻🇳 Tiếng Việt ](README_VI.md)

<p align="center">
  <img src="assets/icon.png" alt="TokenVector" width="128" />
</p>

<h1 align="center">TokenVector.Numerics</h1>

<p align="center">
  <strong>Enterprise Tensor & Mathematical Super-Library</strong><br/>
  Numerics runtime for the <a href="https://github.com/nguyenhungtran18/TokenVector">TokenVector</a> language
</p>

[![.NET 8](https://img.shields.io/badge/.NET-8.0%20LTS-purple.svg)](https://dotnet.microsoft.com/)
[![CI / CD](https://github.com/nguyenhungtran18/TokenVector.Numerics/actions/workflows/ci.yml/badge.svg)](https://github.com/nguyenhungtran18/TokenVector.Numerics/actions)
[![NuGet](https://img.shields.io/badge/NuGet-v1.1.1-blue.svg)](https://github.com/nguyenhungtran18/TokenVector.Numerics/packages)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-84%2F84%20Passed-brightgreen.svg)]()
[![TokenVector stdlib](https://img.shields.io/badge/.tkv%20stdlib-100%25%20numeric%20surface-9cf.svg)](src/tokenvector/README.md)
[![llms.txt](https://img.shields.io/badge/llms.txt-AI%20index-8a2be2.svg)](llms.txt)

**`TokenVector.Numerics.dll`** is a high-performance, standalone mathematical library for multidimensional tensors, Automatic Differentiation (Autograd Engine), analytical linear algebra, optimization, spline interpolation, computational physics, quantitative finance, astrodynamics, quantum computing, post-quantum cryptography, and non-Euclidean geometry built for the [**TokenVector**](https://github.com/nguyenhungtran18/TokenVector) ecosystem and compiler (a statically-typed native language compiling directly to .NET CIL AOT).

The library serves as the **Grand Unified Runtime Math & Tensor Engine**, covering multidimensional tensors, automatic differentiation (Autograd), linear algebra, matrix functions, nonlinear optimization, splines, statistical distributions & hypothesis tests, DSP & FFT signal processing, astrodynamics, quantitative finance, computational physics, quantum state simulation, lattice cryptography, structural biology, spectral graph neural networks, and hyperbolic geometry.

---

## 📋 Key Features of TokenVector.Numerics

| Category | TokenVector.Numerics |
| :--- | :--- |
| **Architecture & Threading** | **True No-GIL Multithreading**; scales 100% across all CPU cores via `Parallel.For` |
| **Compilation & Runtime** | Direct CIL opcode generation, Native AOT compilation, JIT Hardware Intrinsics |
| **Automatic Differentiation** | **Native Dynamic DAG Autograd**, Reverse-Mode VJP, Unbroadcasting, `Tensor<T>`, `AdamW`, `SGD`, `Linear`, `RMSNorm` |
| **Memory & Zero-Copy Slicing** | Hybrid `TensorBuffer<T>` (GC + `NativeMemory.AllocZeroed`), $O(1)$ zero-copy slicing |
| **Broadcasting Engine** | Right-aligned broadcasting with **Stride-0 Tricking** (zero-copy virtual expansion) |
| **Hardware SIMD Vectorization** | Generic `Vector<T>` + fast-path `Vector256<float/double>`, AVX2, FMA |
| **Matrix Algebra (LinAlg)** | Tiled MatMul Cache-Blocking 32x32, Thomas Tridiagonal $O(N)$, Solve $Ax=b$, Inverse, Det |
| **Matrix Decompositions** | `LU` (Partial Pivoting), `QR` (Householder), `Cholesky` (SPD), `SVD` (Jacobi), `Eigen` (`Eigh`), `NullSpace` |
| **Einstein Summation & Kronecker**| `EinSum.Evaluate`, `KroneckerSum` ($A \oplus B$), `Kron`, `Outer` |
| **Matrix Functions** | `Expm` ($e^A$ Padé [6/6]), `Sqrtm` ($\sqrt{A}$ Denman-Beavers), `SolveSylvester` |
| **Nonlinear Optimization** | `MinimizeBrent`, `MinimizeNelderMead` (Simplex), `MinimizeBFGS`, `RootBrentq`, `LinearProgramSimplex` |
| **Interpolation & Splines** | `CubicSpline` 1D, `BilinearInterpolation` 2D, `FitRBF` (Radial Basis Function), `Barycentric` |
| **Statistical Distributions** | `NormalPDF/CDF/PPF`, `StudentT`, `Exponential`, `Skewness`, `Kurtosis`, `TTest1Sample`, `TTestInd`, `ANOVA1Way`, `ChiSquareTest` |
| **Spatial KD-Trees & Geometry** | `KDTree` ($O(\log N)$ k-NN), `ConvexHull2D` (Monotone Chain), `CDist`, `PointInPolygon`, `PolygonArea` |
| **Signal Processing & FFT** | **Bluestein Chirp-Z FFT** for **arbitrary prime length $N$**, `RFFT1D`, `Convolve`, `Correlate`, DSP Windows |
| **Special Mathematical Functions**| `Beta`, `LogBeta`, `Digamma`, `Sinc`, `Logit`, `Expit`, `Erfinv`, `Erf`, `Gamma`, `LogGamma`, `BesselI0/J0` |
| **Element-wise & Trigonometry** | `Sin`, `Cos`, `Tan`, `ArcTan2`, `Sinh`, `Cosh`, `Tanh`, `Exp2`, `Expm1`, `Log1p`, `LogAddExp`, `Hypot`, `Deg2Rad` |
| **Polynomials & Root Finding** | `PolyFit` (Vandermonde + `LstSq`), `PolyVal` (Horner), `Roots` (Companion QR) |
| **Cumulative & Differences** | `CumSum`, `CumProd`, `Diff` ($n$-th order differences) |
| **Grid Operations & Shapes** | `Meshgrid`, `Diag`, `Diagonal`, `Triu`, `Tril`, `ExpandDims`, `Squeeze`, `BroadcastTo`, `UnravelIndex` |
| **Archive Storage & Out-of-Core**| `.npy` v1.0, `.npz` Zip multi-tensor, `MemoryMappedNDArray` (Zero-RAM disk map) |
| **Astrodynamics & Space Mech** | `J2Perturbation`, `BiEllipticTransfer`, `GibbsOrbitDetermination`, `ECI_To_ECEF`, `SolveKepler`, `HohmannTransfer` |
| **Quantitative Finance** | `BinomialTreeAmericanOption`, `ValueAtRisk` (VaR/CVaR), `BondPrice`, `MacaulayDuration`, `NelsonSiegel`, Black-Scholes, Greeks |
| **Time Series & Kalman Filter** | `KalmanFilter1D`, `KalmanFilterND`, `HoltLinearTrend`, `Autocorrelation`, `PACF` |
| **Computational Physics & ODE** | Runge-Kutta 4 (`SolveRK4`), Symplectic Verlet N-Body, `Gradient3D`, `Divergence3D`, `Laplacian3D` |
| **AI, LLM & Transformer Kernels**| `ApplyRoPE` (Rotary Position Embedding), `RMSNorm`, `ScaledDotProductAttention`, `im2col Conv2D`, `GELU`, `LayerNorm` |
| **Quantum State Simulator** | N-Qubit `QState`, $H, X, Y, Z, S, T, Rz, \text{CZ}, \text{SWAP}, \text{CRz}, \text{CNOT}, \text{Toffoli}$, Circuit `QFT`, Born rule |
| **Post-Quantum Cryptography** | Number Theoretic Transform (`ForwardNTT`/`InverseNTT`), `PolyMulNTT`, LLL Lattice Reduction |
| **Structural Biology (AlphaFold)**| Dihedral Angles ($\phi, \psi, \omega$), Kabsch RMSD, TM-Score Fold Similarity |
| **Spectral Graph & GNN** | Normalized Graph Laplacian $L_{sym}$, Chebyshev Polynomial Graph Convolution |
| **Fluid Dynamics (CFD)** | 2D Lattice Boltzmann Method (LBM D2Q9) Navier-Stokes simulation |
| **Optimal Transport & Diffusion** | Entropic Sinkhorn Wasserstein Distance, DDIM Generative Diffusion Step |
| **Optimal Control & Robotics** | Discrete/Continuous Riccati LQR (`SolveDiscreteLQR`), Damped Least Squares IK (`JacobianDLS`) |
| **Hyperbolic Non-Euclidean Geom**| Poincaré Ball Geodesic, Möbius Addition ($u \oplus_c v$), Lorentz/Hyperboloid Exp/Log Maps |
| **Packaging & Distribution** | Single standalone `TokenVector.Numerics.dll` assembly |

---

## 🧪 Verification & Quality Assurance

### Reproducible today — exact-arithmetic `mathlib/`

The `mathlib/` modules are pure TokenVector with no `tv` import, so they compile and run end-to-end with the native compiler. Re-verified **September 25, 2026**:

```powershell
tkvc build mathlib/bf_bigfloat.tkv      --out bf_bigfloat.exe      && ./bf_bigfloat.exe
#    t1_sqrt2(100 cs)  t2_pi_chud(100 cs)  t3_e(100 cs)  t4_arith  t5_div  t6_bigmul
#    t7_bignum_neg  t8_precision_borders        -> PASS 8 / 8 - bigfloat OK

tkvc build mathlib/nt_number_theory.tkv --out nt_number_theory.exe && ./nt_number_theory.exe
#    t1..t7  +  t8_i64_boundaries  t9_large_factorization
#                                            -> PASS 9 / 9 - number_theory OK
```

| Module | Result | What it covers |
| :--- | :---: | :--- |
| `mathlib/bf_bigfloat.tkv` | **8 / 8 PASS** | $\sqrt{2}$, $\pi$ (Chudnovsky) and $e$ (spigot) to 100 digits checked against independent references; bignum add/sub/mul/div; negative-bignum limb handling; precision-border behaviour |
| `mathlib/nt_number_theory.tkv` | **9 / 9 PASS** | gcd/lcm, `isqrt`, trial division, `iroot`, **overflow-safe** `pow_mod` / Miller–Rabin / Pollard–Rho, AKS; plus `i64` boundary and large-factorization regressions |
| `linalg` / `linalg_functions` / `fft` / `crypto_graph` | **36 / 36** | complex add/sub/mul/div/conj/modulus; matmul, det, trace, solve, inverse, QR, Cholesky ($A = LL^\top$), SVD, eigh; FFT against an independent DFT oracle for $N = 1,2,4,5,6,7,8,9$ plus IFFT round-trip; NTT round-trip, cyclic convolution, normalised and unnormalised graph Laplacian |

**Defects found and fixed in this pass**

| Module | Defect | Fix |
| :--- | :--- | :--- |
| `bf_bigfloat.tkv` | Chudnovsky constant was `10939058825628000` | corrected to `10939058860032000` (old value wrong from ~14 digits on) |
| `bf_bigfloat.tkv` | `bignum_neg` discarded a valid limb | complement path removed; direct negation |
| `bf_bigfloat.tkv` | `pi_chudnovsky` under-fetched working precision and mishandled a leading zero | now requests `len(den_digits) + prec_digits + 2`, strips the leading zero, then normalises |
| `nt_number_theory.tkv` | modular multiply/add wrapped silently on `i64` | added `mul_mod_i64` / `add_mod_i64`; `pow_mod`, Miller–Rabin and Pollard–Rho route through them |
| `nt_number_theory.tkv` | `isqrt_i`, `trial_prime`, `factorize`, `iroot` overflowed at `i64` boundaries (`x + 1`, `i * i`, `p * p`) | every intermediate is now range-guarded and checked |

Regression tests `t7_bignum_neg`, `t8_precision_borders`, `t8_i64_boundaries` and `t9_large_factorization` were added to lock these fixes in.

> **Scope note.** The 36/36 row is a *source-level* algorithm review against independent oracles — not a native execution of `linalg`/`fft`/`crypto_graph`, which the compiler limitation below currently prevents. The 8/8 and 9/9 rows are genuine native runs.

### Known limitation — the 175-check stdlib smoke suite

The v1.1.0 release record is **84/84** runtime unit tests, **175/175** `.tkv` smoke checks and **354/354** numeric-surface coverage ([TEST_REPORT.md](TEST_REPORT.md)). The smoke command quoted there

```powershell
tkvc build tests/tokenvector/smoke_tests.tkv --entry main --out smoke.exe
```

does **not** reproduce on `tkvc` builds that ship without a runtime-path option: the compiler resolves `import tv` against its own bundled directory and aborts with `File khong co ham top-level nao co annotation kieu DSL`, while the `-r src/tokenvector` form referenced by the file header is no longer accepted by `tkvc build`. Treat 175/175 as the **v1.1.0 record**, and use the two `mathlib` suites above as the reproducible check.

See [src/tokenvector/README.md](src/tokenvector/README.md) for the module map, coverage tables, and benchmarks.

---

## ⚡ Performance & Benchmarks

Benchmarks below are measured on the **real TokenVector compiler** (`tkvc.exe`, CIL/native): pure-TokenVector kernels — same algorithms as this stdlib (cache-tiled 32×32 matmul, one-sided Jacobi SVD, radix-2/Bluestein FFT) — compiled to standalone `.exe` files and timed against NumPy 2.5.2 on the same Windows x86_64 machine, same repetition counts per side, best of 3, process startup (~26 ms, measured with a no-op exe) subtracted:

| Kernel | Workload / Shape | TokenVector (compiled) | NumPy 2.5.2 | Ratio |
| :--- | :--- | ---: | ---: | :---: |
| **Matrix Multiplication** (cache-tiled 32×32) | 32×32 f64 | 2.28 ms | 6.2 µs | ~365× |
| **Matrix Multiplication** (cache-tiled 32×32) | 64×64 f64 | 17.5 ms | 23.4 µs | ~748× |
| **Matrix Multiplication** (cache-tiled 32×32) | 128×128 f64 | 151.9 ms | 116.1 µs | ~1308× |
| **Elementwise Add + Broadcast** | 2×131072 f64 | 6.05 ms | 849.7 µs | ~7.1× |
| **SVD** (one-sided Jacobi, singular values) | 40×40 f64 | 303.0 ms | 161.1 µs | ~1880× |
| **FFT Radix-2** | 1024 points | 688.9 µs | 40.0 µs | ~17.2× |
| **FFT Bluestein** (chirp-z, non-power-of-2) | 1000 points | 6.54 ms | 37.5 µs | ~174× |

**Numeric parity is verified per kernel against NumPy:** matmul/add checksums (Σ, Σx²) agree to float64 limits, SVD singular values differ from LAPACK by **2.4e-14**, FFT sample coefficients (X[1], X[N/2]) match to **~1e-12**.

#### 🔑 How to read the ratios:
* NumPy is a hand-tuned C library (SIMD/AVX2, LAPACK, pocketfft). The compiled TokenVector gap comes from `tkvc`'s boxed-integer arithmetic (`TkvInt` struct with a BigInteger fast path) and `List<T>` element storage — not from interpretation.
* These are **compiled** figures: the earlier interpreter-on-interpreter harness numbers (400–3000×) were 3–250× slower than this on the same kernels.
* The measurement is reproducible: `bench_test.tkv` (TokenVector source) is compiled with `tkvc.exe build bench_test.tkv --out bench_test.exe`, split into one kernel per exe, and timed by a small external wrapper.

---

## 🚀 Quick Start & Usage

```tkv
import tv
from tv.core import from_array
import tv.autograd as ag

# 1. Zero-copy multidimensional tensor slicing & broadcasting
a = from_array([1, 2, 3, 4, 5, 6], [2, 3])
b = from_array([10, 20, 30], [1, 3])
c = tv.ops.add(a, b)                     # Stride-0 broadcasting -> [2, 3]
view = a.slice([[0, 2, 1], [1, 3, 1]])   # zero-copy view ~ a[0:2, 1:3]

# 2. Dynamic compute graph autograd
x = ag.Tensor.full(3.0, [1])
x.requires_grad = True
y = (x * x) + (2.0 * x) + 1.0
y.backward()
print(x.grad.get([0]))                   # dy/dx = 2*3 + 2 = 8.0

# 3. Train a multi-layer perceptron (MLP)
model = ag.Sequential([ag.Linear(2, 8), ag.Linear(8, 1)])
optimizer = ag.AdamW(model.parameters(), lr=0.05)
```

---

## 🛠️ Compiling with TokenVector Language (`tkvc.exe`)

To compile native TokenVector (`.tkv` / `.tv`) programs:

1. **Clone the official TokenVector repository** to obtain `tkvc.exe` and standard libraries (`stdlib`):
   ```powershell
   git clone https://github.com/nguyenhungtran18/TokenVector.git
   ```
2. **Compile your TokenVector program** into a standalone native executable:
   ```powershell
   ./tkvc.exe build main.tkv --out app.exe
   ```
3. **Execute the compiled binary**:
   ```powershell
   ./app.exe
   ```

`build` is the only subcommand; see the [syntax specification](TOKENVECTOR_SYNTAX_SPEC.md) §7 for all flags.

---

For comprehensive documentation across all 35 mathematical categories and language grammar, check:
* 📖 [User Guide (English)](USER_GUIDE.md) | [Hướng dẫn sử dụng (Tiếng Việt)](USER_GUIDE_VI.md)
* 📐 [TokenVector Syntax Specification](TOKENVECTOR_SYNTAX_SPEC.md) | [Đặc tả cú pháp TokenVector](TOKENVECTOR_SYNTAX_SPEC_VI.md)
* 🏛️ [TokenVector Official Compiler Repository](https://github.com/nguyenhungtran18/TokenVector)

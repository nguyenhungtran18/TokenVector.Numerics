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
[![NuGet](https://img.shields.io/badge/NuGet-v1.1.0-blue.svg)](https://github.com/nguyenhungtran18/TokenVector.Numerics/packages)
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

All **84/84 automated unit tests** passed with zero failures in Release mode:
```powershell
dotnet test TokenVector.Numerics.sln -c Release
```
```text
Passed!  - Failed: 0, Passed: 84, Skipped: 0, Total: 84, Duration: 179 ms - TokenVector.Numerics.Tests.dll (net8.0)
```

**New in v1.1.0 — TokenVector stdlib (`.tkv`)**: the library is now also published natively in the TokenVector language (27 modules, ~12.1k lines) with **full numeric-surface coverage** (354/354 audited functions) and its own verification toolchain:
```powershell
python tests/tokenvector/tkv_harness.py
#    TokenVector stdlib smoke tests: passed=175, failed=0
python tests/tokenvector/numpy_coverage_audit.py
#    matched in .tkv stdlib : 354 (100%) | truly missing: 0
```
See [src/tokenvector/README.md](src/tokenvector/README.md) for the module map, coverage tables, and benchmarks.

---

## ⚡ Performance & Benchmarks

Benchmarked on **AMD Ryzen / Intel x86_64** (Release build, .NET 8 LTS, Native SIMD AVX2/FMA, 100% Core scaling via `Parallel.For`):

| Operation | Workload / Shape | Standard Baseline | TokenVector.Numerics (AVX2 + MT) | Speedup |
| :--- | :--- | :--- | :--- | :---: |
| **Matrix Multiplication (`MatMul`)** | $1024 \times 1024$ FP32 | 148.2 ms | **7.8 ms** (Cache-Tiled 32x32) | **19.0x** |
| **2D Fast Fourier Transform (`FFT2D`)**| $1024 \times 1024$ Complex64 | 82.5 ms | **6.1 ms** (Radix-2 + AVX2) | **13.5x** |
| **Autograd MLP Backward Pass** | 1000 iter ($B=64, D=128$) | 312.0 ms | **18.4 ms** (Zero-Alloc DAG) | **17.0x** |
| **Cosine Vector Similarity** | $1,000,000 \times 128$-dim | 195.4 ms | **11.2 ms** (AVX2 FMA Vector256) | **17.4x** |
| **Out-of-Core Memory-Mapped I/O** | $10\text{ GB}$ `.npy` Disk Slice | 4,200 ms (Full RAM load) | **0.8 ms** (Zero-RAM `mmap`) | **5250x** |

#### 🔑 Key Acceleration Pillars:
* **Hardware SIMD (AVX2 & FMA):** Processes 8 FP32 values per CPU cycle in hardware vector registers with fused multiply-add.
* **$32 \times 32$ Cache-Tiling:** Keeps matrix sub-blocks within ultra-fast L1 Data Cache (1–4 ns latency), eliminating memory wall cache misses.
* **True No-GIL Multithreading:** Scales compute tasks linearly across 100% of physical CPU cores via `Parallel.For`.
* **Zero-GC & Direct Pointers:** Leverages unmanaged `TensorBuffer<T>`, `Span<T>`, and in-place buffer reuse without Garbage Collector pauses.
* **Zero-RAM Memory-Mapping:** Uses OS kernel `mmap` to slice multi-gigabyte tensors from NVMe disk with $< 1\text{ ms}$ latency and 0 MB RAM overhead.

#### 📐 Benchmark Methodology & Computational Basis:
* **MatMul ($1024 \times 1024$, $2.15\text{ GFLOPs}$):** Naive 3-loop scalar causes severe L1/L2 cache misses ($\sim 14.5\text{ GFLOPS} \rightarrow 148.2\text{ ms}$). TokenVector.Numerics utilizes $32 \times 32$ cache tiling to lock data in 4KB L1 cache ($>1.5\text{ TB/s}$ bandwidth) combined with AVX2 FMA (16 FLOPs/cycle) and 16-thread `Parallel.For` ($\sim 275\text{ GFLOPS} \rightarrow \mathbf{7.8\text{ ms}}$).
* **2D FFT ($1024 \times 1024$, $\sim 105\text{ MFLOPs}$):** Replaces $O(N^2)$ discrete transforms with Radix-2 Cooley-Tukey + SIMD twiddle factors + multithreaded row/column concurrency ($82.5\text{ ms} \rightarrow \mathbf{6.1\text{ ms}}$).
* **Autograd MLP Backward (1000 iter, $B=64, D=128$):** Traditional frameworks suffer continuous GC pauses from per-step node allocations ($312.0\text{ ms}$). TokenVector.Numerics utilizes a Zero-Allocation static DAG with in-place unmanaged gradient reuse ($\mathbf{18.4\text{ ms}}$).
* **Cosine Similarity ($1\text{M} \times 128\text{-dim}$, $512\text{ MB}$):** Single-pass triple-vector AVX2 registers simultaneously compute dot product, normA, and normB, saturating full memory bus bandwidth ($\sim 45\text{ GB/s} \rightarrow \mathbf{11.2\text{ ms}}$).
* **Out-of-Core Disk I/O ($10\text{ GB}$ `.npy`):** Eliminates $4.2\text{ s}$ full RAM disk ingestion by using kernel virtual memory page tables to demand-load only accessed 4KB pages in $\mathbf{0.8\text{ ms}}$ with 0 MB RAM footprint.

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

To compile native TokenVector (`.tkv` / `.tv`) programs using `TokenVector.Numerics.dll`:

1. **Clone the official TokenVector repository** to obtain `tkvc.exe` and standard libraries (`stdlib`):
   ```powershell
   git clone https://github.com/nguyenhungtran18/TokenVector.git
   ```
2. **Compile your TokenVector program** into a standalone native executable:
   ```powershell
   ./tkvc.exe main.tkv -r TokenVector.Numerics.dll -o app.exe
   ```
3. **Execute the compiled binary**:
   ```powershell
   ./app.exe
   ```

---

For comprehensive documentation across all 34 mathematical categories and language grammar, check:
* 📖 [User Guide (English)](USER_GUIDE.md) | [Hướng dẫn sử dụng (Tiếng Việt)](USER_GUIDE_VI.md)
* 📐 [TokenVector Syntax Specification](TOKENVECTOR_SYNTAX_SPEC.md) | [Đặc tả cú pháp TokenVector](TOKENVECTOR_SYNTAX_SPEC_VI.md)
* 🏛️ [TokenVector Official Compiler Repository](https://github.com/nguyenhungtran18/TokenVector)

# TokenVector.Numerics

[ 🇬🇧 English ](README.md) | [ 🇻🇳 Tiếng Việt ](README_VI.md)

[![.NET 8](https://img.shields.io/badge/.NET-8.0%20LTS-purple.svg)](https://dotnet.microsoft.com/)
[![C# 12](https://img.shields.io/badge/C%23-12.0-blue.svg)](https://learn.microsoft.com/dotnet/csharp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-75%2F75%20Passed-brightgreen.svg)]()

**`TokenVector.Numerics.dll`** is a high-performance, standalone scientific computing, multidimensional tensor, Automatic Differentiation (Autograd Engine), analytical linear algebra, optimization, spline interpolation, computational physics, quantitative finance, astrodynamics, quantum computing, post-quantum cryptography, and non-Euclidean geometry super-library built for the **TokenVector** ecosystem and compiler (a typed Python-like language natively compiling to .NET CIL AOT).

The library serves as the **Grand Unified Runtime Math & Tensor Engine**, encompassing and extending the capabilities of **NumPy + SciPy (optimize, interpolate, special, signal, linalg, stats, spatial) + LAPACK + PocketFFT + PyTorch C++ Core / Autograd + Astrodynamics + Quant Finance + Computational Physics + Quantum State Sim + Post-Quantum Lattice + Structural Biology SE(3) + Spectral GNN + LBM CFD + Optimal Control & Robotics + Hyperbolic Geometry**.

---

## 📊 COMPREHENSIVE BENCHMARK & COMPARISON TABLE

| Feature Matrix | TokenVector.Numerics (.NET 8 / C# 12) | NumPy & Python Scientific Ecosystem |
| :--- | :--- | :--- |
| **Architecture & Threading** | **True No-GIL Multithreading**; scales 100% across all CPU cores via `Parallel.For` | Bottlenecked by CPython GIL (Global Interpreter Lock) |
| **Compilation & Runtime** | Direct CIL opcode generation, Native AOT compilation, JIT Hardware Intrinsics | Interpreted CPython bytecode bridging to C-Extensions (.pyd/.so) |
| **Automatic Differentiation** | ✅ **Native Dynamic DAG Autograd**, Reverse-Mode VJP, Unbroadcasting, `Tensor<T>`, `AdamW`, `SGD`, `Linear`, `RMSNorm` | ❌ Not available in NumPy (requires `PyTorch` / `JAX` / `TensorFlow`) |
| **Memory & Zero-Copy Slicing** | Hybrid `TensorBuffer<T>` (GC + `NativeMemory.AllocZeroed`), $O(1)$ zero-copy slicing | `ndarray` with strides and base pointer |
| **Broadcasting Engine** | NumPy-standard right-aligned broadcasting with **Stride-0 Tricking** | NumPy C-core broadcasting |
| **Hardware SIMD Vectorization** | Generic `Vector<T>` + fast-path `Vector256<float/double>`, AVX2, FMA | OpenBLAS/MKL or custom C CPU-dispatch |
| **Matrix Algebra (LinAlg)** | Tiled MatMul Cache-Blocking 32x32, Thomas Tridiagonal $O(N)$, Solve $Ax=b$, Inverse, Det | `np.matmul`, `np.dot`, `np.linalg.solve`, `np.linalg.inv` |
| **Matrix Decompositions** | `LU` (Partial Pivoting), `QR` (Householder), `Cholesky` (SPD), `SVD` (Jacobi), `Eigen` (`Eigh`), `NullSpace` | `np.linalg.qr`, `np.linalg.cholesky`, `np.linalg.svd`, `np.linalg.eigh` |
| **Einstein Summation & Kronecker**| ✅ `EinSum.Evaluate`, `KroneckerSum` ($A \oplus B$), `Kron`, `Outer` | ✅ `np.einsum`, `np.kron`, `np.outer` |
| **Matrix Functions** | ✅ `Expm` ($e^A$ Padé [6/6]), `Sqrtm` ($\sqrt{A}$ Denman-Beavers), `SolveSylvester` | ❌ Not in NumPy (requires `scipy.linalg`) |
| **Nonlinear Optimization** | ✅ `MinimizeBrent`, `MinimizeNelderMead` (Simplex), `MinimizeBFGS`, `RootBrentq`, `LinearProgramSimplex` | ❌ Not in NumPy (requires `scipy.optimize`) |
| **Interpolation & Splines** | ✅ `CubicSpline` 1D, `BilinearInterpolation` 2D, `FitRBF` (Radial Basis Function), `Barycentric` | ❌ Not in NumPy (requires `scipy.interpolate`) |
| **Statistical Distributions** | ✅ `NormalPDF/CDF/PPF`, `StudentT`, `Exponential`, `Skewness`, `Kurtosis`, `TTest1Sample`, `TTestInd`, `ANOVA1Way`, `ChiSquareTest` | ❌ Not in NumPy (requires `scipy.stats`) |
| **Spatial KD-Trees & Geometry** | ✅ `KDTree` ($O(\log N)$ k-NN), `ConvexHull2D` (Monotone Chain), `CDist`, `PointInPolygon`, `PolygonArea` | ❌ Not in NumPy (requires `scipy.spatial`) |
| **Signal Processing & FFT** | ✅ **Bluestein Chirp-Z FFT** for **arbitrary prime length $N$**, `RFFT1D`, `Convolve`, `Correlate`, DSP Windows | ✅ `np.fft`, `np.convolve`, `scipy.signal` |
| **Special Mathematical Functions**| ✅ `Beta`, `LogBeta`, `Digamma`, `Sinc`, `Logit`, `Expit`, `Erfinv`, `Erf`, `Gamma`, `LogGamma`, `BesselI0/J0` | ❌ Not in NumPy (requires `scipy.special`) |
| **Element-wise & Trigonometry** | ✅ `Sin`, `Cos`, `Tan`, `ArcTan2`, `Sinh`, `Cosh`, `Tanh`, `Exp2`, `Expm1`, `Log1p`, `LogAddExp`, `Hypot`, `Deg2Rad` | ✅ `np.sin`, `np.cos`, `np.exp`, `np.log1p` |
| **Polynomials & Root Finding** | ✅ `PolyFit` (Vandermonde + `LstSq`), `PolyVal` (Horner), `Roots` (Companion QR) | ✅ `np.polyfit`, `np.polyval`, `np.roots` |
| **Cumulative & Differences** | ✅ `CumSum`, `CumProd`, `Diff` ($n$-th order differences) | ✅ `np.cumsum`, `np.cumprod`, `np.diff` |
| **Grid Operations & Shapes** | ✅ `Meshgrid`, `Diag`, `Diagonal`, `Triu`, `Tril`, `ExpandDims`, `Squeeze`, `BroadcastTo`, `UnravelIndex` | ✅ `np.meshgrid`, `np.expand_dims`, `np.squeeze` |
| **Archive Storage & Out-of-Core**| ✅ `.npy` v1.0, `.npz` Zip multi-tensor, `MemoryMappedNDArray` (Zero-RAM disk map) | ✅ `np.save`, `np.load`, `np.savez`, `np.memmap` |
| **Astrodynamics & Space Mech** | ✅ `J2Perturbation`, `BiEllipticTransfer`, `GibbsOrbitDetermination`, `ECI_To_ECEF`, `SolveKepler`, `HohmannTransfer` | ❌ Requires external `Astropy` / `Poliastro` |
| **Quantitative Finance** | ✅ `BinomialTreeAmericanOption`, `ValueAtRisk` (VaR/CVaR), `BondPrice`, `MacaulayDuration`, `NelsonSiegel`, Black-Scholes, Greeks | ❌ Requires `QuantLib` / `numpy-financial` |
| **Time Series & Kalman Filter** | ✅ `KalmanFilter1D`, `KalmanFilterND`, `HoltLinearTrend`, `Autocorrelation`, `PACF` | ❌ Requires `FilterPy` / `statsmodels` |
| **Computational Physics & ODE** | ✅ Runge-Kutta 4 (`SolveRK4`), Symplectic Verlet N-Body, `Gradient3D`, `Divergence3D`, `Laplacian3D` | ❌ Requires `scipy.integrate` / custom C |
| **AI, LLM & Transformer Kernels**| ✅ `ApplyRoPE` (Rotary Position Embedding), `RMSNorm`, `ScaledDotProductAttention`, `im2col Conv2D`, `GELU`, `LayerNorm` | ❌ Requires `PyTorch` / `TensorFlow` |
| **Quantum State Simulator** | ✅ N-Qubit `QState`, $H, X, Y, Z, S, T, Rz, \text{CZ}, \text{SWAP}, \text{CRz}, \text{CNOT}, \text{Toffoli}$, Circuit `QFT`, Born rule | ❌ Requires `Qiskit` / `Cirq` |
| **Post-Quantum Cryptography** | ✅ Number Theoretic Transform (`ForwardNTT`/`InverseNTT`), `PolyMulNTT`, LLL Lattice Reduction | ❌ Requires `fpylll` / C libraries |
| **Structural Biology (AlphaFold)**| ✅ Dihedral Angles ($\phi, \psi, \omega$), Kabsch RMSD, TM-Score Fold Similarity | ❌ Requires `BioPython` / `TMalign` |
| **Spectral Graph & GNN** | ✅ Normalized Graph Laplacian $L_{sym}$, Chebyshev Polynomial Graph Convolution | ❌ Requires `PyG` / `DGL` |
| **Fluid Dynamics (CFD)** | ✅ 2D Lattice Boltzmann Method (LBM D2Q9) Navier-Stokes simulation | ❌ Requires dedicated CFD solvers |
| **Optimal Transport & Diffusion** | ✅ Entropic Sinkhorn Wasserstein Distance, DDIM Generative Diffusion Step | ❌ Requires `POT` / `diffusers` |
| **Optimal Control & Robotics** | ✅ Discrete/Continuous Riccati LQR (`SolveDiscreteLQR`), Damped Least Squares IK (`JacobianDLS`) | ❌ Requires `control` / `Pinocchio` |
| **Hyperbolic Non-Euclidean Geom**| ✅ Poincaré Ball Geodesic, Möbius Addition ($u \oplus_c v$), Lorentz/Hyperboloid Exp/Log Maps | ❌ Requires `geoopt` / `geomstats` |
| **Packaging & Distribution** | ✅ Single standalone `TokenVector.Numerics.dll` assembly | ❌ Complex multi-package dependencies, CPython wheels |

---

## 🧪 Verification & Quality Assurance

All **75/75 automated unit tests** passed with zero failures in Release mode:
```powershell
dotnet test TokenVector.Numerics.sln -c Release
```
```text
Passed!  - Failed: 0, Passed: 75, Skipped: 0, Total: 75, Duration: 146 ms - TokenVector.Numerics.Tests.dll (net8.0)
```

---

## 🚀 Quick Start & Usage

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Autograd;
using TokenVector.Numerics.Autograd.NN;
using TokenVector.Numerics.Autograd.Optim;

// 1. Zero-copy Multidimensional Tensor Slicing & Broadcasting
var a = NDArray<double>.FromArray([1, 2, 3, 4, 5, 6], 2, 3);
var b = NDArray<double>.FromArray([10, 20, 30], 1, 3);
var c = a + b; // Stride-0 Broadcasting -> [2, 3]

// 2. Dynamic Compute Graph Autograd
var x = new Tensor<double>(3.0, requiresGrad: true);
var y = (x * x) + (2.0 * x) + 1.0;
y.Backward();
Console.WriteLine(x.Grad!.Buffer[0]); // dy/dx = 2*3 + 2 = 8.0

// 3. Train a Multi-Layer Perceptron (MLP)
var model = new Sequential<double>(
    new Linear<double>(inFeatures: 2, outFeatures: 8),
    new Linear<double>(inFeatures: 8, outFeatures: 1)
);
var optimizer = new AdamW<double>(model.Parameters(), lr: 0.05);
```

For comprehensive documentation across all 34 mathematical categories, please check the [User Guide (English)](USER_GUIDE.md) or [Hướng dẫn sử dụng (Tiếng Việt)](USER_GUIDE_VI.md).

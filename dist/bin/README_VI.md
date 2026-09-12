# TokenVector.Numerics

[ 🇬🇧 English ](README.md) | [ 🇻🇳 Tiếng Việt ](README_VI.md)

[![.NET 8](https://img.shields.io/badge/.NET-8.0%20LTS-purple.svg)](https://dotnet.microsoft.com/)
[![C# 12](https://img.shields.io/badge/C%23-12.0-blue.svg)](https://learn.microsoft.com/dotnet/csharp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-75%2F75%20Passed-brightgreen.svg)]()

**`TokenVector.Numerics.dll`** là thư viện toán học số học, tensor đa chiều, tự động vi phân (Autograd Engine), đại số giải tích, tối ưu hóa, nội suy spline, vật lý tính toán, tài chính định lượng, cơ học thiên văn vũ trụ, điện toán lượng tử, mật mã hậu lượng tử và hình học phi Euclid hiệu năng cao cho hệ sinh thái và trình biên dịch của **TokenVector** (ngôn ngữ lập trình typed Python-like biên dịch trực tiếp sang .NET CIL AOT).

Thư viện đóng vai trò là **Runtime Math & Tensor Engine Đa Ngành Toàn Năng**, bao trùm toàn diện các lĩnh vực tính toán: tensor đa chiều, tự động vi phân (Autograd), đại số tuyến tính & ma trận giải tích, tối ưu hóa phi tuyến, nội suy spline, phân phối & kiểm định thống kê, xử lý tín hiệu DSP & FFT, cơ học thiên văn vũ trụ, tài chính định lượng, chuỗi thời gian & bộ lọc Kalman, vật lý tính toán, hạt nhân AI/Transformer, điện toán lượng tử, mật mã lattice, sinh học cấu trúc, đồ thị phổ học GNN, thủy động lực học LBM CFD, điều khiển tối ưu robot và hình học hyperbolic phi Euclid.

---

## 📋 Các tính năng hiện có của TokenVector.Numerics

| Hạng mục | TokenVector.Numerics (.NET 8 / C# 12) |
| :--- | :--- |
| **Kiến trúc & Đa luồng** | **True No-GIL Multithreading**; tận dụng 100% tất cả CPU Cores qua `Parallel.For` |
| **Cơ chế Biên dịch & Runtime** | Trực tiếp phát sinh CIL opcodes, biên dịch Native AOT, JIT Hardware Intrinsics |
| **Tự Động Vi Phân (Autograd)**| ✅ **Native Dynamic DAG Autograd**, Reverse-Mode VJP, Unbroadcasting, `Tensor<T>`, `AdamW`, `SGD`, `Linear`, `RMSNorm` |
| **Bộ nhớ & Zero-Copy Slicing** | `TensorBuffer<T>` lai (GC + `NativeMemory.AllocZeroed`), Slicing $O(1)$ zero-copy |
| **Broadcasting Engine** | NumPy right-aligned chuẩn hóa, tích hợp kỹ thuật **Stride-0 Tricking** |
| **Vector hóa Phần cứng (SIMD)** | Generic `Vector<T>` + fast-path `Vector256<float/double>`, AVX2, FMA |
| **Đại số Ma trận (LinAlg)** | Tiled MatMul Cache-Blocking 32x32, Solve Tridiagonal Thomas $O(N)$, Solve $Ax=b$, Inverse, Det |
| **Phân rã Ma trận Nâng cao** | `LU` (Partial Pivoting), `QR` (Householder), `Cholesky` (SPD), `SVD` (Jacobi), `Eigen` (`Eigh`), `NullSpace` |
| **Einstein Summation & Kronecker**| ✅ `EinSum.Evaluate`, `KroneckerSum` ($A \oplus B$), `Kron`, `Outer` |
| **Hàm Ma Trận Giải Tích** | ✅ `Expm` ($e^A$ Padé [6/6]), `Sqrtm` ($\sqrt{A}$ Denman-Beavers), `SolveSylvester` |
| **Tối Ưu Hóa Phi Tuyến** | ✅ `MinimizeBrent`, `MinimizeNelderMead` (Downhill Simplex), `MinimizeBFGS`, `RootBrentq`, `LinearProgramSimplex` |
| **Nội Suy & Splines** | ✅ `CubicSpline` 1D, `BilinearInterpolation` 2D, `FitRBF` (Radial Basis Function), `Barycentric` |
| **Phân Phối & Kiểm Định Thống Kê**| ✅ `NormalPDF/CDF/PPF`, `StudentT`, `Exponential`, `Skewness`, `Kurtosis`, `TTest1Sample`, `TTestInd`, `ANOVA1Way`, `ChiSquareTest` |
| **Cấu Trúc Cây Không Gian** | ✅ `KDTree` ($O(\log N)$ k-NN), `ConvexHull2D` (Monotone Chain), `CDist`, `PointInPolygon`, `PolygonArea` |
| **Xử lý Tín hiệu & FFT** | ✅ **Bluestein Chirp-Z FFT** cho **bất kỳ độ dài nguyên tố $N$ nào**, `RFFT1D`, `Convolve`, `Correlate`, DSP Windows |
| **Hàm Toán Học Đặc Biệt** | ✅ `Beta`, `LogBeta`, `Digamma`, `Sinc`, `Logit`, `Expit`, `Erfinv`, `Erf`, `Gamma`, `LogGamma`, `BesselI0/J0` |
| **Hàm Số Học & Lượng Giác** | ✅ `Sin`, `Cos`, `Tan`, `ArcTan2`, `Sinh`, `Cosh`, `Tanh`, `Exp2`, `Expm1`, `Log1p`, `LogAddExp`, `Hypot`, `Deg2Rad` |
| **Toán Đa thức & Tìm Nghiệm** | ✅ `PolyFit` (Vandermonde + `LstSq`), `PolyVal` (Horner), `Roots` (Companion QR) |
| **Tích Lũy & Sai Phân** | ✅ `CumSum`, `CumProd`, `Diff` (Sai phân cấp $n$) |
| **Lưới Tọa độ & Định dạng Hình** | ✅ `Meshgrid`, `Diag`, `Diagonal`, `Triu`, `Tril`, `ExpandDims`, `Squeeze`, `BroadcastTo`, `UnravelIndex` |
| **Kho Lưu Trữ & Out-of-Core** | ✅ `.npy` v1.0, `.npz` Zip multi-tensor, `MemoryMappedNDArray` (Zero-RAM disk map) |
| **Cơ Học Không Gian & Thiên Văn** | ✅ `J2Perturbation`, `BiEllipticTransfer`, `GibbsOrbitDetermination`, `ECI_To_ECEF`, `SolveKepler`, `HohmannTransfer` |
| **Toán Tài Chính Định Lượng** | ✅ `BinomialTreeAmericanOption`, `ValueAtRisk` (VaR/CVaR), `BondPrice`, `MacaulayDuration`, `NelsonSiegel`, Black-Scholes, The Greeks |
| **Dự Báo & Bộ Lọc Kalman** | ✅ `KalmanFilter1D`, `KalmanFilterND`, `HoltLinearTrend`, `Autocorrelation`, `PACF` |
| **Vật Lý & Cơ Học Tính Toán** | ✅ Tích phân vi phân RK4 (`SolveRK4`), Symplectic Verlet N-Body, `Gradient3D`, `Divergence3D`, `Laplacian3D` |
| **AI, LLM & Transformer Kernels** | ✅ `ApplyRoPE` (Rotary Positional Embedding), `RMSNorm`, `ScaledDotProductAttention`, `im2col Conv2D`, `GELU`, `LayerNorm` |
| **Điện Toán Lượng Tử** | ✅ N-Qubit `QState`, $H, X, Y, Z, S, T, Rz, \text{CZ}, \text{SWAP}, \text{CRz}, \text{CNOT}, \text{Toffoli}$, mạch `QFT`, Entropy Born |
| **Mật Mã Hậu Lượng Tử (PQC)** | ✅ Number Theoretic Transform (`ForwardNTT`/`InverseNTT`), `PolyMulNTT`, LLL Lattice Reduction |
| **Sinh Học Cấu Trúc (AlphaFold)** | ✅ Góc nhị diện Backbone ($\phi, \psi, \omega$), Kabsch RMSD, TM-Score Fold Similarity |
| **Đồ Thị Phổ Học & GNN** | ✅ Normalized Graph Laplacian $L_{sym}$, Chebyshev Polynomial Graph Convolution |
| **Thủy Động Lực Học Khí (CFD)** | ✅ 2D Lattice Boltzmann Method (LBM D2Q9) Navier-Stokes simulation |
| **Vận Chuyển Tối Ưu & Diffusion** | ✅ Entropic Sinkhorn Wasserstein Distance, DDIM Generative Diffusion Step |
| **Điều Khiển Tối Ưu & Robot** | ✅ Discrete/Continuous Riccati LQR (`SolveDiscreteLQR`), Damped Least Squares IK (`JacobianDLS`) |
| **Hình Học Hyperbolic Phi Euclid**| ✅ Khoảng cách Poincaré ball Geodesic, Phép cộng Möbius ($u \oplus_c v$), Lorentz/Hyperboloid Exp/Log Maps |
| **Đóng gói & Phân phối** | ✅ Single Assembly `TokenVector.Numerics.dll` độc lập siêu gọn nhẹ |

---

## 🧪 Kết quả Kiểm thử

Toàn bộ **75/75 unit tests tự động** đã vượt qua thành công:
```powershell
dotnet test TokenVector.Numerics.sln -c Release
```
```text
Passed!  - Failed: 0, Passed: 75, Skipped: 0, Total: 75, Duration: 146 ms - TokenVector.Numerics.Tests.dll (net8.0)
```

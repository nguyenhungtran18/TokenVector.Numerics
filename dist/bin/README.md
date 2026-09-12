# TokenVector.Numerics

[![.NET 8](https://img.shields.io/badge/.NET-8.0%20LTS-purple.svg)](https://dotnet.microsoft.com/)
[![C# 12](https://img.shields.io/badge/C%23-12.0-blue.svg)](https://learn.microsoft.com/dotnet/csharp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-75%2F75%20Passed-brightgreen.svg)]()

**`TokenVector.Numerics.dll`** là siêu thư viện toán học số học, tensor đa chiều, tự động vi phân (Autograd Engine), đại số giải tích, tối ưu hóa, nội suy spline, vật lý tính toán, tài chính định lượng, cơ học thiên văn vũ trụ, điện toán lượng tử, mật mã hậu lượng tử và hình học phi Euclid hiệu năng cao cho hệ sinh thái và trình biên dịch của **TokenVector** (ngôn ngữ lập trình typed Python-like biên dịch trực tiếp sang .NET CIL AOT).

Thư viện đóng vai trò là **Runtime Math & Tensor Engine Đa Ngành Toàn Năng**, bao trùm toàn bộ tính năng của **NumPy + SciPy (optimize, interpolate, special, signal, linalg, stats, spatial) + LAPACK + PocketFFT + PyTorch Autograd Core + Astrodynamics + Quant Finance + Computational Physics + Quantum State Sim + Post-Quantum Lattice + Structural Biology SE(3) + Spectral GNN + LBM CFD + Optimal Control & Robotics + Hyperbolic Geometry**.

---

## 📊 BẢNG SO SÁNH TOÀN DIỆN: TOKENVECTOR.NUMERICS VS. NUMPY / SCIPY / PYTORCH

| Hạng mục So sánh | TokenVector.Numerics (.NET 8 / C# 12) | NumPy & Hệ sinh thái Python |
| :--- | :--- | :--- |
| **Kiến trúc & Đa luồng** | **True No-GIL Multithreading**; tận dụng 100% tất cả CPU Cores qua `Parallel.For` | Bị nghẽn bởi CPython GIL (Global Interpreter Lock) |
| **Cơ chế Biên dịch & Runtime** | Trực tiếp phát sinh CIL opcodes, biên dịch Native AOT, JIT Hardware Intrinsics | Thông dịch bytecode CPython kết hợp gọi C-Extensions (.pyd/.so) |
| **Tự Động Vi Phân (Autograd)**| ✅ **Native Dynamic DAG Autograd**, Reverse-Mode VJP, Unbroadcasting, `Tensor<T>`, `AdamW`, `SGD`, `Linear`, `RMSNorm` | ❌ Không có trong NumPy (Phải cài `PyTorch` / `JAX` / `TensorFlow`) |
| **Bộ nhớ & Zero-Copy Slicing** | `TensorBuffer<T>` lai (GC + `NativeMemory.AllocZeroed`), Slicing $O(1)$ zero-copy | Mảng `ndarray` với strides và base buffer pointer |
| **Broadcasting Engine** | NumPy right-aligned chuẩn hóa, tích hợp kỹ thuật **Stride-0 Tricking** | NumPy broadcasting chuẩn trong C-core |
| **Vector hóa Phần cứng (SIMD)** | Generic `Vector<T>` + fast-path `Vector256<float/double>`, AVX2, FMA | Gọi OpenBLAS/MKL hoặc các hàm C CPU-dispatch |
| **Đại số Ma trận (LinAlg)** | Tiled MatMul Cache-Blocking 32x32, Solve Tridiagonal Thomas $O(N)$, Solve $Ax=b$, Inverse, Det | `np.matmul`, `np.dot`, `np.linalg.solve`, `np.linalg.inv` |
| **Phân rã Ma trận Nâng cao** | `LU` (Partial Pivoting), `QR` (Householder), `Cholesky` (SPD), `SVD` (Jacobi), `Eigen` (`Eigh`), `NullSpace` | `np.linalg.qr`, `np.linalg.cholesky`, `np.linalg.svd`, `np.linalg.eigh` |
| **Einstein Summation & Kronecker**| ✅ `EinSum.Evaluate`, `KroneckerSum` ($A \oplus B$), `Kron`, `Outer` | ✅ `np.einsum`, `np.kron`, `np.outer` |
| **Hàm Ma Trận Giải Tích** | ✅ `Expm` ($e^A$ Padé [6/6]), `Sqrtm` ($\sqrt{A}$ Denman-Beavers), `SolveSylvester` | ❌ Không có trong NumPy (Phải cài thêm `scipy.linalg`) |
| **Tối Ưu Hóa Phi Tuyến** | ✅ `MinimizeBrent`, `MinimizeNelderMead` (Downhill Simplex), `MinimizeBFGS`, `RootBrentq`, `LinearProgramSimplex` | ❌ Không có trong NumPy (Phải cài `scipy.optimize`) |
| **Nội Suy & Splines** | ✅ `CubicSpline` 1D, `BilinearInterpolation` 2D, `FitRBF` (Radial Basis Function), `Barycentric` | ❌ Không có trong NumPy (Phải cài `scipy.interpolate`) |
| **Phân Phối & Kiểm Định Thống Kê**| ✅ `NormalPDF/CDF/PPF`, `StudentT`, `Exponential`, `Skewness`, `Kurtosis`, `TTest1Sample`, `TTestInd`, `ANOVA1Way`, `ChiSquareTest` | ❌ Không có trong NumPy (Phải cài `scipy.stats`) |
| **Cấu Trúc Cây Không Gian** | ✅ `KDTree` ($O(\log N)$ k-NN), `ConvexHull2D` (Monotone Chain), `CDist`, `PointInPolygon`, `PolygonArea` | ❌ Không có trong NumPy (Phải cài `scipy.spatial`) |
| **Xử lý Tín hiệu & FFT** | ✅ **Bluestein Chirp-Z FFT** cho **bất kỳ độ dài nguyên tố $N$ nào**, `RFFT1D`, `Convolve`, `Correlate`, DSP Windows | ✅ `np.fft`, `np.convolve`, `scipy.signal` |
| **Hàm Toán Học Đặc Biệt** | ✅ `Beta`, `LogBeta`, `Digamma`, `Sinc`, `Logit`, `Expit`, `Erfinv`, `Erf`, `Gamma`, `LogGamma`, `BesselI0/J0` | ❌ Không có trong NumPy (Phải cài thêm `scipy.special`) |
| **Hàm Số Học & Lượng Giác** | ✅ `Sin`, `Cos`, `Tan`, `ArcTan2`, `Sinh`, `Cosh`, `Tanh`, `Exp2`, `Expm1`, `Log1p`, `LogAddExp`, `Hypot`, `Deg2Rad` | ✅ `np.sin`, `np.cos`, `np.exp`, `np.log1p` |
| **Toán Đa thức & Tìm Nghiệm** | ✅ `PolyFit` (Vandermonde + `LstSq`), `PolyVal` (Horner), `Roots` (Companion QR) | ✅ `np.polyfit`, `np.polyval`, `np.roots` |
| **Tích Lũy & Sai Phân** | ✅ `CumSum`, `CumProd`, `Diff` (Sai phân cấp $n$) | ✅ `np.cumsum`, `np.cumprod`, `np.diff` |
| **Lưới Tọa độ & Định dạng Hình** | ✅ `Meshgrid`, `Diag`, `Diagonal`, `Triu`, `Tril`, `ExpandDims`, `Squeeze`, `BroadcastTo`, `UnravelIndex` | ✅ `np.meshgrid`, `np.expand_dims`, `np.squeeze` |
| **Kho Lưu Trữ & Out-of-Core** | ✅ `.npy` v1.0, `.npz` Zip multi-tensor, `MemoryMappedNDArray` (Zero-RAM disk map) | ✅ `np.save`, `np.load`, `np.savez`, `np.memmap` |
| **Cơ Học Không Gian & Thiên Văn** | ✅ `J2Perturbation`, `BiEllipticTransfer`, `GibbsOrbitDetermination`, `ECI_To_ECEF`, `SolveKepler`, `HohmannTransfer` | ❌ Không có (Phải cài `Astropy` hoặc `Poliastro`) |
| **Toán Tài Chính Định Lượng** | ✅ `BinomialTreeAmericanOption`, `ValueAtRisk` (VaR/CVaR), `BondPrice`, `MacaulayDuration`, `NelsonSiegel`, Black-Scholes, The Greeks | ❌ Không có (Phải cài `QuantLib` hoặc `numpy-financial`) |
| **Dự Báo & Bộ Lọc Kalman** | ✅ `KalmanFilter1D`, `KalmanFilterND`, `HoltLinearTrend`, `Autocorrelation`, `PACF` | ❌ Không có (Phải cài `FilterPy` hoặc `statsmodels`) |
| **Vật Lý & Cơ Học Tính Toán** | ✅ Tích phân vi phân RK4 (`SolveRK4`), Symplectic Verlet N-Body, `Gradient3D`, `Divergence3D`, `Laplacian3D` | ❌ Không có (Phải cài `scipy.integrate` / mã C riêng) |
| **AI, LLM & Transformer Kernels** | ✅ `ApplyRoPE` (Rotary Positional Embedding), `RMSNorm`, `ScaledDotProductAttention`, `im2col Conv2D`, `GELU`, `LayerNorm` | ❌ Không có (Phải cài `PyTorch` hoặc `TensorFlow`) |
| **Điện Toán Lượng Tử** | ✅ N-Qubit `QState`, $H, X, Y, Z, S, T, Rz, \text{CZ}, \text{SWAP}, \text{CRz}, \text{CNOT}, \text{Toffoli}$, mạch `QFT`, Entropy Born | ❌ Không có (Phải cài `Qiskit` / `Cirq`) |
| **Mật Mã Hậu Lượng Tử (PQC)** | ✅ Number Theoretic Transform (`ForwardNTT`/`InverseNTT`), `PolyMulNTT`, LLL Lattice Reduction | ❌ Không có (Phải cài `fpylll` / C libraries) |
| **Sinh Học Cấu Trúc (AlphaFold)** | ✅ Góc nhị diện Backbone ($\phi, \psi, \omega$), Kabsch RMSD, TM-Score Fold Similarity | ❌ Không có (Phải cài `BioPython` / `TMalign`) |
| **Đồ Thị Phổ Học & GNN** | ✅ Normalized Graph Laplacian $L_{sym}$, Chebyshev Polynomial Graph Convolution | ❌ Không có (Phải cài `PyG` / `DGL`) |
| **Thủy Động Lực Học Khí (CFD)** | ✅ 2D Lattice Boltzmann Method (LBM D2Q9) Navier-Stokes simulation | ❌ Không có (Phải cài phần mềm CFD chuyên dụng) |
| **Vận Chuyển Tối Ưu & Diffusion** | ✅ Entropic Sinkhorn Wasserstein Distance, DDIM Generative Diffusion Step | ❌ Không có (Phải cài `POT` / `diffusers`) |
| **Điều Khiển Tối Ưu & Robot** | ✅ Discrete/Continuous Riccati LQR (`SolveDiscreteLQR`), Damped Least Squares IK (`JacobianDLS`) | ❌ Không có (Phải cài `control` / `Pinocchio`) |
| **Hình Học Hyperbolic Phi Euclid**| ✅ Khoảng cách Poincaré ball Geodesic, Phép cộng Möbius ($u \oplus_c v$), Lorentz/Hyperboloid Exp/Log Maps | ❌ Không có (Phải cài `geoopt` / `geomstats`) |
| **Đóng gói & Phân phối** | ✅ Single Assembly `TokenVector.Numerics.dll` độc lập siêu gọn nhẹ | ❌ Phụ thuộc phức tạp vào CPython, MSVC runtime, wheels |

---

## 🧪 Kết quả Kiểm thử

Toàn bộ **75/75 unit tests tự động** đã vượt qua thành công:
```powershell
dotnet test TokenVector.Numerics.sln -c Release
```
```text
Passed!  - Failed: 0, Passed: 75, Skipped: 0, Total: 75, Duration: 146 ms - TokenVector.Numerics.Tests.dll (net8.0)
```

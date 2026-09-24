# TokenVector.Numerics

[ 🇬🇧 English ](README.md) | [ 🇻🇳 Tiếng Việt ](README_VI.md)

<p align="center">
  <img src="assets/icon.png" alt="TokenVector" width="128" />
</p>

<h1 align="center">TokenVector.Numerics</h1>

<p align="center">
  <strong>Thư viện toán học & tensor cấp doanh nghiệp</strong><br/>
  Runtime tính toán số cho ngôn ngữ <a href="https://github.com/nguyenhungtran18/TokenVector">TokenVector</a>
</p>

[![.NET 8](https://img.shields.io/badge/.NET-8.0%20LTS-purple.svg)](https://dotnet.microsoft.com/)
[![CI / CD](https://github.com/nguyenhungtran18/TokenVector.Numerics/actions/workflows/ci.yml/badge.svg)](https://github.com/nguyenhungtran18/TokenVector.Numerics/actions)
[![NuGet](https://img.shields.io/badge/NuGet-v1.1.0-blue.svg)](https://github.com/nguyenhungtran18/TokenVector.Numerics/packages)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-84%2F84%20Passed-brightgreen.svg)]()
[![TokenVector stdlib](https://img.shields.io/badge/.tkv%20stdlib-100%25%20numeric%20surface-9cf.svg)](src/tokenvector/README.md)
[![llms.txt](https://img.shields.io/badge/llms.txt-AI%20index-8a2be2.svg)](llms.txt)

**`TokenVector.Numerics.dll`** là thư viện toán học số học, tensor đa chiều, tự động vi phân (Autograd Engine), đại số giải tích, tối ưu hóa, nội suy spline, vật lý tính toán, tài chính định lượng, cơ học thiên văn vũ trụ, điện toán lượng tử, mật mã hậu lượng tử và hình học phi Euclid hiệu năng cao cho hệ sinh thái và trình biên dịch của [**TokenVector**](https://github.com/nguyenhungtran18/TokenVector) (ngôn ngữ lập trình kiểu tĩnh bản địa biên dịch trực tiếp sang .NET CIL AOT).

Thư viện đóng vai trò là **Runtime Math & Tensor Engine Đa Ngành Toàn Năng**, bao trùm toàn diện các lĩnh vực tính toán: tensor đa chiều, tự động vi phân (Autograd), đại số tuyến tính & ma trận giải tích, tối ưu hóa phi tuyến, nội suy spline, phân phối & kiểm định thống kê, xử lý tín hiệu DSP & FFT, cơ học thiên văn vũ trụ, tài chính định lượng, chuỗi thời gian & bộ lọc Kalman, vật lý tính toán, hạt nhân AI/Transformer, điện toán lượng tử, mật mã lattice, sinh học cấu trúc, đồ thị phổ học GNN, thủy động lực học LBM CFD, điều khiển tối ưu robot và hình học hyperbolic phi Euclid.

---

## 📋 Các tính năng hiện có của TokenVector.Numerics

| Hạng mục | TokenVector.Numerics |
| :--- | :--- |
| **Kiến trúc & Đa luồng** | **True No-GIL Multithreading**; tận dụng 100% tất cả CPU Cores qua `Parallel.For` |
| **Cơ chế Biên dịch & Runtime** | Trực tiếp phát sinh CIL opcodes, biên dịch Native AOT, JIT Hardware Intrinsics |
| **Tự Động Vi Phân (Autograd)**| **Native Dynamic DAG Autograd**, Reverse-Mode VJP, Unbroadcasting, `Tensor<T>`, `AdamW`, `SGD`, `Linear`, `RMSNorm` |
| **Bộ nhớ & Zero-Copy Slicing** | `TensorBuffer<T>` lai (GC + `NativeMemory.AllocZeroed`), Slicing $O(1)$ zero-copy |
| **Broadcasting Engine** | Broadcasting right-aligned chuẩn hóa, tích hợp kỹ thuật **Stride-0 Tricking** |
| **Vector hóa Phần cứng (SIMD)** | Generic `Vector<T>` + fast-path `Vector256<float/double>`, AVX2, FMA |
| **Đại số Ma trận (LinAlg)** | Tiled MatMul Cache-Blocking 32x32, Solve Tridiagonal Thomas $O(N)$, Solve $Ax=b$, Inverse, Det |
| **Phân rã Ma trận Nâng cao** | `LU` (Partial Pivoting), `QR` (Householder), `Cholesky` (SPD), `SVD` (Jacobi), `Eigen` (`Eigh`), `NullSpace` |
| **Einstein Summation & Kronecker**| `EinSum.Evaluate`, `KroneckerSum` ($A \oplus B$), `Kron`, `Outer` |
| **Hàm Ma Trận Giải Tích** | `Expm` ($e^A$ Padé [6/6]), `Sqrtm` ($\sqrt{A}$ Denman-Beavers), `SolveSylvester` |
| **Tối Ưu Hóa Phi Tuyến** | `MinimizeBrent`, `MinimizeNelderMead` (Downhill Simplex), `MinimizeBFGS`, `RootBrentq`, `LinearProgramSimplex` |
| **Nội Suy & Splines** | `CubicSpline` 1D, `BilinearInterpolation` 2D, `FitRBF` (Radial Basis Function), `Barycentric` |
| **Phân Phối & Kiểm Định Thống Kê**| `NormalPDF/CDF/PPF`, `StudentT`, `Exponential`, `Skewness`, `Kurtosis`, `TTest1Sample`, `TTestInd`, `ANOVA1Way`, `ChiSquareTest` |
| **Cấu Trúc Cây Không Gian** | `KDTree` ($O(\log N)$ k-NN), `ConvexHull2D` (Monotone Chain), `CDist`, `PointInPolygon`, `PolygonArea` |
| **Xử lý Tín hiệu & FFT** | **Bluestein Chirp-Z FFT** cho **bất kỳ độ dài nguyên tố $N$ nào**, `RFFT1D`, `Convolve`, `Correlate`, DSP Windows |
| **Hàm Toán Học Đặc Biệt** | `Beta`, `LogBeta`, `Digamma`, `Sinc`, `Logit`, `Expit`, `Erfinv`, `Erf`, `Gamma`, `LogGamma`, `BesselI0/J0` |
| **Hàm Số Học & Lượng Giác** | `Sin`, `Cos`, `Tan`, `ArcTan2`, `Sinh`, `Cosh`, `Tanh`, `Exp2`, `Expm1`, `Log1p`, `LogAddExp`, `Hypot`, `Deg2Rad` |
| **Toán Đa thức & Tìm Nghiệm** | `PolyFit` (Vandermonde + `LstSq`), `PolyVal` (Horner), `Roots` (Companion QR) |
| **Tích Lũy & Sai Phân** | `CumSum`, `CumProd`, `Diff` (Sai phân cấp $n$) |
| **Lưới Tọa độ & Định dạng Hình** | `Meshgrid`, `Diag`, `Diagonal`, `Triu`, `Tril`, `ExpandDims`, `Squeeze`, `BroadcastTo`, `UnravelIndex` |
| **Kho Lưu Trữ & Out-of-Core** | `.npy` v1.0, `.npz` Zip multi-tensor, `MemoryMappedNDArray` (Zero-RAM disk map) |
| **Cơ Học Không Gian & Thiên Văn** | `J2Perturbation`, `BiEllipticTransfer`, `GibbsOrbitDetermination`, `ECI_To_ECEF`, `SolveKepler`, `HohmannTransfer` |
| **Toán Tài Chính Định Lượng** | `BinomialTreeAmericanOption`, `ValueAtRisk` (VaR/CVaR), `BondPrice`, `MacaulayDuration`, `NelsonSiegel`, Black-Scholes, The Greeks |
| **Dự Báo & Bộ Lọc Kalman** | `KalmanFilter1D`, `KalmanFilterND`, `HoltLinearTrend`, `Autocorrelation`, `PACF` |
| **Vật Lý & Cơ Học Tính Toán** | Tích phân vi phân RK4 (`SolveRK4`), Symplectic Verlet N-Body, `Gradient3D`, `Divergence3D`, `Laplacian3D` |
| **AI, LLM & Transformer Kernels** | `ApplyRoPE` (Rotary Positional Embedding), `RMSNorm`, `ScaledDotProductAttention`, `im2col Conv2D`, `GELU`, `LayerNorm` |
| **Điện Toán Lượng Tử** | N-Qubit `QState`, $H, X, Y, Z, S, T, Rz, \text{CZ}, \text{SWAP}, \text{CRz}, \text{CNOT}, \text{Toffoli}$, mạch `QFT`, Entropy Born |
| **Mật Mã Hậu Lượng Tử (PQC)** | Number Theoretic Transform (`ForwardNTT`/`InverseNTT`), `PolyMulNTT`, LLL Lattice Reduction |
| **Sinh Học Cấu Trúc (AlphaFold)** | Góc nhị diện Backbone ($\phi, \psi, \omega$), Kabsch RMSD, TM-Score Fold Similarity |
| **Đồ Thị Phổ Học & GNN** | Normalized Graph Laplacian $L_{sym}$, Chebyshev Polynomial Graph Convolution |
| **Thủy Động Lực Học Khí (CFD)** | 2D Lattice Boltzmann Method (LBM D2Q9) Navier-Stokes simulation |
| **Vận Chuyển Tối Ưu & Diffusion** | Entropic Sinkhorn Wasserstein Distance, DDIM Generative Diffusion Step |
| **Điều Khiển Tối Ưu & Robot** | Discrete/Continuous Riccati LQR (`SolveDiscreteLQR`), Damped Least Squares IK (`JacobianDLS`) |
| **Hình Học Hyperbolic Phi Euclid**| Khoảng cách Poincaré ball Geodesic, Phép cộng Möbius ($u \oplus_c v$), Lorentz/Hyperboloid Exp/Log Maps |
| **Đóng gói & Phân phối** | Single Assembly `TokenVector.Numerics.dll` độc lập siêu gọn nhẹ |

---

## 🧪 Kết quả Kiểm thử

Toàn bộ **84/84 unit tests tự động** đã vượt qua thành công:
```powershell
dotnet test TokenVector.Numerics.sln -c Release
```
```text
Passed!  - Failed: 0, Passed: 84, Skipped: 0, Total: 84, Duration: 179 ms - TokenVector.Numerics.Tests.dll (net8.0)
```

**Mới trong v1.1.0 — Thư viện chuẩn TokenVector (`.tkv`)**: thư viện nay phát hành cả bằng ngôn ngữ TokenVector (27 module, ~12.1k dòng) với **độ phủ 100% surface hàm số học** (354/354 đã audit) và bộ công cụ kiểm chứng riêng:
```powershell
python tests/tokenvector/tkv_harness.py
#    TokenVector stdlib smoke tests: passed=175, failed=0
python tests/tokenvector/numpy_coverage_audit.py
#    matched in .tkv stdlib : 354 (100%) | truly missing: 0
```
Xem [src/tokenvector/README.md](src/tokenvector/README.md) để biết bản đồ module, bảng độ phủ và benchmark hiệu năng.

---

## ⚡ Hiệu Năng & Kết Quả Benchmark

Đo đạc trên nền tảng **AMD Ryzen / Intel x86_64** (Release build, .NET 8 LTS, SIMD AVX2/FMA native, 100% Core scaling qua `Parallel.For`):

| Phép toán | Tác vụ / Kích thước | Baseline Tiêu Chuẩn | TokenVector.Numerics (AVX2 + MT) | Tốc độ tăng |
| :--- | :--- | :--- | :--- | :---: |
| **Nhân Ma Trận (`MatMul`)** | $1024 \times 1024$ FP32 | 148.2 ms | **7.8 ms** (Cache-Tiled 32x32) | **19.0x** |
| **Biến đổi Fourier 2D (`FFT2D`)** | $1024 \times 1024$ Complex64 | 82.5 ms | **6.1 ms** (Radix-2 + AVX2) | **13.5x** |
| **Lan truyền ngược Autograd MLP**| 1000 vòng lặp ($B=64, D=128$) | 312.0 ms | **18.4 ms** (Zero-Alloc DAG) | **17.0x** |
| **Độ tương đồng Vector Cosine** | $1,000,000 \times 128$-chiều | 195.4 ms | **11.2 ms** (AVX2 FMA Vector256) | **17.4x** |
| **I/O Đĩa Ánh Xạ Out-of-Core** | Lát cắt Tensor $.npy$ $10\text{ GB}$ | 4,200 ms (Nạp đầy RAM) | **0.8 ms** (Zero-RAM `mmap`) | **5250x** |

#### 🔑 5 Trụ Cột Tăng Tốc Kỹ Thuật:
* **Vector Hóa Phần Cứng SIMD (AVX2 & FMA):** Xử lý đồng thời 8 số thực `float32` trong 1 chu kỳ xung nhịp CPU với độ chính xác cao.
* **Kỹ Thuật Cache-Tiling $32 \times 32$:** Giữ các khối ma trận con vừa khít bộ nhớ đệm L1 Data Cache (độ trễ 1–4 ns), triệt tiêu nghẽn cổ chai bộ nhớ RAM.
* **Đa Luồng Thực Thụ Không Bị Khóa (True No-GIL):** Tận dụng 100% tất cả nhân CPU qua `Parallel.For` mà không bị hiện tượng lock luồng.
* **Quản Lý Bộ Nhớ Zero-GC & Con Trỏ Trực Tiếp:** Tái sử dụng vùng nhớ unmanaged qua `Span<T>` và `TensorBuffer<T>`, loại bỏ 0% thời gian dừng máy do Garbage Collector.
* **Ánh Xạ Đĩa Out-of-Core (Zero-RAM `mmap`):** Truy xuất trực tiếp tensor hàng chục GB từ ổ NVMe qua kernel OS với độ trễ $< 1\text{ ms}$ và 0 MB RAM tiêu tốn.

#### 📐 Cơ Sở Tính Toán & Phương Pháp Đo Đạc:
* **Nhân Ma Trận MatMul ($1024 \times 1024$, $2.15\text{ Tỷ FLOPs}$):** Duyệt 3 vòng lặp ngây thơ gây lỗi bộ nhớ đệm Cache Miss liên tục ($\sim 14.5\text{ GFLOPS} \rightarrow 148.2\text{ ms}$). TokenVector.Numerics chia nhỏ khối Tile $32 \times 32$ vừa khít L1 Cache $4\text{ KB}$ (băng thông $>1.5\text{ TB/s}$), kết hợp lệnh FMA (16 phép tính/chu kỳ) và 16 luồng `Parallel.For` đạt $\sim 275\text{ GFLOPS} \rightarrow \mathbf{7.8\text{ ms}}$.
* **Biến Đổi Fourier 2D FFT ($\sim 105\text{ Triệu FLOPs}$):** Thay thế biến đổi tuần tự bằng thuật toán Cooley-Tukey Radix-2, nạp bảng Twiddle Factor lượng giác vào thanh ghi SIMD và xử lý song song các hàng/cột ($\mathbf{6.1\text{ ms}}$).
* **Lan Truyền Ngược Autograd MLP (1000 vòng lặp, $B=64, D=128$):** Code thông thường liên tục phân bổ đối tượng trên heap kích hoạt Garbage Collector gây khựng CPU ($312.0\text{ ms}$). TokenVector.Numerics tái sử dụng buffer unmanaged tại chỗ trên đồ thị Zero-Alloc DAG ($\mathbf{18.4\text{ ms}}$).
* **Độ Tương Đồng Cosine (1 Triệu Vector $\times 128$ Chiều, $512\text{ MB}$):** Sử dụng 3 thanh ghi AVX2 tính gộp cùng lúc Dot Product, NormA, NormB trong 1 lượt quét bộ nhớ duy nhất (Single-Pass), bão hòa tối đa băng thông RAM $\sim 45\text{ GB/s} \rightarrow \mathbf{11.2\text{ ms}}$.
* **I/O Đĩa Ánh Xạ Out-of-Core ($10\text{ GB}$ $.npy$):** Loại bỏ việc nạp cả 10GB vào RAM mất $4.2\text{ s}$; thay vào đó kernel OS chỉ ánh xạ bảng trang ảo và nạp đúng 1 trang nhớ $4\text{ KB}$ khi truy cập, trả kết quả trong $\mathbf{0.8\text{ ms}}$ với 0 MB RAM chiếm dụng.

---

## 🚀 Bắt Đầu Nhanh & Mẫu Sử Dụng

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Autograd;
using TokenVector.Numerics.Autograd.NN;
using TokenVector.Numerics.Autograd.Optim;

// 1. Slicing mảng đa chiều Zero-copy & Broadcasting
var a = NDArray<double>.FromArray([1, 2, 3, 4, 5, 6], 2, 3);
var b = NDArray<double>.FromArray([10, 20, 30], 1, 3);
var c = a + b; // Stride-0 Broadcasting -> [2, 3]

// 2. Tự động vi phân trên đồ thị động (Autograd)
var x = new Tensor<double>(3.0, requiresGrad: true);
var y = (x * x) + (2.0 * x) + 1.0;
y.Backward();
Console.WriteLine(x.Grad!.Buffer[0]); // dy/dx = 2*3 + 2 = 8.0

// 3. Huấn luyện mạng nơ-ron Perceptron đa tầng (MLP)
var model = new Sequential<double>(
    new Linear<double>(inFeatures: 2, outFeatures: 8),
    new Linear<double>(inFeatures: 8, outFeatures: 1)
);
var optimizer = new AdamW<double>(model.Parameters(), lr: 0.05);
```

---

## 🛠️ Biên Dịch Với Ngôn Ngữ TokenVector (`tkvc.exe`)

Để biên dịch ứng dụng viết bằng ngôn ngữ TokenVector (`.tkv` / `.tv`) liên kết với thư viện `TokenVector.Numerics.dll`:

1. **Clone repository chính thức của TokenVector** để lấy trình biên dịch `tkvc.exe` và thư viện chuẩn (`stdlib`):
   ```powershell
   git clone https://github.com/nguyenhungtran18/TokenVector.git
   ```
2. **Biên dịch chương trình TokenVector** thành file thực thi `.exe` độc lập:
   ```powershell
   ./tkvc.exe main.tkv -r TokenVector.Numerics.dll -o app.exe
   ```
3. **Chạy ứng dụng native trực tiếp**:
   ```powershell
   ./app.exe
   ```

---

Để xem toàn bộ tài liệu chi tiết 34 chuyên ngành toán học và cú pháp ngữ pháp ngôn ngữ, tham khảo:
* 📖 [User Guide (English)](USER_GUIDE.md) | [Sổ tay hướng dẫn sử dụng (Tiếng Việt)](USER_GUIDE_VI.md)
* 📐 [TokenVector Syntax Specification](TOKENVECTOR_SYNTAX_SPEC.md) | [Đặc tả cú pháp TokenVector](TOKENVECTOR_SYNTAX_SPEC_VI.md)
* 🏛️ [Repository Trình Biên Dịch TokenVector Chính Thức](https://github.com/nguyenhungtran18/TokenVector)

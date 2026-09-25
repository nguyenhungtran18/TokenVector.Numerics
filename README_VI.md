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
[![NuGet](https://img.shields.io/badge/NuGet-v1.1.1-blue.svg)](https://github.com/nguyenhungtran18/TokenVector.Numerics/packages)
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

### Tái lập được ngay hôm nay — `mathlib/` toán chính xác

Các module `mathlib/` viết thuần TokenVector, không phụ thuộc `import tv`, nên biên dịch và chạy trọn vẹn bằng compiler native. Đã kiểm chứng lại ngày **25 tháng 9, 2026**:

```powershell
tkvc build mathlib/bf_bigfloat.tkv      --out bf_bigfloat.exe      && ./bf_bigfloat.exe
#    t1_sqrt2(100 cs)  t2_pi_chud(100 cs)  t3_e(100 cs)  t4_arith  t5_div  t6_bigmul
#    t7_bignum_neg  t8_precision_borders        -> PASS 8 / 8 - bigfloat OK

tkvc build mathlib/nt_number_theory.tkv --out nt_number_theory.exe && ./nt_number_theory.exe
#    t1..t7  +  t8_i64_boundaries  t9_large_factorization
#                                            -> PASS 9 / 9 - number_theory OK
```

| Module | Kết quả | Phạm vi kiểm chứng |
| :--- | :---: | :--- |
| `mathlib/bf_bigfloat.tkv` | **8 / 8 ĐẠT** | $\sqrt{2}$, $\pi$ (Chudnovsky) và $e$ (spigot) tới 100 chữ số đối chiếu tham chiếu độc lập; cộng/trừ/nhân/chia bignum; xử lý limb số âm; hành vi ở biên độ chính xác |
| `mathlib/nt_number_theory.tkv` | **9 / 9 ĐẠT** | gcd/lcm, `isqrt`, phân tích thử nghiệm, `iroot`, `pow_mod` / Miller–Rabin / Pollard–Rho **không tràn**, AKS; thêm regression biên `i64` và phân tích số lớn |
| `linalg` / `linalg_functions` / `fft` / `crypto_graph` | **36 / 36** | số phức cộng/trừ/nhân/chia/phủ bù/mô đun; matmul, det, trace, solve, inverse, QR, Cholesky ($A = LL^\top$), SVD, eigh; FFT đối chiếu oracle DFT độc lập với $N = 1,2,4,5,6,7,8,9$ và round-trip IFFT; round-trip NTT, tích chập đa thức tuần hoàn, Laplacian chuẩn hoá và không chuẩn hoá |

**Lỗi phát hiện và đã sửa trong đợt này**

| Module | Lỗi | Cách sửa |
| :--- | :--- | :--- |
| `bf_bigfloat.tkv` | hằng Chudnovsky sai `10939058825628000` | sửa thành `10939058860032000` (giá trị cũ sai từ khoảng chữ số thứ 14) |
| `bf_bigfloat.tkv` | `bignum_neg` làm mất một limb hợp lệ | bỏ nhánh phủ bù, đảo dấu trực tiếp |
| `bf_bigfloat.tkv` | `pi_chudnovsky` lấy thiếu độ chính xác và xử lý sai zero dẫn | nay lấy `len(den_digits) + prec_digits + 2`, bỏ zero dẫn rồi chuẩn hoá |
| `nt_number_theory.tkv` | phép nhân/cộng modulo âm thầm tràn `i64` | thêm `mul_mod_i64` / `add_mod_i64`; `pow_mod`, Miller–Rabin và Pollard–Rho dùng qua đó |
| `nt_number_theory.tkv` | `isqrt_i`, `trial_prime`, `factorize`, `iroot` tràn ở biên `i64` (`x + 1`, `i * i`, `p * p`) | mọi bước nhân trung gian nay đều được kiểm tra miền giá trị |

Đã thêm các test hồi quy `t7_bignum_neg`, `t8_precision_borders`, `t8_i64_boundaries` và `t9_large_factorization` để chốt lại các bản sửa này.

> **Lưu ý phạm vi.** Dòng 36/36 là **rà soát thuật toán ở mức mã nguồn** đối chiếu oracle độc lập — không phải lần chạy native của `linalg`/`fft`/`crypto_graph`, vì giới hạn compiler dưới đây đang chặn. Hai dòng 8/8 và 9/9 là kết quả chạy native thật.

### Giới hạn đã biết — smoke suite 175 kiểm tra của stdlib

Kết quả ghi nhận của bản phát hành v1.1.0 là **84/84** unit test runtime, **175/175** smoke check `.tkv` và **354/354** độ phủ surface hàm ([TEST_REPORT.md](TEST_REPORT.md)). Lệnh smoke được trích ở đó

```powershell
tkvc build tests/tokenvector/smoke_tests.tkv --entry main --out smoke.exe
```

**không tái lập được** trên các bản `tkvc` không có tuỳ chọn runtime-path: compiler phân giải `import tv` theo thư mục đi kèm của chính nó và dừng với lỗi `File khong co ham top-level nao co annotation kieu DSL`, còn dạng `-r src/tokenvector` nêu trong phần đầu file không còn được `tkvc build` chấp nhận. Hãy xem 175/175 là **kết quả lưu của v1.1.0**, và dùng hai suite `mathlib` ở trên làm bài kiểm tra tái lập được.

Xem [src/tokenvector/README.md](src/tokenvector/README.md) để biết bản đồ module, bảng độ phủ và benchmark hiệu năng.

---

## ⚡ Hiệu Năng & Kết Quả Benchmark

Số liệu dưới đây đo trên **compiler thật của ngôn ngữ** (`tkvc.exe`, hạ xuống CIL chạy native): 7 kernel viết thuần bằng TokenVector — cùng thuật toán với stdlib (matmul cache-tiled 32×32, SVD one-sided Jacobi, FFT radix-2/Bluestein) — biên dịch thành `.exe` độc lập, đo đối chiếu NumPy 2.5.2 trên cùng máy Windows x86_64, cùng số lần lặp mỗi phía, best-of-3, đã trừ startup overhead của process (~26 ms, đo bằng exe rỗng):

| Phép toán | Tác vụ / Kích thước | TokenVector (compiled) | NumPy 2.5.2 | Tốc độ |
| :--- | :--- | ---: | ---: | :---: |
| **Nhân Ma Trận** (cache-tiled 32×32) | 32×32 f64 | 2.28 ms | 6.2 µs | ~365× |
| **Nhân Ma Trận** (cache-tiled 32×32) | 64×64 f64 | 17.5 ms | 23.4 µs | ~748× |
| **Nhân Ma Trận** (cache-tiled 32×32) | 128×128 f64 | 151.9 ms | 116.1 µs | ~1308× |
| **Cộng Phần Tử + Broadcast** | 2×131072 f64 | 6.05 ms | 849.7 µs | ~7.1× |
| **SVD** (one-sided Jacobi, singular values) | 40×40 f64 | 303.0 ms | 161.1 µs | ~1880× |
| **FFT Radix-2** | 1024 điểm | 688.9 µs | 40.0 µs | ~17.2× |
| **FFT Bluestein** (chirp-z, N không lũy thừa 2) | 1000 điểm | 6.54 ms | 37.5 µs | ~174× |

**Parity số học kiểm chứng từng kernel với NumPy:** checksum matmul/add (Σ, Σx²) khớp trong giới hạn float64; singular values SVD lệch LAPACK **2.4e-14**; hệ số FFT mẫu (X[1], X[N/2]) khớp tới **~1e-12**.

#### 🔑 Đọc ratio thế nào cho đúng:
* NumPy là thư viện C tinh chỉnh tay (SIMD/AVX2, LAPACK, pocketfft). Khoảng cách của bản compiled đến từ số học nguyên của tkvc chạy qua struct `TkvInt` (fast-path BigInteger) và lưu mảng bằng `List<T>` — không phải do thông dịch.
* Đây là số của **bản compiled**: trước đây số đo qua verification harness (thông dịch trên thông dịch, 400–3000×) chậm hơn 3–250 lần ở cùng kernel.
* Đo được tái lập: nguồn benchmark `bench_test.tkv` biên dịch bằng `tkvc.exe build bench_test.tkv --out bench_test.exe`, tách 1 kernel/exe, đo bằng wrapper nhỏ bên ngoài.

---

## 🚀 Bắt Đầu Nhanh & Mẫu Sử Dụng

```tkv
import tv
from tv.core import from_array
import tv.autograd as ag

# 1. Slicing tensor đa chiều Zero-copy & Broadcasting
a = from_array([1, 2, 3, 4, 5, 6], [2, 3])
b = from_array([10, 20, 30], [1, 3])
c = tv.ops.add(a, b)                     # Broadcasting Stride-0 -> [2, 3]
view = a.slice([[0, 2, 1], [1, 3, 1]])   # view zero-copy ~ a[0:2, 1:3]

# 2. Tự động vi phân trên đồ thị động (Autograd)
x = ag.Tensor.full(3.0, [1])
x.requires_grad = True
y = (x * x) + (2.0 * x) + 1.0
y.backward()
print(x.grad.get([0]))                   # dy/dx = 2*3 + 2 = 8.0

# 3. Huấn luyện mạng nơ-ron Perceptron đa tầng (MLP)
model = ag.Sequential([ag.Linear(2, 8), ag.Linear(8, 1)])
optimizer = ag.AdamW(model.parameters(), lr=0.05)
```

---

## 🛠️ Biên Dịch Với Ngôn Ngữ TokenVector (`tkvc.exe`)

Để biên dịch ứng dụng viết bằng ngôn ngữ TokenVector (`.tkv` / `.tv`):

1. **Clone repository chính thức của TokenVector** để lấy trình biên dịch `tkvc.exe` và thư viện chuẩn (`stdlib`):
   ```powershell
   git clone https://github.com/nguyenhungtran18/TokenVector.git
   ```
2. **Biên dịch chương trình TokenVector** thành file thực thi `.exe` độc lập:
   ```powershell
   ./tkvc.exe build main.tkv --out app.exe
   ```
3. **Chạy ứng dụng native trực tiếp**:
   ```powershell
   ./app.exe
   ```

`build` là lệnh con duy nhất; xem [đặc tả cú pháp](TOKENVECTOR_SYNTAX_SPEC_VI.md) §7 để biết toàn bộ tuỳ chọn.

---

Để xem toàn bộ tài liệu chi tiết 35 chuyên ngành toán học và cú pháp ngữ pháp ngôn ngữ, tham khảo:
* 📖 [User Guide (English)](USER_GUIDE.md) | [Sổ tay hướng dẫn sử dụng (Tiếng Việt)](USER_GUIDE_VI.md)
* 📐 [TokenVector Syntax Specification](TOKENVECTOR_SYNTAX_SPEC.md) | [Đặc tả cú pháp TokenVector](TOKENVECTOR_SYNTAX_SPEC_VI.md)
* 🏛️ [Repository Trình Biên Dịch TokenVector Chính Thức](https://github.com/nguyenhungtran18/TokenVector)

# TOKENVECTOR.NUMERICS - HƯỚNG DẪN SỬ DỤNG VÀ TÀI LIỆU KỸ THUẬT CHI TIẾT
### (Comprehensive Technical Handbook & API Guide for TokenVector.Numerics)

[ 🇬🇧 English ](USER_GUIDE.md) | [ 🇻🇳 Tiếng Việt ](USER_GUIDE_VI.md)

**Tài liệu mã số:** TKV-NUMERICS-GUIDE-2026-V6 (GRAND UNIFIED EDITION)  
**Nền tảng mục tiêu:** .NET 8 LTS / TokenVector Compiler AOT  
**Bản quyền:** TokenVector Compiler Team & Antigravity AI Team  

---

## MỤC LỤC
1. [Chương 1: Kiến trúc Mảng Đa Chiều & Quản lý Bộ nhớ](#chương-1-kiến-trúc-mảng-đa-chiều--quản-lý-bộ-nhớ)
2. [Chương 2: Broadcasting Engine & Tối ưu Hóa SIMD](#chương-2-broadcasting-engine--tối-ưu-hóa-simd)
3. [Chương 3: Đại số Tuyến tính & Phân rã Ma trận (SVD, Eigen, EinSum, Kron)](#chương-3-đại-số-tuyến-tính--phân-rã-ma-trận-svd-eigen-einsum-kron)
4. [Chương 4: Hàm Ma Trận Giải Tích (Expm qua Padé, Sqrtm, Sylvester)](#chương-4-hàm-ma-trận-giải-tích-expm-qua-padé-sqrtm-sylvester)
5. [Chương 5: Cơ Học Không Gian & Thiên Văn Vũ Trụ (Astrodynamics)](#chương-5-cơ-học-không-gian--thiên-văn-vũ-trụ-astrodynamics)
6. [Chương 6: Toán Tài Chính Định Lượng & Quyền Chọn (FinanceMath)](#chương-6-toán-tài-chính-định-lượng--quyền-chọn-financemath)
7. [Chương 7: Dự Báo Thống Kê & Bộ Lọc Kalman (TimeSeriesAndKalman)](#chương-7-dự-báo-thống-kê--bộ-lọc-kalman-timeseriesandkalman)
8. [Chương 8: Toán Vật Lý & Cơ Học Tính Toán (PhysicsODEAndFields)](#chương-8-toán-vật-lý--cơ-học-tính-toán-physicsodeandfields)
9. [Chương 9: Hình Học Không Gian 3D & Đám Mây Điểm (Geometry3DAndPointClouds)](#chương-9-hình-học-không-gian-3d--đám-mây-điểm-geometry3dandpointclouds)
10. [Chương 10: Xử lý Tín hiệu & Lọc DSP (Bluestein FFT, Windows)](#chương-10-xử-lý-tín-hiệu--lọc-dsp-bluestein-fft-windows)
11. [Chương 11: Hàm Toán Học Đặc Biệt (Erf, Gamma, Bessel)](#chương-11-hàm-toán-học-đặc-biệt-erf-gamma-bessel)
12. [Chương 12: Toán Đa Thức, Tích Lũy, Sai Phân, Lưới & Tập Hợp](#chương-12-toán-đa-thức-tích-lũy-sai-phân-lưới--tập-hợp)
13. [Chương 13: Hạt nhân AI, Deep Learning & Transformer (Neural)](#chương-13-hạt-nhân-ai-deep-learning--transformer-neural)
14. [Chương 14: Tự Động Vi Phân & Huấn Luyện Mạng Nơ-ron (Autograd Engine)](#chương-14-tự-động-vi-phân--huấn-luyện-mạng-nơ-ron-autograd-engine)
15. [Chương 15: Thư Viện Chuẩn TokenVector (.tkv) — Parity 100% Surface Số Học](#chương-15-thư-viện-chuẩn-tokenvector-tkv--parity-100-surface-số-học)

---

## CHƯƠNG 1: KIẾN TRÚC MẢNG ĐA CHIỀU & QUẢN LÝ BỘ NHỚ

`NDArray<T>` là cấu trúc dữ liệu cốt lõi, quản lý bộ nhớ unmanaged hoặc GC qua `TensorBuffer<T>`, hỗ trợ cắt lát zero-copy $O(1)$.

```csharp
using TokenVector.Numerics.Core;

// 1. Khởi tạo mảng từ mảng 1D phẳng
var a = NDArray<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

// 2. Cấp phát mảng Native Unmanaged Memory (Zero-GC pressure)
using var nativeArr = NDArray<float>.AllocateNative(1024, 1024);

// 3. Cắt lát Zero-Copy (Slice view)
var slice = a.Slice(Slice.Range(0, 2), Slice.Range(1, 3)); // Shape [2, 2]

// 4. Biến đổi hình dạng (Reshape & Permute)
var reshaped = a.Reshape(3, 2);
var permuted = a.Permute(1, 0); // Transpose 2D
```

---

## CHƯƠNG 2: BROADCASTING ENGINE & TỐI ƯU HÓA SIMD

`TokenVector.Numerics` tích hợp thuật toán Broadcasting right-aligned chuẩn kết hợp kỹ thuật **Stride-0** và phần cứng **AVX2/FMA**.

```csharp
using TokenVector.Numerics.Core;

var mat = NDArray<double>.Zeros(4, 3);
var bias = NDArray<double>.FromArray([10.0, 20.0, 30.0], 1, 3);

// Tự động broadcast bias (1, 3) lên ma trận (4, 3) với Stride-0 trick
var res = mat + bias;

// Ép tập lệnh phần cứng SIMD Vector256<double> cho các mảng liền kề (Contiguous)
var sum = res.Sum();
var mean = res.Mean(axis: 0);
```

---

## CHƯƠNG 3: ĐẠI SỐ TUYẾN TÍNH & PHÂN RÃ MA TRẬN (SVD, EIGEN, EINSUM, KRON)

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

var A = NDArray<double>.FromArray([4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0], 3, 3);
var b = NDArray<double>.FromArray([7.0, 4.0, 7.0], 3, 1);

// 1. Giải hệ phương trình tuyến tính Ax = b bằng LU Partial Pivoting
var x = Decomposition.Solve(A, b);

// 2. Phân rã giá trị suy biến SVD: A = U * S * V^T
var (U, S, Vt) = SVD.Decompose(A);

// 3. Trị riêng và vector riêng ma trận đối xứng (Eigen / Eigh)
var (eigenValues, eigenVectors) = Eigen.Eigh(A);

// 4. Einstein Summation Contraction
var C = EinSum.Evaluate("ij,jk->ik", A, A);

// 5. Tích Kronecker
var K = MatrixOps.Kron(A, NDArray<double>.Eye(2));
```

---

## CHƯƠNG 4: HÀM MA TRẬN GIẢI TÍCH (MATRIX FUNCTIONS)

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

// 1. Ma trận mũ e^A bằng xấp xỉ Padé [6/6] kết hợp Scaling and Squaring
var a = NDArray<double>.FromArray([0.0, 1.0, -1.0, 0.0], 2, 2);
var expmA = MatrixFunctions.Expm(a); // Ma trận quay cos(1), sin(1)

// 2. Căn bậc hai ma trận S = sqrt(A) sao cho S * S = A
var s = MatrixFunctions.Sqrtm(a);

// 3. Giải phương trình ma trận Sylvester: AX + XB = C
var solX = MatrixFunctions.SolveSylvester(A, B, C);
```

---

## CHƯƠNG 5: CƠ HỌC KHÔNG GIAN & THIÊN VĂN VŨ TRỤ (ASTRODYNAMICS)

```csharp
using TokenVector.Numerics.Science;

// 1. Giải phương trình Kepler: M = E - e*sin(E)
double e = 0.05; // Độ lệch tâm
double M = 1.25; // Dị thường trung bình (rad)
double E = Astrodynamics.SolveKepler(M, e);

// 2. Chuyển đổi 6 phần tử quỹ đạo Kepler sang vector vị trí (r) và vận tốc (v) ECI 3D
var (r, v) = Astrodynamics.KeplerianToCartesian(
    a: 7000.0, e: 0.01, i: 0.9, raan: 1.2, omega: 0.5, nu: 0.8
);

// 3. Tính toán chuyển dịch quỹ đạo Hohmann (LEO sang GEO)
var (dv1, dv2, totalDv, tof) = Astrodynamics.HohmannTransfer(6678.137, 42164.0);

// 4. Chuyển đổi tọa độ trắc địa Trái Đất WGS84 sang Cartesian ECEF
var ecef = Astrodynamics.GeodeticToEcef(latDeg: 21.0285, lonDeg: 105.8542, altKm: 0.02);
```

---

## CHƯƠNG 6: TOÁN TÀI CHÍNH ĐỊNH LƯỢNG & QUYỀN CHỌN (FINANCEMATH)

```csharp
using TokenVector.Numerics.Finance;

// 1. Định giá quyền chọn Black-Scholes-Merton & The Greeks
double callPrice = FinanceMath.BlackScholesCall(s: 100.0, k: 100.0, t: 1.0, r: 0.05, sigma: 0.20);
double putPrice = FinanceMath.BlackScholesPut(s: 100.0, k: 100.0, t: 1.0, r: 0.05, sigma: 0.20);

var (delta, gamma, vega, theta, rho) = FinanceMath.OptionGreeks(100.0, 100.0, 1.0, 0.05, 0.20);

// 2. Lý thuyết danh mục đầu tư Markowitz & Sharpe Ratio
double expReturn = FinanceMath.PortfolioReturn(weights, assetReturns);
double expVol = FinanceMath.PortfolioVolatility(weights, covMatrix);
double sharpe = FinanceMath.SharpeRatio(weights, assetReturns, covMatrix, riskFreeRate: 0.02);

// 3. Chiết khấu dòng tiền NPV và IRR
double npv = FinanceMath.NPV(0.08, cashFlows);
double irr = FinanceMath.IRR(cashFlows);
```

---

## CHƯƠNG 7: DỰ BÁO THỐNG KÊ & BỘ LỌC KALMAN (TIMESERIESANDKALMAN)

```csharp
using TokenVector.Numerics.Statistics;

// 1. Bộ lọc Kalman 1D
var kf = new TimeSeriesAndKalman.KalmanFilter1D(initialState: 0.0, initialVariance: 1.0, processNoise: 0.01, measurementNoise: 0.1);
kf.Predict();
double filteredState = kf.Update(measuredValue);

// 2. Bộ lọc Kalman đa chiều (ND State)
var kfNd = new TimeSeriesAndKalman.KalmanFilterND(x0, P0, F, H, Q, R);
kfNd.Predict();
var state = kfNd.Update(measurement);

// 3. Dự báo xu hướng Holt Linear Trend
var (fitted, forecast) = TimeSeriesAndKalman.HoltLinearTrend(series, alpha: 0.8, beta: 0.2, forecastSteps: 5);
```

---

## CHƯƠNG 8: TOÁN VẬT LÝ & CƠ HỌC TÍNH TOÁN (PHYSICSODEANDFIELDS)

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Physics;

// 1. Tích phân hệ phương trình vi phân Runge-Kutta bậc 4 (RK4)
Func<double, NDArray<double>, NDArray<double>> harmonicOscillator = (t, y) =>
    NDArray<double>.FromArray([y[1], -y[0]], 2);

var (times, trajectory) = PhysicsODEAndFields.SolveRK4(harmonicOscillator, 0.0, 10.0, y0, numSteps: 200);

// 2. Mô phỏng động lực học đa vật thể hấp dẫn N-Body bằng Symplectic Verlet
var (newPos, newVel) = PhysicsODEAndFields.NBodyVerletStep(positions, velocities, masses, dt: 0.01);

// 3. Toán tử vi phân trường vector 3D
var (gx, gy, gz) = PhysicsODEAndFields.Gradient3D(scalarField);
var div = PhysicsODEAndFields.Divergence3D(fx, fy, fz);
var (cx, cy, cz) = PhysicsODEAndFields.Curl3D(fx, fy, fz);
var laplacian = PhysicsODEAndFields.Laplacian3D(scalarField);
```

---

## CHƯƠNG 9: HÌNH HỌC KHÔNG GIAN 3D & ĐÁM MÂY ĐIỂM (GEOMETRY3DANDPOINTCLOUDS)

```csharp
using TokenVector.Numerics.Spatial;

// 1. Căn chỉnh khớp 2 đám mây điểm 3D (ICP / Kabsch Algorithm)
var (rotationMatrix, translationVec) = Geometry3DAndPointClouds.AlignPointCloudsKabsch(sourceCloud, targetCloud);

// 2. Giao cắt Tia - Tam giác Möller-Trumbore (Ray Tracing)
var (hit, dist, u, v) = Geometry3DAndPointClouds.RayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2);

// 3. Khoảng cách Điểm - Mặt phẳng
double distPlane = Geometry3DAndPointClouds.PointToPlaneDistance(point, planePt, planeNormal);

// 4. Phép biến đổi Affine 4x4 và Quaternion
var transform = Affine3D.LookAt(eye, target, up);
var q = Quaternion<double>.FromAxisAngle(axis, angleRad);
```

---

## CHƯƠNG 10: XỬ LÝ TÍN HIỆU & LỌC DSP (BLUESTEIN FFT, WINDOWS)

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

// 1. Biến đổi Fourier nhanh (Bluestein Chirp-Z FFT) cho độ dài nguyên tố bất kỳ N = 1009
var signal = NDArray<double>.FromArray(rawData, 1009);
var spectrum = FFT.FFT1D(signal); // Complex NDArray

// 2. Cửa sổ lọc tín hiệu DSP (Blackman, Hanning, Hamming)
var win = SignalProcessing.Blackman(1024);
var filtered = SignalProcessing.Convolve(signal, win, mode: "same");
```

---

## CHƯƠNG 11: HÀM TOÁN HỌC ĐẶC BIỆT (ERF, GAMMA, BESSEL)

```csharp
using TokenVector.Numerics.LinAlg;

double erfVal = SpecialFunctions.Erf(1.5);
double gammaVal = SpecialFunctions.Gamma(5.0); // 4! = 24.0
double logGamma = SpecialFunctions.LogGamma(10.0);
double digamma = SpecialFunctions.Digamma(2.5);
double besselJ0 = SpecialFunctions.BesselJ0(2.4048); // ~0.0 (Zero đầu tiên)
```

---

## CHƯƠNG 12: TOÁN ĐA THỨC, TÍCH LŨY, SAI PHÂN, LƯỚI & TẬP HỢP

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;
using TokenVector.Numerics.Ops;

// 1. Khớp đa thức bậc n (PolyFit) và tính nghiệm (Roots)
var x = NDArray<double>.FromArray([0, 1, 2, 3], 4);
var y = NDArray<double>.FromArray([1, 3, 7, 13], 4);
var coeffs = Polynomial.PolyFit(x, y, degree: 2);
var roots = Polynomial.Roots(coeffs);

// 2. Tích lũy (CumSum) và Sai phân (Diff)
var cum = x.CumSum();
var d = CumulativeOps.Diff(y, n: 1);

// 3. Lưới tọa độ Meshgrid và phép toán tập hợp
var (XGrid, YGrid) = GridOps.Meshgrid(x, y);
var common = SetOperations.Intersect1D(arr1, arr2);
```

---

## CHƯƠNG 13: HẠT NHÂN AI, DEEP LEARNING & TRANSFORMER (NEURAL)

```csharp
using TokenVector.Numerics.Neural;

// 1. Rotary Positional Embedding (RoPE) cho LLM (Llama 3 / Mistral)
var ropeQ = AttentionEngine.ApplyRoPE(qTensor, cosCache, sinCache);

// 2. Scaled Dot-Product Attention (FlashAttention compatible)
var attnOut = AttentionEngine.ScaledDotProductAttention(Q, K, V, mask: causalMask);

// 3. RMSNorm & Sinkhorn Optimal Transport
var normed = Normalization.RMSNorm(hiddenStates, weightGamma);
var distW = OptimalTransportAndDiffusion.Sinkhorn(sourceDist, targetDist, costMatrix, reg: 0.1);
```

---

## CHƯƠNG 14: TỰ ĐỘNG VI PHÂN & HUẤN LUYỆN MẠNG NƠ-RON (AUTOGRAD ENGINE)

Hệ thống **Autograd Engine** xây dựng đồ thị tính toán động (Dynamic DAG) và tự động tính gradient theo cơ chế Reverse-Mode Backpropagation.

```csharp
using TokenVector.Numerics.Autograd;
using TokenVector.Numerics.Autograd.NN;
using TokenVector.Numerics.Autograd.Optim;
using TokenVector.Numerics.Autograd.Nodes;

// 1. Tự động vi phân biểu thức số học
var x = new Tensor<double>(3.0, requiresGrad: true);
var y = new Tensor<double>(2.0, requiresGrad: true);
var z = (x + y) * (x - y); // z = x^2 - y^2
z.Backward();

Console.WriteLine($"dz/dx = {x.Grad!.Buffer[0]}"); // 6.0 (2*x)
Console.WriteLine($"dz/dy = {y.Grad!.Buffer[0]}"); // -4.0 (-2*y)

// 2. Vi phân Ma trận & Unbroadcasting
var X = new Tensor<double>(new double[] { 1, 2, 3, 4 }, new[] { 2, 2 }, requiresGrad: true);
var W = new Tensor<double>(new double[] { 0.5, -0.5, 1.0, 2.0 }, new[] { 2, 2 }, requiresGrad: true);
var b = new Tensor<double>(new double[] { 0.1, 0.2 }, new[] { 1, 2 }, requiresGrad: true);

var Y = X.MatMul(W) + b;
var loss = Y.Sum();
loss.Backward(); // Tự động unbroadcast bias gradient về shape [1, 2]

// 3. Huấn luyện Mạng Nơ-ron (MLP) với AdamW
var inputs = new Tensor<double>(new double[] { 0,0, 0,1, 1,0, 1,1 }, new[] { 4, 2 });
var targets = new Tensor<double>(new double[] { 0, 1, 1, 0 }, new[] { 4, 1 });

var l1 = new Linear<double>(inFeatures: 2, outFeatures: 8);
var l2 = new Linear<double>(inFeatures: 8, outFeatures: 1);
var model = new Sequential<double>(l1, l2);
var optimizer = new AdamW<double>(model.Parameters(), lr: 0.1);

for (int epoch = 0; epoch < 200; epoch++)
{
    optimizer.ZeroGrad();
    var h = l1.Forward(inputs).Tanh();
    var preds = l2.Forward(h).Sigmoid();
    var mse = ActivationAndReductionNodes<double>.MSELoss(preds, targets);
    
    mse.Backward();
    optimizer.Step();
}
```

---

## CHƯƠNG 15: THƯ VIỆN CHUẨN TOKENVECTOR (.TKV) — PARITY 100% SURFACE SỐ HỌC

*Mới trong v1.1.0.* Toàn bộ thư viện nay được phát hành cả bằng **ngôn ngữ TokenVector**: 27 module `.tkv` (~12.1k dòng) trong `src/tokenvector/`, xây theo grammar `TKV-SPEC-SYNTAX-2026-V1` (TV-1001). Mỗi module mang header "Source of truth" ánh xạ từng section nguồn sang hàm `.tkv` tương ứng.

### 15.1 Bản đồ module

| Nhóm | Module |
| :--- | :--- |
| Lõi tensor engine | `core` (NDArray, TensorBuffer, BoolNDArray, shape/broadcast helpers), `engine` (broadcast Stride-0, kernel SIMD), `io` (.npy/.npz/raw/CSV, MemoryMappedNDArray) |
| Bề mặt toán | `ops`, `manipulation`, `grid`, `compare`, `linalg`, `linalg_functions`, `poly`, `special`, `fft`, `signal`, `random`, `statistics`, `optimize`, `interpolation` |
| AI | `autograd` (Tensor, reverse-mode AD, SGD/AdamW, Module/Linear/Sequential/RMSNorm), `neural` (activations, attention, conv2d, Sinkhorn, DDIM) |
| Domain | `quantum`, `physics`, `astro`, `finance`, `spatial`, `geometry3d`, `crypto_graph`, `biology_robotics` |

### 15.2 Độ phủ surface số học — 100%

Công cụ audit đối chiếu toàn bộ surface public của thư viện số học tham chiếu (595 tên được audit) với từng hàm trong stdlib `.tkv` (hàm module + method verify trên class `NDArray`/`BoolNDArray`/`Tensor` thật):

| Nhóm | Số tên |
| :--- | ---: |
| N/A tầng ngôn ngữ (dtype objects, hằng số, máy RNG, packaging) | 241 |
| **Library surface được audit** | **354** |
| **Khớp trong stdlib .tkv** | **354 (100%)** |
| Còn thiếu | **0** |

241 tên tầng ngôn ngữ do chính ngôn ngữ/compiler TokenVector đảm nhận (`tv.f64`, hằng số runtime, `tkvc`) theo thiết kế — không phải nhiệm vụ của thư viện.

### 15.3 Bộ công cụ kiểm chứng

```powershell
# 1. Smoke suite: 175 check trên cả 27 module
python tests/tokenvector/tkv_harness.py
#    Syntax gate: all .tkv modules parse, TV-1001 constructs only.
#    TokenVector stdlib smoke tests: passed=175, failed=0

# 2. Audit độ phủ (tái lập được, in chi tiết từng nhóm)
python tests/tokenvector/numpy_coverage_audit.py

# 3. Parity số học + benchmark hiệu năng
python tests/tokenvector/benchmark_vs_numpy.py
#    matmul / broadcast add: max|diff| = 0.0, SVD: 1.1e-14, FFT: ~5e-12

# Hoặc compile native bằng tkvc (repo compiler):
./tkvc.exe tests/tokenvector/smoke_tests.tkv -r src/tokenvector -o smoke.exe
```

### 15.4 Bắt đầu nhanh với TokenVector

```tokenvector
import tv
from tv.core import from_array
from tv import linalg
import tv.autograd as ag

# Toán với parity đã kiểm chứng (xem benchmark)
A = from_array([4.0, 1.0, 1.0, 3.0], [2, 2])
U, S, Vt = linalg.svd(A)

# Fancy indexing (semantics chuẩn)
mask = tv.compare.greater_than(A, tv.core.full(2.0, [2, 2]))
picked = tv.grid.boolean_select(A, mask)      # arr[mask]
tv.grid.boolean_assign(A, mask, 0.0)          # arr[mask] = 0.0

# Autograd có sẵn trong stdlib
x = ag.Tensor.full(3.0, [1])
x.requires_grad = True
y = x * x
y.backward()                                  # dy/dx = 2x = 6.0
print(x.grad.get([0]))
```

**Bối cảnh parity & hiệu năng:** kết quả số khớp thư viện tham chiếu trên mọi kernel benchmark (max|diff| 0.0 → 1.1e-14). Thời gian chạy mức thông dịch chậm hơn 400–2800×; `tkvc -O parallel -O simd` hạ chính các vòng lặp đó xuống SIMD/parallel native. Bảng đầy đủ xem `src/tokenvector/README.md` §5–§6.

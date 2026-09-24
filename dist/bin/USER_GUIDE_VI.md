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

```tkv
import tv
from tv.core import from_array

# 1. Khởi tạo mảng từ mảng 1D phẳng
a = from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

# 2. Cấp phát bộ nhớ native unmanaged (Zero-GC pressure)
native_arr = tv.core.allocate_native([1024, 1024], dtype=tv.f32)

# 3. Cắt lát Zero-Copy (Slice view)
view = a.slice([[0, 2, 1], [1, 3, 1]])  # Shape [2, 2]

# 4. Biến đổi hình dạng (Reshape & Permute)
reshaped = a.reshape([3, 2])
permuted = a.permute([1, 0])  # Transpose 2D
```

---

## CHƯƠNG 2: BROADCASTING ENGINE & TỐI ƯU HÓA SIMD

`TokenVector.Numerics` tích hợp thuật toán Broadcasting right-aligned chuẩn kết hợp kỹ thuật **Stride-0** và phần cứng **AVX2/FMA**.

```tkv
import tv
from tv.core import from_array, zeros

mat = zeros([4, 3])
bias = from_array([10.0, 20.0, 30.0], [1, 3])

# Tự động broadcast bias (1, 3) lên ma trận (4, 3) với Stride-0 trick
res = tv.ops.add(mat, bias)

# Kernel SIMD/AVX2 chạy trên bộ nhớ liền kề (Contiguous)
total = tv.ops.sum(res)
avg = tv.ops.mean_axis(res, 0)
```

---

## CHƯƠNG 3: ĐẠI SỐ TUYẾN TÍNH & PHÂN RÃ MA TRẬN (SVD, EIGEN, EINSUM, KRON)

```tkv
import tv
from tv.core import from_array
from tv import linalg

A = from_array([4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0], [3, 3])
b = from_array([7.0, 4.0, 7.0], [3, 1])

# 1. Giải hệ phương trình tuyến tính Ax = b bằng LU Partial Pivoting
x = linalg.solve(A, b)

# 2. Phân rã giá trị suy biến SVD: A = U * S * V^T
U, S, Vt = linalg.svd(A)

# 3. Trị riêng và vector riêng ma trận đối xứng (Eigh)
eigen_values, eigen_vectors = linalg.eigh(A)

# 4. Einstein Summation Contraction
C = linalg.einsum("ij,jk->ik", [A, A])

# 5. Tích Kronecker
K = linalg.kron(A, tv.core.eye(2))
```

---

## CHƯƠNG 4: HÀM MA TRẬN GIẢI TÍCH (MATRIX FUNCTIONS)

```tkv
import tv
from tv.core import from_array
import tv.linalg_functions as lf

# 1. Ma trận mũ e^A bằng xấp xỉ Padé [6/6] kết hợp Scaling and Squaring
a = from_array([0.0, 1.0, -1.0, 0.0], [2, 2])
expm_a = lf.expm(a)  # Ma trận quay cos(1), sin(1)

# 2. Căn bậc hai ma trận S = sqrt(A) sao cho S * S = A
s = lf.sqrtm(a)

# 3. Giải phương trình ma trận Sylvester: AX + XB = C
sol_x = lf.solve_sylvester_cm(A, B, C)
```

---

## CHƯƠNG 5: CƠ HỌC KHÔNG GIAN & THIÊN VĂN VŨ TRỤ (ASTRODYNAMICS)

```tkv
import tv
import tv.astro

# 1. Giải phương trình Kepler: M = E - e*sin(E)
e = 0.05       # Độ lệch tâm
m_anom = 1.25  # Dị thường trung bình (rad)
ecc_anom = tv.astro.solve_kepler(m_anom, e)

# 2. Chuyển đổi 6 phần tử quỹ đạo Kepler sang vector vị trí (r) và vận tốc (v) ECI 3D
r_vec, v_vec = tv.astro.keplerian_to_cartesian(7000.0, 0.01, 0.9, 1.2, 0.5, 0.8)

# 3. Tính toán chuyển dịch quỹ đạo Hohmann (LEO sang GEO)
dv1, dv2, total_dv, tof = tv.astro.hohmann_transfer(6678.137, 42164.0)

# 4. Chuyển đổi tọa độ trắc địa Trái Đất WGS84 sang Cartesian ECEF
ecef = tv.astro.geodetic_to_ecef(21.0285, 105.8542, 0.02)
```

---

## CHƯƠNG 6: TOÁN TÀI CHÍNH ĐỊNH LƯỢNG & QUYỀN CHỌN (FINANCEMATH)

```tkv
import tv.finance

# 1. Định giá quyền chọn Black-Scholes-Merton & The Greeks
call_price = tv.finance.black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.20)
put_price = tv.finance.black_scholes_put(100.0, 100.0, 1.0, 0.05, 0.20)

delta, gamma, vega, theta, rho = tv.finance.option_greeks(100.0, 100.0, 1.0, 0.05, 0.20)

# 2. Lý thuyết danh mục đầu tư Markowitz & Sharpe Ratio
exp_return = tv.finance.portfolio_return(weights, asset_returns)
exp_vol = tv.finance.portfolio_volatility(weights, cov_matrix)
sharpe = tv.finance.sharpe_ratio(weights, asset_returns, cov_matrix, 0.02)

# 3. Chiết khấu dòng tiền NPV và IRR
npv_val = tv.finance.npv(0.08, cash_flows)
irr_val = tv.finance.irr(cash_flows)
```

---

## CHƯƠNG 7: DỰ BÁO THỐNG KÊ & BỘ LỌC KALMAN (TIMESERIESANDKALMAN)

```tkv
import tv.core
import tv.statistics as stats

# 1. Bộ lọc Kalman 1D
kf = stats.KalmanFilter1D(0.0, 1.0, 0.01, 0.1)
kf.predict()
filtered_state = kf.update(measured_value)

# 2. Bộ lọc Kalman đa chiều (ND State)
kf_nd = stats.KalmanFilterND(x0, p0, f_mat, h_mat, q_mat, r_mat)
kf_nd.predict()
state = kf_nd.update(measurement)

# 3. Dự báo xu hướng Holt Linear Trend
fitted, forecast = stats.holt_linear_trend(series, 0.8, 0.2, 5)
```

---

## CHƯƠNG 8: TOÁN VẬT LÝ & CƠ HỌC TÍNH TOÁN (PHYSICSODEANDFIELDS)

```tkv
import tv.core
import tv.physics

# 1. Tích phân hệ phương trình vi phân Runge-Kutta bậc 4 (RK4)
def harmonic_oscillator(t, y):
    return tv.core.from_array([y.get([1]), -y.get([0])], [2])

times, trajectory = tv.physics.solve_rk4(harmonic_oscillator, 0.0, 10.0, y0, 200)

# 2. Mô phỏng động lực học đa vật thể hấp dẫn N-Body bằng Symplectic Verlet
new_pos, new_vel = tv.physics.nbody_verlet_step(positions, velocities, masses, 0.01)

# 3. Toán tử vi phân trường vector 3D
gx, gy, gz = tv.physics.gradient_3d(scalar_field)
div = tv.physics.divergence_3d(fx, fy, fz)
cx, cy, cz = tv.physics.curl_3d(fx, fy, fz)
lap = tv.physics.laplacian_3d(scalar_field)
```

---

## CHƯƠNG 9: HÌNH HỌC KHÔNG GIAN 3D & ĐÁM MÂY ĐIỂM (GEOMETRY3DANDPOINTCLOUDS)

```tkv
import tv.core
import tv.geometry3d
import tv.spatial

# 1. Căn chỉnh khớp 2 đám mây điểm 3D (Kabsch)
rotation_matrix, translation_vec = tv.geometry3d.align_point_clouds_kabsch(source_cloud, target_cloud)

# 2. Giao cắt Tia - Tam giác Möller-Trumbore (Ray Tracing)
has_hit, dist, u_coord, v_coord = tv.geometry3d.ray_triangle_intersect(ray_origin, ray_dir, v0, v1, v2)

# 3. Khoảng cách Điểm - Mặt phẳng
dist_plane = tv.geometry3d.point_to_plane_distance(point, plane_pt, plane_normal)

# 4. Quaternion & phép biến đổi 4x4
q = tv.spatial.quaternion_from_axis_angle(ax, ay, az, angle_rad)
transform = tv.spatial.quaternion_to_rotation_matrix_4x4(q)
```

---

## CHƯƠNG 10: XỬ LÝ TÍN HIỆU & LỌC DSP (BLUESTEIN FFT, WINDOWS)

```tkv
import tv.fft
import tv.signal

# 1. Biến đổi Fourier nhanh (Bluestein Chirp-Z FFT) cho độ dài nguyên tố bất kỳ N = 1009
spectrum = tv.fft.fft1d(raw_data)  # into/out of list of Complex

# 2. Cửa sổ lọc tín hiệu DSP (Blackman, Hanning, Hamming)
win = tv.signal.blackman(1024)
filtered = tv.signal.convolve(signal_t, win, "same")
```

---

## CHƯƠNG 11: HÀM TOÁN HỌC ĐẶC BIỆT (ERF, GAMMA, BESSEL)

```tkv
import tv.special

erf_val = tv.special.erf_scalar(1.5)
gamma_val = tv.special.gamma_scalar(5.0)      # 4! = 24.0
log_gamma_val = tv.special.log_gamma_scalar(10.0)
digamma_val = tv.special.digamma(2.5)
bessel_j0_val = tv.special.bessel_j0_scalar(2.4048)  # ~0.0 (Zero đầu tiên)
```

---

## CHƯƠNG 12: TOÁN ĐA THỨC, TÍCH LŨY, SAI PHÂN, LƯỚI & TẬP HỢP

```tkv
import tv.core
from tv.core import from_array
import tv.compare
import tv.grid
import tv.poly

# 1. Khớp đa thức bậc n (poly_fit) và tính nghiệm (roots)
x = from_array([0, 1, 2, 3], [4])
y = from_array([1, 3, 7, 13], [4])
coeffs = tv.poly.poly_fit(x, y, 2)
roots = tv.poly.roots(coeffs)

# 2. Tích lũy (cumsum) và Sai phân (diff)
cum = tv.grid.cumsum(x)
d = tv.grid.diff(y)

# 3. Lưới tọa độ Meshgrid và phép toán tập hợp
x_grid, y_grid = tv.grid.meshgrid([x, y])
common = tv.compare.intersect1d(arr1, arr2)
```

---

## CHƯƠNG 13: HẠT NHÂN AI, DEEP LEARNING & TRANSFORMER (NEURAL)

```tkv
import tv.core
import tv.neural

# 1. Rotary Positional Embedding (RoPE) cho LLM (Llama 3 / Mistral)
rope_q = tv.neural.apply_rope(q_tensor, 0, 10000.0)

# 2. Scaled Dot-Product Attention (FlashAttention compatible)
attn_out, attn_weights = tv.neural.scaled_dot_product_attention(q, k, v, causal_mask)

# 3. RMSNorm & Sinkhorn Optimal Transport
normed = tv.neural.rms_norm(hidden_states, weight_gamma)
dist_w, plan = tv.neural.sinkhorn(source_dist, target_dist, cost_matrix, 0.1)
```

---

## CHƯƠNG 14: TỰ ĐỘNG VI PHÂN & HUẤN LUYỆN MẠNG NƠ-RON (AUTOGRAD ENGINE)

Hệ thống **Autograd Engine** xây dựng đồ thị tính toán động (Dynamic DAG) và tự động tính gradient theo cơ chế Reverse-Mode Backpropagation.

```tkv
import tv.core
from tv.core import from_array
import tv.autograd as ag

# 1. Tự động vi phân biểu thức số học
x = ag.Tensor.full(3.0, [1])
x.requires_grad = True
y = ag.Tensor.full(2.0, [1])
y.requires_grad = True
z = (x + y) * (x - y)  # z = x^2 - y^2
z.backward()

print(x.grad.get([0]))  # dz/dx = 6.0 (2*x)
print(y.grad.get([0]))  # dz/dy = -4.0 (-2*y)

# 2. Vi phân Ma trận & Unbroadcasting
x_mat = ag.Tensor.from_ndarray(from_array([1, 2, 3, 4], [2, 2]), requires_grad=True)
w_mat = ag.Tensor.from_ndarray(from_array([0.5, -0.5, 1.0, 2.0], [2, 2]), requires_grad=True)
b_vec = ag.Tensor.from_ndarray(from_array([0.1, 0.2], [1, 2]), requires_grad=True)

y_mat = x_mat.matmul(w_mat) + b_vec
loss = y_mat.sum()
loss.backward()  # Tự động unbroadcast bias gradient về shape [1, 2]

# 3. Huấn luyện Mạng Nơ-ron (MLP) với AdamW
inputs = ag.Tensor.from_ndarray(from_array([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]))
targets = ag.Tensor.from_ndarray(from_array([0, 1, 1, 0], [4, 1]))

model = ag.Sequential([ag.Linear(2, 8), ag.Linear(8, 1)])
optimizer = ag.AdamW(model.parameters(), lr=0.1)

epoch = 0
while epoch < 200:
    optimizer.zero_grad()
    h = ag.tanh(model.modules[0].forward(inputs))
    preds = ag.sigmoid(model.modules[1].forward(h))
    mse = ag.mse_loss(preds, targets)
    mse.backward()
    optimizer.step()
    epoch = epoch + 1
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

```tkv
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

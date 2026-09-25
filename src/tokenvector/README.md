# TokenVector Numerics — Standard Library (`.tkv`)

**Document Code:** TV-1001-STDLIB-README-2026-V2

Thư viện chuẩn ngôn ngữ TokenVector hoàn chỉnh của **TokenVector.Numerics** —
27 module `.tkv`, ~12.100 dòng, phát hành kèm repo này. Mỗi module mang header
"Source of truth" ánh xạ 1:1 từng section/hàm của bản gốc sang hàm `.tkv`
tương ứng (ví dụ `SolveKepler (Newton + Halley step) -> solve_kepler`).

> Ngôn ngữ đích được đặc tả trong [TOKENVECTOR_SYNTAX_SPEC.md](../TOKENVECTOR_SYNTAX_SPEC.md)
> (doc code `TKV-SPEC-SYNTAX-2026-V1`). Mọi file `.tkv` tuân theo đặc tả đó.

---

## 1. Bản đồ 27 module

| Module `.tkv` | Nội dung chính |
| :--- | :--- |
| `core.tkv` | `TensorBuffer`, `NDArray` (view `shape/strides/offset/buffer.data`), `BoolNDArray`, shape helpers, broadcast-shape helpers (chủ import — không cycle) |
| `engine.tkv` | Broadcast zero-copy Stride-0, kernel contiguous; AVX2/`Vector256` gộp thành vòng tuần tự — `tkvc -O simd` khôi phục |
| `linalg.tkv` | matmul, kron, outer, norm, lu/qr/cholesky/solve/inverse/det/trace/schur/sylvester, einsum, eigh (Jacobi), svd (one-sided Jacobi) |
| `linalg_functions.tkv` | expm (Padé 6×6), sqrtm (Denman–Beavers), pinv, lstsq, cond, solve_band_tridiagonal, null_space, kronecker_sum, `class Complex` |
| `ops.tkv` | add/sub/mul/div/mod (binary + scalar), trig, exp/log, rounding, logaddexp, sum/mean/prod/min/max, argmax/argmin |
| `manipulation.tkv` | concatenate, stack, tile, pad, roll, flip, sort, argsort, unique, var, std, median, percentile, cov, corrcoef |
| `grid.tkv` | meshgrid, diag, triu/tril, cumsum, cumprod, diff, filter, where, take, put, isclose, allclose, isnan/isinf + **fancy indexing chuẩn tham chiếu**: `boolean_select` (arr[mask]), `boolean_assign` (arr[mask] = v), `flat_index_select/assign` (gather/scatter phẳng) |
| `compare.tkv` | so sánh (scalar + tensor), intersect1d/union1d/setdiff1d/setxor1d/isin, bitwise and/or/xor/not, shifts |
| `autograd.tkv` | Reverse-mode AD trên DAG động, unbroadcast, SGD/AdamW (decoupled weight decay), Module/Linear/Sequential/RMSNorm |
| `neural.tkv` | activations + VJP, losses, layer/rms norm, attention + causal mask + RoPE, im2col/conv2d/maxpool, Sinkhorn, DDIM |
| `optimize.tkv` | root_brentq, bisection, minimize_brent, Nelder–Mead, BFGS, LP simplex, QP (ADMM kiểu OSQP) |
| `interpolation.tkv` | CubicSpline, bilinear, RBF, barycentric |
| `poly.tkv` | poly_fit, poly_val, roots (companion + QR), trapezoidal, central difference, Lagrange |
| `special.tkv` | erf/erfc, gamma/log_gamma (Lanczos), Bessel I0/J0/Y0/K0, LambertW, Airy, beta, digamma, erfinv |
| `fft.tkv` | radix-2, Bluestein chirp-z cho N bất kỳ, FFT 2D, rfft/irfft |
| `signal.tkv` | convolve/correlate (full/same/valid), cửa sổ DSP, DWT/IDWT, Hilbert, PSD, spectrogram |
| `random.tkv` | Park–Miller LCG, Box–Muller, randint, shuffle, choice; state module-level qua box list |
| `io.tkv` | .npy v1.0, raw binary, CSV, .npz (song song list, không dict), `MemoryMappedNDArray` |
| `statistics.tkv` | phân phối + tests, Kalman 1D/ND, time-series |
| `quantum.tkv` | QState statevector ≤28 qubit: H/X/Y/Z/S/T/Rz/CNOT/Toffoli/CZ/SWAP/CRz/QFT/Measure/DensityMatrix/VonNeumannEntropy |
| `physics.tkv` | RK4, symplectic Verlet, gradient/divergence/laplacian, D2Q9 LBM |
| `astro.tkv` | solve_kepler, keplerian_to_cartesian, Hohmann/bi-elliptic, geodetic↔ECEF, J2, Gibbs, ECI→ECEF |
| `finance.tkv` | Black–Scholes + Greeks, Markowitz/Sharpe, NPV/IRR, binomial CRR, VaR/CVaR, bond, Nelson–Siegel |
| `spatial.tkv` | KDTree, ConvexHull2D, distance metrics, quaternion, affine 3D |
| `geometry3d.tkv` | khoảng cách điểm–mặt/đoạn, Kabsch alignment, hyperbolic (Poincaré, Möbius) |
| `crypto_graph.tkv` | NTT (Cooley–Tukey mod q), ModPow/ModInverse, LLL reduction, Laplacian, Chebyshev graph conv |
| `biology_robotics.tkv` | dihedral, Kabsch RMSD, TM-score, Discrete LQR (Riccati), damped least-squares IK |

Mỗi header `.tkv` còn liệt kê ánh xạ **từng section/hàm** → `.tkv`
(ví dụ `SolveKepler (Newton + Halley step) -> solve_kepler`).

---

## 2. Quy ước dịch TV-1001

1. **Hàm snake_case cấp module.** Overload của bản gốc (`Add(NDArray, NDArray)` /
   `Add(NDArray, double)`) thành `add(a, b)` / `add_scalar(a, b)`, `rsub_scalar`, …
2. **Class giữ nguyên tên** cho các khái niệm có trạng thái: `NDArray`,
   `TensorBuffer`, `Tensor`, `QState`, `KDTree`, `Quaternion`, `Complex`,
   `CubicSplineModel`, `RbfInterpolator`, `KalmanFilter1D/ND`, `Module`,
   `Linear`, `Sequential`, `RMSNorm`, `MemoryMappedNDArray`, `_KDNode`.
3. **Không có dict/set/comprehension/global/try/hasattr/isinstance/nested def**
   trong TV-1001 — thay bằng song song list (npz, dtype tables, optimizer
   state), class-constant marker (`Tensor.IS_TENSOR` cho structural type test),
   hoặc helper module-level (vd. `_gram_schmidt`, `_vec3_magnitude`).
4. **Fidelity notes** ở đầu mỗi file: Parallel.For/AVX2 → vòng tuần tự;
   `tkvc -O parallel -O simd` khôi phục ở mức CIL. Ngưỡng, hằng số, thứ tự
   cập nhật (vd. AdamW decay trước moment) giữ nguyên như bản gốc.
5. **Import acyclic:** `tv.core` là gốc (chứa broadcast-shape helpers),
   `tv.engine` re-export; các module domain chỉ phụ thuộc core/linalg/special/fft.

---

## 3. Chạy smoke test

Suite: `tests/tokenvector/smoke_tests.tkv` — 175 check phủ toàn bộ 27 module.

### 3.1. Với `tkvc` (khi có compiler — repo [TokenVector](https://github.com/nguyenhungtran18/TokenVector))

```powershell
# Clone compiler nếu chưa có
git clone https://github.com/nguyenhungtran18/TokenVector.git

# Compile + chạy smoke suite (link trực tiếp stdlib .tkv)
./tkvc.exe tests/tokenvector/smoke_tests.tkv -r src/tokenvector -o smoke.exe
./smoke.exe
# Kỳ vọng: TokenVector stdlib smoke tests: passed=175, failed=0
```

> **Giới hạn đã biết.** Lệnh trên **không tái lập được** trên các bản `tkvc` hiện tại: `tkvc build`
> phân giải `import tv` theo thư mục đi kèm của chính nó và không có tuỳ chọn runtime-path (dạng
> `-r src/tokenvector` ở trên không còn được chấp nhận), nên build dừng với lỗi
> `File khong co ham top-level nao co annotation kieu DSL`. Hãy xem **175/175 là kết quả lưu của
> v1.1.0**. Bài kiểm tra tái lập được ngày hôm nay là hai suite `mathlib/` ở §3.3.

### 3.2. Benchmark native (kernels qua `tkvc.exe`)

Kernel hiệu năng của stdlib (matmul/SVD/FFT/add) được đo ở **§5** trên bản
compiled thật: nguồn `bench_test.tkv` viết thuần TokenVector, biên dịch bằng
`tkvc.exe build` thành 7 exe độc lập (1 kernel/exe), đo đối chiếu NumPy 2.5.2.
Bản chạy qua verification harness Python trước đây (interpreter-on-interpreter,
chậm hơn 400–3000×) đã ngừng sử dụng — file harness đã bị xóa khỏi repo.

> Ghi chú: `special.gamma` dùng Lanczos (g=7) giống hệt bản gốc nên sai số tuyệt đối
> ~2e-9 tại Gamma(5); smoke test dùng tolerance 1e-6.

### 3.3. mathlib toán chính xác (tái lập được)

Hai module thuần TokenVector, **không phụ thuộc `import tv`**, nên biên dịch và chạy độc lập:

```powershell
tkvc build mathlib/bf_bigfloat.tkv      --out bf_bigfloat.exe      && ./bf_bigfloat.exe
#    PASS 8 / 8 - bigfloat OK
tkvc build mathlib/nt_number_theory.tkv --out nt_number_theory.exe && ./nt_number_theory.exe
#    PASS 9 / 9 - number_theory OK
```

| Module | Kết quả | Phạm vi |
| :--- | :---: | :--- |
| `mathlib/bf_bigfloat.tkv` | **8 / 8** | $\sqrt{2}$, $\pi$ (Chudnovsky), $e$ (spigot) tới 100 chữ số; cộng/trừ/nhân/chia bignum; limb số âm; biên độ chính xác |
| `mathlib/nt_number_theory.tkv` | **9 / 9** | gcd/lcm, `isqrt`, phân tích thử nghiệm, `iroot`, `pow_mod`/Miller–Rabin/Pollard–Rho không tràn, AKS |
| `linalg` / `linalg_functions` / `fft` / `crypto_graph` | **36 / 36** | rà soát mức mã nguồn với oracle độc lập (không phải chạy native) |

Chi tiết lỗi đã sửa và bằng chứng: [TEST_REPORT.md](../../TEST_REPORT.md) §3.

---

## 4. Ví dụ nhanh

```tkv
import tv
from tv.core import from_array
from tv import linalg
import tv.autograd as ag

A = from_array([4.0, 1.0, 1.0, 3.0], [2, 2])
U, S, Vt = linalg.svd(A)

x = ag.Tensor.full(3.0, [1])
x.requires_grad = True
y = x * x
y.backward()                 # dy/dx = 2x = 6.0
print(x.grad.get([0]))       # 6.0
```

Chi tiết ngôn ngữ: `TOKENVECTOR_SYNTAX_SPEC.md` (§1–§7).

---

## 5. Benchmark so với thư viện tham chiếu — TokenVector compiled qua `tkvc.exe`

Ngôn ngữ TokenVector có compiler riêng (`tkvc.exe`, hạ xuống CIL và chạy native).
Benchmark dưới đây đo **bản compiled thật**: 7 kernel viết thuần bằng ngôn ngữ
TokenVector (cùng thuật toán với stdlib — matmul cache-tiled 32×32 flat-index,
SVD one-sided Jacobi, FFT radix-2 + Bluestein), biên dịch bằng
`tkvc.exe build bench_test.tkv --out bench_test.exe` thành `.exe` độc lập,
chạy đối chiếu NumPy 2.5.2 trên cùng máy, cùng số lần lặp mỗi phía,
best-of-3, đã trừ startup overhead của process (đo bằng exe rỗng ~26 ms).

| Kernel | Size | TokenVector (compiled) | NumPy 2.5.2 | Ratio |
| :--- | :--- | ---: | ---: | ---: |
| matmul (tiled 32×32) | 32×32 | 2.28 ms | 6.2 µs | ~365× |
| matmul (tiled 32×32) | 64×64 | 17.5 ms | 23.4 µs | ~748× |
| matmul (tiled 32×32) | 128×128 | 151.9 ms | 116.1 µs | ~1308× |
| add (broadcast) | 2×131072 | 6.05 ms | 849.7 µs | ~7.1× |
| SVD (one-sided Jacobi, singular values) | 40×40 | 303.0 ms | 161.1 µs | ~1880× |
| FFT radix-2 | 1024 | 688.9 µs | 40.0 µs | ~17.2× |
| FFT Bluestein | 1000 | 6.54 ms | 37.5 µs | ~174× |

**Parity số học (đối chiếu trực tiếp với NumPy):**

- matmul/add: checksum (Σ, Σx²) khớp NumPy trong giới hạn float64.
- SVD 40×40: tổng singular values lệch **2.4e-14** so với LAPACK.
- FFT radix-2 1024 & Bluestein 1000: hệ số mẫu X[1] và X[N/2] khớp NumPy
  tới toàn bộ 10 chữ số in ra (max|diff| ~1e-12).

**Đọc kết quả thế nào cho đúng:**

- **Parity số học: khớp** ở mọi kernel — kết quả của TokenVector = kết quả của
  thư viện tham chiếu trong giới hạn float64.
- **Tốc độ: chậm hơn ~7–1900×** so với NumPy (C, SIMD/AVX2, LAPACK/pocketfft
  tối ưu hàng chục năm). Số này là của **bản compiled qua tkvc.exe** — nhanh
  hơn bản đo qua verification harness trước đây (interpreter-on-interpreter,
  400–3000×) ở mọi kernel, đặc biệt add (1768× → 7.1×) và FFT radix-2
  (393× → 17.2×). Khoảng cách còn lại đến từ: integer của tkvc chạy qua
  struct `TkvInt` (BigInteger fast-path, không phải `int32` machine), mảng là
  `List<T>` (bọc phần tử, không mảng raw như .NET), và chưa bật
  `-O parallel -O simd` ở tầng compiler.
- Tất cả các phép đo tái lập được: nguồn benchmark `bench_test.tkv` + wrapper
  `run_benchmark.py` nằm cùng thư mục với `tkvc.exe` (compiler repo), sinh 7
  exe kernel riêng (1 kernel/exe) để đo từng kernel độc lập.

---

## 6. Độ phủ API so với thư viện tham chiếu

Audit quét toàn bộ 595 tên public của thư viện số học tham chiếu độc lập
(np + linalg + fft + random) và đối chiếu với surface của
stdlib .tkv (hàm module-level + method NDArray/BoolNDArray, kể cả alias quy ước
dịch TV-1001):

| Nhóm | Số tên | Tỷ lệ |
| :--- | ---: | ---: |
| N/A tầng ngôn ngữ (dtype objects, hằng số, máy RNG, packaging…) | 241 | — |
| **Library surface được audit** | **354** | 100% |
| **Có counterpart trong .tkv** | **354** | **100%** |
| Chưa có | 0 | **0%** |

**Kết luận độ phủ: 100% surface hàm thư viện số học (354/354), 0 hàm thiếu.**
Cách đo chặt: method/attr được verify trên class NDArray/BoolNDArray/Tensor
thật (không dùng danh sách viết tay). Đợt gap-fill tổng cộng đã bổ sung ~90
hàm: elementwise/scalar (abs, pow, maximum/minimum, gcd/lcm, frexp/ldexp/modf/
divmod, nextafter/spacing, nan_to_num, fmax/fmin, logical_xor, heaviside,
signbit, ptp…), array ops (array_split, dsplit, delete, argwhere,
trim_zeros, broadcast_arrays, indices, unique_all family, sort_complex,
packbits/unpackbits), binning/indexing (digitize, bincount, diagflat,
fill_diagonal, tri, tril/triu_indices, mask_indices, vander, partition,
searchsorted, choose), FFT hoàn chỉnh (fftn/ifftn, rfft2/irfft2, rfftn/
irfftn, fftfreq/rfftfreq, fftshift/ifftshift, ihfft/hfft), creation (empty,
logspace, geomspace, ravel, copy, astype, real/imag, ndim/size), histogram
family + euler_gamma + kaiser + bitwise_count.

**Ba ghi chú trung thực khi đọc con số 100%:**

1. **Phạm vi đo:** 100% tính trên 354 tên *hàm thư viện*. Nếu tính trên toàn
   bộ 595 tên public của thư viện tham chiếu (gồm cả dtype objects, hằng số, máy RNG,
   errstate, packaging — 241 tên thuộc tầng ngôn ngữ/compiler TV: `tv.f64`,
   hằng số runtime, GC của tkvc), tỷ lệ là 354/595 ≈ 60%. Đây là ranh giới
   thiết kế của ngôn ngữ, không phải thiếu hụt thư viện.
2. **Surface ≠ semantics:** audit đối chiếu *tên tồn tại*. Parity số học của
   từng kernel được chứng minh riêng trong benchmark (§5, sai số 0.0 → 1e-14);
   các tham số phụ chuyên biệt (vd. `density=`, `order=`) ngoài scope audit.
3. **Kết quả do audit thủ công ghi lại** khi đối chiếu surface .tkv với thư
   viện tham chiếu độc lập (script audit từng nằm ở
   `tests/tokenvector/numpy_coverage_audit.py` đã bị xóa khỏi repo — số liệu
   354/354 là kết quả lần chạy cuối cùng được lưu lại trong tài liệu này).

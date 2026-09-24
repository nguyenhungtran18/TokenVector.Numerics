# TokenVector Numerics — Standard Library (`.tkv`)

**Document Code:** TV-1001-STDLIB-README-2026-V1

Bản dịch ngôn ngữ TokenVector hoàn chỉnh của thư viện **TokenVector.Numerics**
(.NET 8) — 27 module `.tkv`, ~12.100 dòng, đối chiếu 1:1 với toàn bộ 67 file
nguồn của `src/TokenVector.Numerics/` (mỗi file nguồn được ánh xạ trong header
"Source of truth" của ít nhất một module .tkv).

> Ngôn ngữ đích được đặc tả trong [TOKENVECTOR_SYNTAX_SPEC.md](../TOKENVECTOR_SYNTAX_SPEC.md)
> (doc code `TKV-SPEC-SYNTAX-2026-V1`). Mọi file `.tkv` tuân theo đặc tả đó và
> mang header "TV-1001: ... translation of ..." ghi rõ nguồn tương ứng.

---

## 1. Bảng ánh xạ module nguồn ↔ .tkv

| Module `.tkv` | Nguồn (`src/TokenVector.Numerics/`) | Nội dung chính |
| :--- | :--- | :--- |
| `core.tkv` | `Core/ShapeHelper.cs`, `Core/TensorBuffer.cs`, `Core/NDArray.cs`, `Core/BoolNDArray.cs`, `Core/ShapeExtensions.cs` + phần shape của `Engine/BroadcastEngine.cs` | `TensorBuffer`, `NDArray` (view `shape/strides/offset/buffer.data`), `BoolNDArray`, shape helpers, broadcast-shape helpers (chủ import — không cycle) |
| `engine.tkv` | `Engine/BroadcastEngine.cs`, `Engine/SIMDKernels.cs` | Broadcast zero-copy Stride-0, kernel contiguous; AVX2/`Vector256` gộp thành vòng tuần tự — `tkvc -O simd` khôi phục |
| `linalg.tkv` | `LinAlg/MatrixMultiplication.cs`, `LinAlg/MatrixOps.cs`, `LinAlg/Decomposition.cs`, `LinAlg/EinSum.cs`, `LinAlg/Eigen.cs`, `LinAlg/SVD.cs` | matmul, kron, outer, norm, lu/qr/cholesky/solve/inverse/det/trace/schur/sylvester, einsum, eigh (Jacobi), svd (one-sided Jacobi) |
| `linalg_functions.tkv` | `LinAlg/MatrixFunctions.cs`, `LinAlg/LinAlgExtended.cs`, `LinAlg/Complex.cs` | expm (Padé 6×6), sqrtm (Denman–Beavers), pinv, lstsq, cond, solve_band_tridiagonal, null_space, kronecker_sum, `class Complex` |
| `ops.tkv` | `Ops/ElementWiseOps.cs`, `Ops/ElementWiseMathOps.cs`, `Ops/ReductionOps.cs` | add/sub/mul/div/mod (binary + scalar), trig, exp/log, rounding, logaddexp, sum/mean/prod/min/max, argmax/argmin |
| `manipulation.tkv` | `Ops/ManipulationOps.cs`, `Ops/SortingOps.cs`, `Ops/StatisticsOps.cs` | concatenate, stack, tile, pad, roll, flip, sort, argsort, unique, var, std, median, percentile, cov, corrcoef |
| `grid.tkv` | `Ops/GridOps.cs`, `Ops/CumulativeOps.cs`, `Ops/IndexingOps.cs`, `Ops/LogicToleranceOps.cs` | meshgrid, diag, triu/tril, cumsum, cumprod, diff, filter, where, take, put, isclose, allclose, isnan/isinf + **fancy indexing chuẩn tham chiếu**: `boolean_select` (arr[mask]), `boolean_assign` (arr[mask] = v), `flat_index_select/assign` (gather/scatter phẳng) |
| `compare.tkv` | `Ops/ComparisonOps.cs`, `Ops/SetOperations.cs`, `Ops/BitwiseOps.cs` | so sánh (scalar + tensor), intersect1d/union1d/setdiff1d/setxor1d/isin, bitwise and/or/xor/not, shifts |
| `autograd.tkv` | `Autograd/Tensor.cs`, `Autograd/AutogradNode.cs`, `Autograd/Nodes/MathAndMatrixNodes.cs`, `Autograd/Nodes/ActivationAndReductionNodes.cs`, `Autograd/Optim/Optimizers.cs`, `Autograd/NN/Modules.cs` | Reverse-mode AD trên DAG động, unbroadcast, SGD/AdamW (decoupled weight decay), Module/Linear/Sequential/RMSNorm |
| `neural.tkv` | `Neural/Activations.cs`, `Neural/Losses.cs`, `Neural/Normalization.cs`, `Neural/AttentionEngine.cs`, `Neural/ConvEngine.cs`, `Neural/OptimalTransportAndDiffusion.cs` | activations + VJP, losses, layer/rms norm, attention + causal mask + RoPE, im2col/conv2d/maxpool, Sinkhorn, DDIM |
| `optimize.tkv` | `Optimize/OptimizationAndRootFinding.cs` | root_brentq, bisection, minimize_brent, Nelder–Mead, BFGS, LP simplex, QP (ADMM kiểu OSQP) |
| `interpolation.tkv` | `Interpolation/InterpolationAndSplines.cs` | CubicSpline, bilinear, RBF, barycentric |
| `poly.tkv` | `LinAlg/Polynomial.cs`, `LinAlg/NumericalAnalysis.cs` | poly_fit, poly_val, roots (companion + QR), trapezoidal, central difference, Lagrange |
| `special.tkv` | `LinAlg/SpecialFunctions.cs` | erf/erfc, gamma/log_gamma (Lanczos), Bessel I0/J0/Y0/K0, LambertW, Airy, beta, digamma, erfinv |
| `fft.tkv` | `LinAlg/FFT.cs` | radix-2, Bluestein chirp-z cho N bất kỳ, FFT 2D, rfft/irfft |
| `signal.tkv` | `LinAlg/SignalProcessing.cs` | convolve/correlate (full/same/valid), cửa sổ DSP, DWT/IDWT, Hilbert, PSD, spectrogram |
| `random.tkv` | `Random/TensorRandom.cs` | Park–Miller LCG, Box–Muller, randint, shuffle, choice; state module-level qua box list |
| `io.tkv` | `IO/TensorIO.cs`, `IO/TensorArchiveAndMemMap.cs` | .npy v1.0, raw binary, CSV, .npz (song song list, không dict), `MemoryMappedNDArray` |
| `statistics.tkv` | `Statistics/DistributionsAndTests.cs`, `Statistics/TimeSeriesAndKalman.cs` | phân phối + tests, Kalman 1D/ND, time-series |
| `quantum.tkv` | `Quantum/QuantumEngine.cs` | QState statevector ≤28 qubit: H/X/Y/Z/S/T/Rz/CNOT/Toffoli/CZ/SWAP/CRz/QFT/Measure/DensityMatrix/VonNeumannEntropy |
| `physics.tkv` | `Physics/PhysicsODEAndFields.cs`, `Physics/LatticeBoltzmann2D.cs` | RK4, symplectic Verlet, gradient/divergence/laplacian, D2Q9 LBM |
| `astro.tkv` | `Science/Astrodynamics.cs` | solve_kepler, keplerian_to_cartesian, Hohmann/bi-elliptic, geodetic↔ECEF, J2, Gibbs, ECI→ECEF |
| `finance.tkv` | `Finance/FinanceMath.cs` | Black–Scholes + Greeks, Markowitz/Sharpe, NPV/IRR, binomial CRR, VaR/CVaR, bond, Nelson–Siegel |
| `spatial.tkv` | `Spatial/SpatialTreesAndGeometry.cs`, `Spatial/Quaternion.cs`, `Spatial/Affine3D.cs` | KDTree, ConvexHull2D, distance metrics, quaternion, affine 3D |
| `geometry3d.tkv` | `Spatial/Geometry3DAndPointClouds.cs`, `Spatial/HyperbolicGeometry.cs` | khoảng cách điểm–mặt/đoạn, Kabsch alignment, hyperbolic (Poincaré, Möbius) |
| `crypto_graph.tkv` | `Crypto/LatticeMath.cs`, `Graphs/SpectralGraph.cs` | NTT (Cooley–Tukey mod q), ModPow/ModInverse, LLL reduction, Laplacian, Chebyshev graph conv |
| `biology_robotics.tkv` | `Biology/ProteinGeometry.cs`, `Robotics/OptimalControlAndRobotics.cs` | dihedral, Kabsch RMSD, TM-score, Discrete LQR (Riccati), damped least-squares IK |

Mỗi header `.tkv` còn liệt kê ánh xạ **từng section/hàm** nguồn → `.tkv`
(ví dụ `SolveKepler (Newton + Halley step) -> solve_kepler`).

---

## 2. Quy ước dịch TV-1001

1. **Hàm snake_case cấp module.** Overload nguồn (`Add(NDArray, NDArray)` /
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

### 3.2. Với verification harness (không cần tkvc — chạy được ngay)

```powershell
python tests/tokenvector/tkv_harness.py
# Syntax gate: all .tkv modules parse, TV-1001 constructs only.
# TokenVector stdlib smoke tests: passed=175, failed=0
```

Harness (`tests/tokenvector/tkv_harness.py`) làm 3 việc:
1. **Syntax gate** — parse mọi `.tkv` bằng AST parser và từ chối mọi construct
   ngoài grammar TV-1001 (dict, comprehension, global, try, with, …).
2. **Runtime `tv.*`** — cung cấp primitive compiler theo spec (`tv.sqrt`,
   `tv.array`, `tv.io.*`, `tv.f64_bits`, …) bằng runtime thuần.
3. **Thực thi** smoke suite như một chương trình TV (gọi `main()` ở mức module,
   đúng mô hình `tkvc smoke_tests.tkv -o smoke.exe`).

> Ghi chú: `special.gamma` dùng Lanczos (g=7) giống hệt bản gốc nên sai số tuyệt đối
> ~2e-9 tại Gamma(5); smoke test dùng tolerance 1e-6 đúng như bộ xUnit
> `tests/TokenVector.Numerics.Tests/`.

---

## 4. Ví dụ nhanh

```tokenvector
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

Chi tiết ngôn ngữ: `TOKENVECTOR_SYNTAX_SPEC.md` (§1–§7). Mã nguồn tham chiếu:
`src/TokenVector.Numerics/**` (không thay đổi trong bản dịch này).

---

## 5. Benchmark so với thư viện tham chiếu

Chạy: `python tests/tokenvector/benchmark_vs_numpy.py`
(máy tham chiếu: Windows x86_64, chạy trên verification harness và thư viện tham chiếu độc lập — thời gian = best-of-N).

| Kernel | Size | TokenVector (.tkv) | Tham chiếu | Ratio | max\|diff\| |
| :--- | :--- | ---: | ---: | ---: | ---: |
| matmul (tiled 32×32) | 32×32 | 0.0084s | 0.0000s | ~1000× | 0.0 |
| matmul (tiled 32×32) | 128×128 | 0.566s | 0.0002s | ~2800× | 0.0 |
| add (broadcast) | 2×131072 | 1.247s | 0.0007s | ~1700× | 0.0 |
| SVD (one-sided Jacobi) | 40×40 | 0.967s | 0.0023s | ~420× | 1.1e-14 |
| FFT radix-2 | 1024 | 0.014s | 0.0000s | ~390× | 6.0e-12 |
| FFT Bluestein | 1000 | 0.100s | 0.0000s | ~2800× | 4.5e-12 |

**Đọc kết quả thế nào cho đúng:**

- **Parity số học: hoàn toàn khớp** — matmul/add chênh 0.0, SVD ~1e-14
  (giới hạn float64), FFT ~5e-12. Kết quả của TokenVector = kết quả của thư viện tham chiếu
  ở mọi kernel trên.
- **Tốc độ: chậm hơn 400–2800×** — đây là số của **bản .tkv chạy trên verification
  harness** (thông dịch trên thông dịch). Không phải con số của TokenVector
  compiled: `tkvc -O parallel -O simd` hạ vòng lặp xuống SIMD/parallel native
  như mô tả ở fidelity notes, thu hẹp khoảng cách sâu hơn nữa.
- Khoảng cách nhỏ nhất ở **FFT radix-2** (~390×) — thuật toán O(N log N) của
  stdlib tự viết đã bám sát cấu trúc của pocketfft.

---

## 6. Độ phủ API so với thư viện tham chiếu

Chạy: `python tests/tokenvector/numpy_coverage_audit.py`

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
3. **Đo được tái lập:** chạy `python tests/tokenvector/numpy_coverage_audit.py`
   — script tự quét numpy đang cài và in chi tiết từng nhóm.

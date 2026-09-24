# 🚀 TokenVector.Numerics v1.1.0 Thông Báo Phát Hành

[🇬🇧 View English Version](RELEASE_NOTES.md)

---

**Phiên bản phát hành:** `v1.1.0`  
**Ngày phát hành:** 24/09/2026  
**Nền tảng mục tiêu:** .NET 8.0 LTS + thư viện chuẩn TokenVector (`.tkv`, đặc tả TV-1001)  
**Giấy phép (License):** [MIT License](LICENSE)  
**Kho lưu trữ (Repository):** [https://github.com/nguyenhungtran18/TokenVector.Numerics](https://github.com/nguyenhungtran18/TokenVector.Numerics)  
**Gói NuGet:** `TokenVector.Numerics` (v1.1.0)

---

## 🌟 Tổng Quan & Điểm Mới trong v1.1.0

Bản phát hành **v1.1.0** đóng gói **bản dịch ngôn ngữ TokenVector hoàn chỉnh** của toàn bộ thư viện (27 module `.tkv`, ~12.1k dòng, đặc tả `TKV-SPEC-SYNTAX-2026-V1`), đạt **độ phủ 100% surface hàm thư viện số học đã audit** (354/354 tên, 0 thiếu) với parity số học kiểm chứng độc lập, cùng đường nóng matmul cache-tiled, fancy indexing chuẩn tham chiếu, và bộ công cụ kiểm chứng tái lập được (syntax gate + runtime harness + coverage audit + benchmark).

---

## 🚀 Các Tính Năng & Bổ Sung Chi Tiết trong v1.1.0

### 1. Bản dịch ngôn ngữ TokenVector đầy đủ (`src/tokenvector/`, TV-1001)
* **27 module `.tkv`** (~12.1k dòng) — xây theo quy ước TV-1001 trên toàn bộ 67 file nguồn, mỗi ánh xạ được ghi trong header "Source of truth" của từng file.
* **Quy ước TV-1001:** hàm snake_case cấp module, giữ nguyên tên class (`NDArray`, `Tensor`, `QState`, `KDTree`, …), các construct dict/`isinstance`/nested-def thay bằng song song list và structural check, Parallel.For/AVX2 gộp về vòng tuần tự (`tkvc -O parallel -O simd` khôi phục ở mức CIL).
* **Đồ thị import không có cycle:** broadcast-shape helpers đặt trong `tv.core`, `tv.engine` re-export.

### 2. Độ phủ 100% surface hàm thư viện số học
* **354/354 hàm thư viện số học đã audit đều có counterpart `.tkv`** — đo bằng `tests/tokenvector/numpy_coverage_audit.py`: script quét toàn bộ surface public của một thư viện tham chiếu độc lập (595 tên) và verify method trên class `NDArray`/`BoolNDArray`/`Tensor` thật.
* **~80 hàm mới** trên các nhóm: elementwise/scalar (`abs`, `pow`, `maximum/minimum`, `gcd/lcm`, `frexp/ldexp/modf/divmod`, `nextafter/spacing`, `nan_to_num`, `fmax/fmin`, `heaviside`, `signbit`, `ptp`), array ops (`array_split`, `delete`, `argwhere`, `trim_zeros`, `broadcast_arrays`, `indices`, họ `unique_all`, `sort_complex`, `packbits/unpackbits`), binning/indexing (`digitize`, `bincount`, `diagflat`, `fill_diagonal`, `tri`, `tril/triu_indices`, `mask_indices`, `vander`, `partition`, `searchsorted`, `choose`), họ FFT hoàn chỉnh (`fftn/ifftn`, `rfft2/irfft2`, `rfftn/irfftn`, `fftfreq/rfftfreq`, `fftshift/ifftshift`, `ihfft/hfft`), creation (`empty`, `logspace`, `geomspace`, `ravel`, `copy`, `astype`, `real/imag`, `ndim/size`), `histogram`/`histogram2d`/`histogramdd`, `euler_gamma`, `kaiser`, `bitwise_count`.
* 241 tên public còn lại được audit (dtype objects, hằng số, máy RNG, errstate/printing, packaging) thuộc tầng ngôn ngữ/compiler trong TokenVector (`tv.f64`, hằng số runtime, `tkvc`) theo thiết kế.

### 3. Hiệu năng & Parity số học so với thư viện tham chiếu
* **Matmul cache-tiled** (khối 32×32) viết lại bằng flat-index arithmetic thuần và vòng trong unit-stride (`matmul_2d`, `batch_matmul`) — giữ nguyên API, kết quả không đổi.
* **Bộ benchmark** (`tests/tokenvector/benchmark_vs_numpy.py`): parity số học mức **0.0** (matmul, broadcast add), **1.1e-14** (SVD one-sided Jacobi), **~5e-12** (FFT radix-2 & Bluestein) so với thư viện tham chiếu độc lập. Tỷ lệ tốc độ của bản thông dịch được báo kèm ghi chú về bản `tkvc` đã compile.

### 4. Fancy Indexing chuẩn tham chiếu (`grid.tkv`)
* `boolean_select` / `boolean_assign` — `arr[mask]` và `arr[mask] = values` với mask boolean broadcast right-aligned (giá trị scalar hoặc mảng đúng độ dài, semantics lỗi chuẩn).
* `flat_index_select` / `flat_index_assign` — `np.take`/`np.put` với index âm và bounds-check `mode='raise'`.

### 5. Bộ công cụ kiểm chứng (`tests/tokenvector/`)
* **`tkv_harness.py`** — syntax gate bằng AST từ chối mọi construct ngoài grammar TV-1001, cộng runtime `tv.*` thuần Python (array factory, primitive `tv.io` file/zip/mmap, bit-reinterpretation IEEE-754) để thực thi chương trình `.tkv`.
* **`numpy_coverage_audit.py`** — audit độ phủ tái lập được, in chi tiết từng nhóm.
* **`benchmark_vs_numpy.py`** — micro-benchmark parity + hiệu năng.

---

## 📊 Số Liệu Kiểm Chứng (v1.1.0)

| Bộ kiểm | Kết quả |
| :--- | :--- |
| Runtime xUnit (`TokenVector.Numerics.Tests`) | **84 / 84 pass (100%)** |
| Smoke suite `.tkv` (27 module) | **175 / 175 pass (100%)** |
| Độ phủ surface thư viện số học | **354 / 354 = 100%, 0 thiếu** |
| Parity số học với thư viện tham chiếu (kernel benchmark) | max\|diff\| 0.0 → 1.1e-14 |
| Syntax gate `.tkv` (grammar TV-1001) | 27/27 module sạch |

---

# 🚀 TokenVector.Numerics v1.0.1 Thông Báo Phát Hành (Bản Trước)

**Phiên bản phát hành:** `v1.0.1`  
**Ngày phát hành:** 13/09/2026  
**Nền tảng mục tiêu:** .NET 8.0 LTS  
**Giấy phép (License):** [MIT License](LICENSE)  
**Kho lưu trữ (Repository):** [https://github.com/nguyenhungtran18/TokenVector.Numerics](https://github.com/nguyenhungtran18/TokenVector.Numerics)  
**Gói NuGet:** `TokenVector.Numerics` (v1.0.1)

---

## 🌟 Tổng Quan & Điểm Mới trong v1.0.1

Bản cập nhật **v1.0.1** bổ sung các hàm toán học cao cấp, giải thuật phân rã ma trận đại số tuyến tính, biến đổi sóng Wavelet, giải quy hoạch toàn phương (Quadratic Programming) cho điều khiển drone/robotics và tích phân Symplectic bảo toàn năng lượng cho cơ học lượng tử / thiên văn học.

---

## 🚀 Các Tính Năng & Bổ Sung Chi Tiết trong v1.0.1

### 1. Hàm Toán Học Đặc Biệt Cao Cấp (`SpecialFunctions`)
* **Hàm Lambert W (`LambertW`):** Tính nghiệm phương trình siêu việt $W_k(x) e^{W_k(x)} = x$ sử dụng phương pháp lặp bậc 3 Halley. Hỗ trợ nhánh chính $k=0$ (cho $x \ge -1/e$) và nhánh phụ $k=-1$ (cho $-1/e \le x < 0$) trên cả số thực và Tensor.
* **Hàm Bessel Loại 2 (`BesselY0`, `BesselK0`):**
  * $Y_0(x)$ (Hàm Neumann bậc 0).
  * $K_0(x)$ (Hàm Bessel biến đổi loại 2 bậc 0).
* **Hàm Airy (`AiryAi`, `AiryBi`):** Giải phương trình vi phân $y'' - x y = 0$ sử dụng chuỗi Maclaurin quanh 0 và khai triển tiệm cận cho $|x|$ lớn.

### 2. Phân Rã Ma Trận & Phương Trình Ma Trận Tuyến Tính (`LinAlg`)
* **Phân Rã Schur Thực (`Decomposition.Schur`):** Phân rã ma trận vuông thực $A = Q T Q^T$ (với $Q$ trực giao và $T$ dạng tam giác tựa - quasi-upper triangular) bằng thuật toán biến đổi Hessenberg và phép lặp QR có dịch chuyển (Shifted QR).
* **Giải Phương Trình Sylvester (`Decomposition.SolveSylvester`):** Giải chính xác phương trình ma trận liên tục $A X + X B = C$ (và phương trình Lyapunov khi $B = A^T$) thông qua phép vector hóa tích Kronecker.

### 3. Biến Đổi Sóng Wavelet & Xử Lý Tín Hiệu Giải Tích (`SignalProcessing`)
* **Biến Đổi Wavelet Rời Rạc (`DWT` & `IDWT`):** Biến đổi Wavelet 1D thuận/nghịch hỗ trợ ngân hàng bộ lọc Haar và Daubechies-4 (`db4`), đảm bảo tái tạo tín hiệu nguyên bản chính xác 100%.
* **Biến Đổi Hilbert & Tín Hiệu Giải Tích (`Hilbert`, `AnalyticSignal`):** Tạo tín hiệu giải tích $x_a(t) = x(t) + i \mathcal{H}[x(t)]$ với độ phức tạp $O(N \log N)$ thông qua FFT.

### 4. Bộ Giải Quy Hoạch Toàn Phương (`Optimize.QPSolve`)
* **Trình Giải QP Lồi (Convex QP Solver):** Giải bài toán tối ưu hóa toàn phương có ràng buộc:
  $$\min_x \frac{1}{2} x^T P x + q^T x \quad \text{thỏa mãn} \quad G x \le h, \quad A x = b, \quad lb \le x \le ub$$
* **Động Cơ Toán Tử Tách ADMM (Operator-Splitting ADMM):** Tối ưu hóa phân tích nhân tử ma trận và phép chiếu Proximal, đặc thù cho điều khiển dự báo mô hình phi tuyến (NMPC), robot bầy đàn và thiết bị bay không người lái (drone).

### 5. Tích Phân Symplectic Bảo Toàn Năng Lượng (`PhysicsODEAndFields.SolveSymplecticVerlet`)
* **Thuật Toán Tích Phân Leapfrog / Velocity-Verlet:** Tích phân hệ động lực học Hamiltonian $H(q, p) = \frac{1}{2m} p^T p + V(q)$, bảo toàn thể tích không gian pha và tổng năng lượng mà không bị trôi/tiêu hao năng lượng theo thời gian.

---

## 📊 Kết Quả Kiểm Thử Nghiệm Thu (Test Suite)

* **Tổng số bài test:** **84 / 84 Tests Passed (100% Pass)**
* **Lỗi (Failures):** **0**
* **Thời gian thực thi:** **~138 ms**
* **Môi trường kiểm thử:** `TokenVector.Numerics.Tests` (xUnit, Release x64)

---

## 📦 Gói Phát Hành (`dist/`)

| Tệp phát hành | Đường dẫn | Mô tả |
| :--- | :--- | :--- |
| **Release Zip** | `dist/TokenVector.Numerics-v1.0.1-Release.zip` | Gói phát hành độc lập v1.0.1 hoàn chỉnh |
| **Gói NuGet** | `dist/nuget/TokenVector.Numerics.1.0.1.nupkg` | Gói NuGet chính thức v1.0.1 |
| **Thư viện DLL** | `dist/bin/TokenVector.Numerics.dll` | Thư viện liên kết động .NET 8 standalone |

---

## 🤝 Đóng Góp & Liên Hệ

Chúng tôi trân trọng mọi đóng góp, phản hồi và báo lỗi từ cộng đồng lập trình viên và nghiên cứu khoa học trên [GitHub Repository](https://github.com/nguyenhungtran18/TokenVector.Numerics).

# 🚀 TokenVector.Numerics v1.1.1 Thông Báo Phát Hành

[🇬🇧 View English Version](RELEASE_NOTES.md)

---

**Phiên bản phát hành:** `v1.1.1`  
**Ngày phát hành:** 25/09/2026  
**Nền tảng mục tiêu:** .NET 8.0 LTS + thư viện chuẩn TokenVector (`.tkv`, đặc tả TV-1001)  
**Giấy phép (License):** [MIT License](LICENSE)  
**Kho lưu trữ (Repository):** [https://github.com/nguyenhungtran18/TokenVector.Numerics](https://github.com/nguyenhungtran18/TokenVector.Numerics)  
**Gói NuGet:** `TokenVector.Numerics` (v1.1.1)

---

## 🌟 Tổng Quan & Điểm Mới trong v1.1.1

Đây là bản phát hành **tài liệu và độ chính xác**. Bản này bổ sung module toán chính xác mới `mathlib/`, ghi nhận lại toàn bộ công việc kiểm thử các module toán, và phát hành lại gói NuGet với README đã cập nhật.

> **Lưu ý về binary — đọc trước khi nâng cấp.** File `TokenVector.Numerics.dll` trong bản này **giống hệt byte-for-byte với v1.1.0**. Phần source engine C# đã bị loại khỏi kho lưu trữ ngay trong bản v1.1.0 (`feat!: remove runtime engine sources`), nên assembly không thể biên dịch lại từ mã nguồn nữa. **v1.1.1 chỉ là bản tăng phiên bản tài liệu** — phần thêm `mathlib/` là mã nguồn `.tkv`, không thuộc assembly. Nếu cần binary được biên dịch lại, phải khôi phục source engine trước.

---

## 🔬 Mới: `mathlib/` toán chính xác (`.tkv`)

Hai module thuần TokenVector, không phụ thuộc `tv`, nên biên dịch và chạy độc lập bằng `tkvc`:

* **`mathlib/bf_bigfloat.tkv`** — số học bignum cơ số $10^4$, căn bậc hai dạng chia dài từng chữ số, $\pi$ bằng Chudnovsky binary splitting (công thức Gourdon), $e$ bằng spigot chuỗi, và phép chia bignum đầy đủ.
* **`mathlib/nt_number_theory.tkv`** — gcd/lcm, căn bậc hai nguyên, phân tích thử nghiệm, `iroot`, `pow_mod` / Miller–Rabin / Pollard–Rho **không tràn**, và kiểm tra nguyên AKS.

## 🐛 Sửa lỗi độ chính xác (5 lỗi)

| Module | Lỗi | Ảnh hưởng | Cách sửa |
| :--- | :--- | :--- | :--- |
| `bf_bigfloat.tkv` | hằng Chudnovsky `10939058825628000` | $\pi$ sai từ khoảng chữ số 14 | sửa thành `10939058860032000` |
| `bf_bigfloat.tkv` | `bignum_neg` làm mất một limb | sai kết quả với bignum âm | bỏ nhánh phủ bù |
| `bf_bigfloat.tkv` | `pi_chudnovsky` thiếu độ chính xác, xử lý sai zero dẫn | sai các chữ số thấp khi độ chính xác cao | thêm chữ số bảo vệ, bỏ zero dẫn rồi chuẩn hoá |
| `nt_number_theory.tkv` | phép nhân/cộng modulo âm thầm tràn `i64` | sai tính nguyên và phân tích gần $2^{63}$ | thêm `mul_mod_i64` / `add_mod_i64` |
| `nt_number_theory.tkv` | `isqrt_i`, `trial_prime`, `factorize`, `iroot` tràn ở biên | hỏng ở đầu dải `i64` | kiểm tra miền mọi bước nhân trung gian |

## 🧪 Kết quả kiểm chứng

| Suite | Kết quả | Loại |
| :--- | :---: | :--- |
| `mathlib/bf_bigfloat.tkv` | **8 / 8** | chạy native (`tkvc build` + thực thi) |
| `mathlib/nt_number_theory.tkv` | **9 / 9** | chạy native (`tkvc build` + thực thi) |
| `linalg` / `linalg_functions` / `fft` / `crypto_graph` | **36 / 36** | rà soát mức mã nguồn với oracle độc lập |

Đã thêm bốn test hồi quy: `t7_bignum_neg`, `t8_precision_borders`, `t8_i64_boundaries`, `t9_large_factorization`.

## 📖 Đính chính tài liệu

Lệnh smoke 175 kiểm tra `.tkv` được trích khắp tài liệu **không tái lập được** trên các bản `tkvc` hiện tại — compiler phân giải `import tv` theo thư mục đi kèm của chính nó và không có tuỳ chọn runtime-path, nên build dừng với lỗi `File khong co ham top-level nao co annotation kieu DSL`. README, báo cáo kiểm thử, `llms.txt` và thông báo này nay nói rõ **175/175 là kết quả lưu của v1.1.0** và chỉ định hai suite `mathlib` là bài kiểm tra tái lập được. Bảng benchmark (kernel `tkvc` đã biên dịch so với NumPy 2.5.2) giữ nguyên và vẫn được quy cho lần đo của v1.1.0.

## ⚙️ CI

Job `verify-stdlib` trước đây chạy `tkvc build tests/tokenvector/smoke_tests.tkv` — lệnh dừng vì lỗi phân giải `import tv`, tức là một cổng kiểm chứng không bao giờ có thể xanh. Nó được thay bằng **`verify-mathlib`**, build và chạy hai suite thực sự tái lập được (`PASS 8/8`, `PASS 9/9`). Khi máy chủ không có `tkvc`, job bỏ qua kèm thông báo rõ ràng thay vì báo đạt giả. Smoke suite vẫn nằm ngoài CI cho tới khi compiler resolve được stdlib từ thư mục dự án.

## 📦 Đóng gói

| Artifact | Nội dung |
| :--- | :--- |
| `packages/TokenVector.Numerics.1.1.1.nupkg` | Cùng `lib/net8.0/TokenVector.Numerics.dll` và tài liệu XML với v1.1.0, kèm README của bản phát hành này |
| `packages/TokenVector.Numerics.1.1.1.snupkg` | Symbols (cùng PDB với v1.1.0) |
| `dist/TokenVector.Numerics-v1.1.1-Release.zip` | Bundle đầy đủ: binary, tài liệu hiện hành, 27 module stdlib, hai module `mathlib`, smoke suite và các gói 1.1.1 |

Bundle được lắp ghép chỉ từ file đã được git track, nên các file đã xoá như Python harness và `TokenVector.Numerics.deps.json` không thể lọt vào bản phát hành. Hai zip 1.0.1 và 1.1.0 vẫn còn trong repository như hồ sơ phát hành lịch sử; chúng có trước khi harness bị gỡ nên vẫn chứa các file đó.

---

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
* **27 module `.tkv`** (~12.1k dòng) — thư viện chuẩn TokenVector hoàn chỉnh, mỗi ánh xạ được ghi trong header "Source of truth" của từng file.
* **Quy ước TV-1001:** hàm snake_case cấp module, giữ nguyên tên class (`NDArray`, `Tensor`, `QState`, `KDTree`, …), các construct dict/`isinstance`/nested-def thay bằng song song list và structural check, Parallel.For/AVX2 gộp về vòng tuần tự (`tkvc -O parallel -O simd` khôi phục ở mức CIL).
* **Đồ thị import không có cycle:** broadcast-shape helpers đặt trong `tv.core`, `tv.engine` re-export.

### 2. Độ phủ 100% surface hàm thư viện số học
* **354/354 hàm thư viện số học đã audit đều có counterpart `.tkv`** — audit với toàn bộ surface public của một thư viện tham chiếu độc lập (595 tên), verify method trên class `NDArray`/`BoolNDArray`/`Tensor` thật.
* **~80 hàm mới** trên các nhóm: elementwise/scalar (`abs`, `pow`, `maximum/minimum`, `gcd/lcm`, `frexp/ldexp/modf/divmod`, `nextafter/spacing`, `nan_to_num`, `fmax/fmin`, `heaviside`, `signbit`, `ptp`), array ops (`array_split`, `delete`, `argwhere`, `trim_zeros`, `broadcast_arrays`, `indices`, họ `unique_all`, `sort_complex`, `packbits/unpackbits`), binning/indexing (`digitize`, `bincount`, `diagflat`, `fill_diagonal`, `tri`, `tril/triu_indices`, `mask_indices`, `vander`, `partition`, `searchsorted`, `choose`), họ FFT hoàn chỉnh (`fftn/ifftn`, `rfft2/irfft2`, `rfftn/irfftn`, `fftfreq/rfftfreq`, `fftshift/ifftshift`, `ihfft/hfft`), creation (`empty`, `logspace`, `geomspace`, `ravel`, `copy`, `astype`, `real/imag`, `ndim/size`), `histogram`/`histogram2d`/`histogramdd`, `euler_gamma`, `kaiser`, `bitwise_count`.
* 241 tên public còn lại được audit (dtype objects, hằng số, máy RNG, errstate/printing, packaging) thuộc tầng ngôn ngữ/compiler trong TokenVector (`tv.f64`, hằng số runtime, `tkvc`) theo thiết kế.

### 3. Hiệu năng & Parity số học (compiled qua `tkvc.exe`)
* **Matmul cache-tiled** (khối 32×32) viết lại bằng flat-index arithmetic thuần và vòng trong unit-stride (`matmul_2d`, `batch_matmul`) — giữ nguyên API, kết quả không đổi.
* **Benchmark bản compiled vs NumPy 2.5.2** (kernel thuần TokenVector biên dịch thành exe độc lập bằng `tkvc.exe`, cùng máy, best-of-3, đã trừ startup): parity số học xác nhận từng kernel — checksum matmul/add trong giới hạn float64, singular values SVD lệch **2.4e-14** so với LAPACK, FFT **~1e-12**. Ratio so với NumPy: add ~7×, FFT radix-2 ~17×, matmul 365–1308×, SVD ~1880×. Số đo cũ qua harness Python (thông dịch trên thông dịch, 400–3000×) đã lỗi thời — harness đã bị xóa khỏi repo.

### 4. Fancy Indexing chuẩn tham chiếu (`grid.tkv`)
* `boolean_select` / `boolean_assign` — `arr[mask]` và `arr[mask] = values` với mask boolean broadcast right-aligned (giá trị scalar hoặc mảng đúng độ dài, semantics lỗi chuẩn).
* `flat_index_select` / `flat_index_assign` — `np.take`/`np.put` với index âm và bounds-check `mode='raise'`.

### 5. Bộ công cụ kiểm chứng (`tests/tokenvector/`)
* **`smoke_tests.tkv`** — 175 check trên cả 27 module, compile và chạy native: `tkvc.exe build smoke_tests.tkv --entry main --out smoke.exe`.
* Kết quả audit độ phủ: **354/354** tên hàm thư viện số học khớp (audit với thư viện tham chiếu độc lập; script audit đã bị xóa cùng bộ harness Python).

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

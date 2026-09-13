# 🚀 TokenVector.Numerics v1.0.1 Thông Báo Phát Hành

[🇬🇧 View English Version](RELEASE_NOTES.md)

---

**Phiên bản phát hành:** `v1.0.1`  
**Ngày phát hành:** 13/09/2026  
**Nền tảng mục tiêu:** .NET 8.0 LTS (C# 12)  
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

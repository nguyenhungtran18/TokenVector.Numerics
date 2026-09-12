# 🚀 TokenVector.Numerics v1.0.0 Thông Báo Phát Hành

[🇬🇧 View English Version](RELEASE_NOTES.md)

---

**Phiên bản phát hành:** `v1.0.0`  
**Ngày phát hành:** 13/09/2026  
**Nền tảng mục tiêu:** .NET 8.0 LTS (C# 12)  
**Giấy phép (License):** [MIT License](LICENSE)  
**Kho lưu trữ (Repository):** [https://github.com/nguyenhungtran18/TokenVector.Numerics](https://github.com/nguyenhungtran18/TokenVector.Numerics)  
**Gói NuGet:** `TokenVector.Numerics` (v1.0.0)

---

## 🌟 Tổng Quan

Chúng tôi vui mừng công bố bản **phát hành chính thức v1.0.0** của **TokenVector.Numerics** — thư viện tính toán khoa học, tensor đa chiều và động cơ vi phân tự động (Autograd) siêu hiệu năng được xây dựng từ đầu bằng C# 12 và .NET 8 LTS, hoàn toàn không phụ thuộc vào bất kỳ thư viện bên thứ ba nào.

`TokenVector.Numerics` được thiết kế làm hạt nhân tính toán cho ngôn ngữ biên dịch **TokenVector** và là nền tảng toán học độc lập tốc độ cao phục vụ AI, robotics và mô phỏng khoa học trên hệ sinh thái .NET.

---

## 🚀 Các Tính Năng & Điểm Nhấn Nổi Bật

### 1. Động Cơ Vi Phân Tự Động Động (Dynamic Autograd DAG)
* **Xây dựng đồ thị tính toán động:** Hỗ trợ duyệt đồ thị ngược theo thứ tự topo (Topological Backward Pass).
* **Tự động Unbroadcasting Gradient:** Tự động thực hiện phép giảm tổng (Sum-reduction) trên các trục phát tán kích thước 1 trong quá trình lan truyền ngược, đảm bảo gradient luôn khớp 100% với kích thước trọng số.
* **Quản lý vòng đời Tensor:** Hỗ trợ đầy đủ `.backward()`, `.zero_grad()`, `.detach()`, và tích lũy gradient.
* **Nạp chồng toán tử:** Đầy đủ toán tử đại số (`+`, `-`, `*`, `/`, `^`, âm 1 ngôi `-`, phép nhân ma trận `@`).

### 2. Module Nơ-ron & Bộ Tối Ưu Hóa Nâng Cao
* **Kiến trúc Layer:** `Linear<T>` (khởi tạo trọng số Kaiming Uniform), `Sequential<T>`, và `RMSNorm<T>` (Root Mean Square Normalization).
* **Hàm kích hoạt & Hàm mất mát (Loss):** Hỗ trợ tính đạo hàm VJP vector hóa cho `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`, `MSELoss`, và `CrossEntropyLoss` (Log-Sum-Exp ổn định số học).
* **Bộ tối ưu hóa chuẩn công nghiệp:**
  * **SGD:** Stochastic Gradient Descent với Momentum và suy giảm trọng số $L_2$ (Weight Decay).
  * **AdamW:** Thuật toán tối ưu hóa thích nghi với suy giảm trọng số tách rời và hiệu chỉnh độ lệch moment ($m_t, v_t$).

### 3. Phần Cứng SIMD & Bộ Nhớ Native Zero-GC
* **Quản lý bộ nhớ Unmanaged:** Sử dụng `NativeMemory.AllocZeroed` và con trỏ trực tiếp, loại bỏ hoàn toàn hiện tượng dừng Garbage Collection trong các tác vụ tính toán lớn.
* **Tối ưu hóa phần cứng SIMD:** Tận dụng tập lệnh phần cứng `Vector256<float>`, `Vector256<double>`, AVX2, và FMA để khai thác tối đa sức mạnh CPU đa nhân.

### 4. 34 Lĩnh Vực Khoa Học Chuyên Sâu
* **Đại số tuyến tính (`LinAlg`):** Nhân ma trận tốc độ cao ($O(N^3)$ tối ưu), Phân tích suy biến SVD, Tìm trị riêng/Vector riêng (Eigen), Phân rã Cholesky, Ma trận mũ xấp xỉ Padé ($\exp(A)$).
* **Xử lý tín hiệu:** Biến đổi Fourier nhanh 1D/2D (Cooley-Tukey Radix-2 FFT) và Biến đổi Cosine rời rạc (DCT).
* **Cơ học quỹ đạo & Hàng không vũ trụ:** Lan truyền phần tử quỹ đạo Kepler, giải bài toán Lambert, chuyển đổi vector trạng thái quỹ đạo vệ tinh.
* **Robotics & Động lực học:** Động lực học Quadrotor 6 bậc tự do (6-DoF), tính lực cản khí động học, phép xoay không gian 3D bằng Quaternion.
* **Tài chính định lượng:** Định giá quyền chọn châu Âu Black-Scholes, tính hệ số rủi ro Greeks ($\Delta, \Gamma, \Theta, \text{Vega}, \rho$), mô phỏng đường đi Monte Carlo.
* **Điện toán lượng tử:** Mô phỏng statevector Qubit và các cổng logic lượng tử ($H$, $X$, $Y$, $Z$, $\text{CNOT}$, Phase).

---

## 📊 Kết Quả Kiểm Thử Nghiệm Thu (Test Suite)

Toàn bộ thư viện đã được xác minh qua bộ kiểm thử unit test toàn diện với tỷ lệ vượt qua tuyệt đối:

* **Tổng số bài test:** **75 / 75 Tests Passed (100% Pass)**
* **Lỗi (Failures):** **0**
* **Thời gian thực thi:** **~146 ms**
* **Môi trường kiểm thử:** `TokenVector.Numerics.Tests` (xUnit, Release x64)
* **Báo cáo chi tiết:** Xem tại [`TEST_REPORT_VI.md`](TEST_REPORT_VI.md).

---

## 📦 Gói Phát Hành (`dist/`)

Các tệp nhị phân phát hành đã được đóng gói sẵn trong thư mục `dist/`:

| Tệp phát hành | Đường dẫn | Kích thước | Mô tả |
| :--- | :--- | :---: | :--- |
| **Release Zip** | `dist/TokenVector.Numerics-v1.0.0-Release.zip` | `174 KB` | Gói phát hành độc lập hoàn chỉnh |
| **Gói NuGet** | `dist/nuget/TokenVector.Numerics.1.0.0.nupkg` | `123 KB` | Gói cài đặt NuGet chính thức |
| **Thư viện DLL** | `dist/bin/TokenVector.Numerics.dll` | `234 KB` | Thư viện liên kết động .NET 8 standalone |

---

## 💻 Ví Dụ Code C# Mẫu (Hội Tụ Học Mạng XOR)

```csharp
using TokenVector.Numerics.Autograd;
using TokenVector.Numerics.Autograd.NN;
using TokenVector.Numerics.Autograd.Optim;

// 1. Khởi tạo mạng MLP 2 tầng (2 -> 8 -> 1)
var model = new Sequential<float>(
    new Linear<float>(2, 8),
    new Linear<float>(8, 1)
);
var optimizer = new AdamW<float>(model.Parameters(), lr: 0.05f);

// 2. Dữ liệu huấn luyện XOR
var x = Tensor<float>.From(new float[,] { {0,0}, {0,1}, {1,0}, {1,1} }, requiresGrad: false);
var y = Tensor<float>.From(new float[,] { {0}, {1}, {1}, {0} }, requiresGrad: false);

// 3. Vòng lặp huấn luyện
for (int epoch = 0; epoch < 200; epoch++)
{
    optimizer.ZeroGrad();
    var pred = model.Forward(x);
    var loss = pred.MseLoss(y);
    loss.Backward();
    optimizer.Step();
}
```

---

## 🤝 Đóng Góp & Liên Hệ

Chúng tôi trân trọng mọi đóng góp, phản hồi và báo lỗi từ cộng đồng lập trình viên và nghiên cứu khoa học trên [GitHub Repository](https://github.com/nguyenhungtran18/TokenVector.Numerics).

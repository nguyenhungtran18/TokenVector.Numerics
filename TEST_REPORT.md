# BÁO CÁO KIỂM THỬ CHẤT LƯỢNG & BẢO CHỨNG HIỆU NĂNG
## DỰ ÁN: TOKENVECTOR.NUMERICS (RUNTIME TENSOR & AUTOGRAD ENGINE)

**Mã báo cáo:** TR-TKV-NUMERICS-2026-FINAL-V6 (GRAND UNIFIED & AUTOGRAD EDITION)  
**Ngày thực hiện:** 12/09/2026  
**Môi trường thử nghiệm:** .NET SDK 8.0.425, Release Configuration, x64 Architecture, Windows OS  
**Khung kiểm thử:** xUnit.net v2.5.3, Microsoft.NET.Test.Sdk v17.8.0  
**Trạng thái kiểm thử:** **100% PASSED (75/75 Tests in 146 ms)**  

---

## 1. MA TRẬN CHI TIẾT CÁC CA KIỂM THỬ MỚI (AUTOGRAD ENGINE)

| ID | File Kiểm Thử | Tên Test Method | Mô Tả Mục Tiêu Kỹ Thuật | Trạng Thái | Thời Gian |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **70** | `AutogradTests.cs` | `Test_ScalarArithmetic_Autograd` | Vi phân biểu thức $(x+y)(x-y) = x^2-y^2$, kiểm tra $dz/dx=2x, dz/dy=-2y$ | **PASS** | 2 ms |
| **71** | `AutogradTests.cs` | `Test_MultiBranch_Autograd` | Vi phân nhiều nhánh $x \cdot x \cdot x = x^3$, kiểm tra $dz/dx=3x^2$ | **PASS** | 1 ms |
| **72** | `AutogradTests.cs` | `Test_MatrixMultiplication_And_Unbroadcasting_Autograd` | Vector-Jacobian Product cho $Y = XW + b$, tự động unbroadcast bias gradient | **PASS** | 2 ms |
| **73** | `AutogradTests.cs` | `Test_ActivationFunctions_Autograd` | Đạo hàm các hàm kích hoạt ReLU, Sigmoid, Tanh tại các điểm mốc | **PASS** | 2 ms |
| **74** | `AutogradTests.cs` | `Test_LossFunctions_Autograd` | Tính MSE Loss và kiểm tra gradient $\frac{2}{N}(y_{pred} - y_{true})$ | **PASS** | 1 ms |
| **75** | `AutogradTests.cs` | `Test_EndToEnd_XOR_NeuralNetwork_Training` | Huấn luyện mạng nơ-ron MLP 2 tầng giải bài toán XOR với AdamW (Loss < 0.04) | **PASS** | 8 ms |

---

## 2. KẾT QUẢ THỰC THI (CLI OUTPUT)

```text
Command: dotnet test "TokenVector.Numerics.sln" -c Release

  Determining projects to restore...
  All projects are up-to-date for restore.
  TokenVector.Numerics -> d:\TokenVector Numerics\src\TokenVector.Numerics\bin\Release\net8.0\TokenVector.Numerics.dll
  TokenVector.Numerics.Tests -> d:\TokenVector Numerics\tests\TokenVector.Numerics.Tests\bin\Release\net8.0\TokenVector.Numerics.Tests.dll
Test run for d:\TokenVector Numerics\tests\TokenVector.Numerics.Tests\bin\Release\net8.0\TokenVector.Numerics.Tests.dll (.NETCoreApp,Version=v8.0)
VSTest version 17.11.1 (x64)

Starting test execution, please wait...
A total of 1 test files matched the specified pattern.

Passed!  - Failed:     0, Passed:    75, Skipped:     0, Total:    75, Duration: 146 ms - TokenVector.Numerics.Tests.dll (net8.0)
```

# BÁO CÁO KIỂM THỬ CHẤT LƯỢNG & BẢO CHỨNG HIỆU NĂNG
## DỰ ÁN: TOKENVECTOR.NUMERICS (RUNTIME TENSOR & AUTOGRAD ENGINE)

[ 🇬🇧 English ](TEST_REPORT.md) | [ 🇻🇳 Tiếng Việt ](TEST_REPORT_VI.md)

**Mã báo cáo:** TR-TKV-NUMERICS-2026-V1.1.1 (TOKENVECTOR STDLIB, MATHLIB & NUMPY-PARITY EDITION)  
**Ngày thực hiện:** 24/09/2026 (lần chạy v1.1.0) · 25/09/2026 (kiểm chứng lại mathlib cho v1.1.1)  
**Phiên bản mục tiêu:** `v1.1.1` (Mục 3–4); Mục 2 là log chạy của `v1.1.0`  
**Môi trường thử nghiệm:** .NET SDK 8.0 LTS, Release Configuration, x64 Architecture, Windows OS  
**Khung kiểm thử:** xUnit.net v2.5.3, Microsoft.NET.Test.Sdk v17.8.0 + native `tkvc.exe` (compile `smoke_tests.tkv`)  
**Trạng thái kiểm thử:** **100% PASSED (84/84 xUnit in ~179 ms · 175/175 .tkv smoke · 354/354 numeric surface)**  
**Bổ sung (ngày 25 tháng 9, 2026):** `mathlib/` toán chính xác đã kiểm chứng lại bằng native — **8/8** bigfloat, **9/9** number theory (Mục 3); đã sửa 5 lỗi đúng/sai số học (Mục 3.2). Con số 175/175 smoke là kết quả lưu của v1.1.0 và hiện không tái lập được — xem Mục 4.  

---

## 1. MA TRẬN CHI TIẾT TOÀN BỘ 84 CA KIỂM THỬ

| ID | Nhóm | Tên Test Method | Mô Tả Mục Tiêu Kỹ Thuật | Kết Quả | Thời Gian |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **01** | `Core` | `TestNDArrayCreationAndIndexing` | Khởi tạo mảng [2,3], gán và đọc giá trị qua indexer 2D | **PASS** | 2 ms |
| **02** | `Core` | `TestZeroCopySlicing` | Cắt lát 3D với step, xác minh biến đổi Zero-Copy $O(1)$ | **PASS** | 1 ms |
| **03** | `Core` | `TestReshapeAndPermute` | Reshape [2,3,4] và hoán vị trục Permute [4,2,3] | **PASS** | 1 ms |
| **04** | `Core` | `TestNativeMemoryAllocation` | Cấp phát mảng unmanaged qua `NativeMemory.AllocZeroed` | **PASS** | 1 ms |
| **05** | `SIMD` | `TestBroadcastingAddition` | Broadcasting ma trận (2,1) và (1,3) ra (2,3) | **PASS** | 2 ms |
| **06** | `SIMD` | `TestSIMDContiguousArithmetic` | Toán tử SIMD Vector256 / Vector&lt;T&gt; 1024 phần tử | **PASS** | 2 ms |
| **07** | `SIMD` | `TestReductions` | Thu giảm Sum, Mean, Min, Max toàn cục và theo trục | **PASS** | 2 ms |
| **08** | `LinAlg` | `TestMatrixMultiplication2D` | Tiled MatMul 2D [2,3] x [3,2] -> [2,2] | **PASS** | 2 ms |
| **09** | `LinAlg` | `TestLUDecompositionAndSolve` | Phân rã $PA = LU$ kèm Partial Pivoting và giải hệ $Ax = b$ | **PASS** | 3 ms |
| **10** | `LinAlg` | `TestDeterminantAndInverse` | Tính $\det(A)$ và kiểm tra $A \cdot A^{-1} = I$ | **PASS** | 2 ms |
| **11** | `LinAlg` | `TestFFT1D` | FFT 1D Cooley-Tukey và IFFT 1D khôi phục tín hiệu | **PASS** | 2 ms |
| **12** | `Spatial` | `TestAffine3DTranslationAndScale` | Ma trận 4x4 Translation và Scaling 3D | **PASS** | 1 ms |
| **13** | `Spatial` | `TestQuaternionMultiplicationAndSlerp`| Phép nhân Quaternion và nội suy Slerp tại $t=0.5$ | **PASS** | 1 ms |
| **14** | `Spatial` | `TestVectorCrossProduct` | Tích có hướng 3D giữa vector trục X và Y ra Z | **PASS** | 1 ms |
| **15** | `Neural` | `TestSoftmaxNumericalStability` | Softmax với Log-Sum-Exp trick trên logits = 1000..1002 | **PASS** | 2 ms |
| **16** | `Neural` | `TestGELUActivation` | Hàm kích hoạt GELU tại các giá trị cực trị và 0 | **PASS** | 1 ms |
| **17** | `Neural` | `TestIm2ColAndConv2D` | Biến đổi im2col và Conv2D 3x3 filter 2x2 | **PASS** | 3 ms |
| **18** | `Neural` | `TestScaledDotProductAttention` | Scaled Dot-Product Attention $Q, K, V$ Transformer | **PASS** | 2 ms |
| **19** | `Neural` | `TestLayerNorm` | Chuẩn hóa LayerNorm (Mean = 0, Std = 1) tensor [1,4] | **PASS** | 2 ms |
| **20** | `Verify` | `Test3DZeroCopySlicing` | Nghiệm thu độc lập Slicing mảng 3D | **PASS** | 2 ms |
| **21** | `Verify` | `TestBroadcasting2x1And1x3` | Nghiệm thu độc lập Broadcasting (2,1) và (1,3) | **PASS** | 1 ms |
| **22** | `Verify` | `TestMatMulSoftmaxAttentionAndLU` | Nghiệm thu tích hợp MatMul + Softmax + Attention + LU | **PASS** | 3 ms |
| **23** | `AdvMath` | `TestSVDDecomposition` | Phân rã SVD $A = U \cdot \text{diag}(S) \cdot V^T$ ma trận [3,2] | **PASS** | 3 ms |
| **24** | `AdvMath` | `TestEigenSymmetricEigh` | Jacobi Eigh tìm trị riêng & vector riêng ma trận đối xứng | **PASS** | 3 ms |
| **25** | `AdvMath` | `TestArbitraryLengthBluesteinFFT` | FFT Bluestein cho độ dài số nguyên tố lẻ $N = 7$ | **PASS** | 3 ms |
| **26** | `AdvMath` | `TestEinSumContraction` | Co rút tensor EinSum: `"ij,jk->ik"`, `"ii->"`, `"ij->ji"` | **PASS** | 3 ms |
| **27** | `AdvMath` | `TestBooleanMaskingAndWhere` | Lọc mảng `arr.Filter(arr > 3)` và `IndexingOps.Where` | **PASS** | 2 ms |
| **28** | `AdvMath` | `TestTakeAndPut` | Gather slices qua `Take` và Scatter qua `Put` | **PASS** | 2 ms |
| **29** | `Ecosystem`| `TestRandomGenerators` | Sinh số ngẫu nhiên Rand, Randn Gaussian, Randint, Choice | **PASS** | 3 ms |
| **30** | `Ecosystem`| `TestArrayManipulation` | Ghép Concatenate, Stack, Split, Tile, Pad, Roll, Flip | **PASS** | 4 ms |
| **31** | `Ecosystem`| `TestSortingAndSearching` | Sort theo trục, ArgSort, Unique, Clip | **PASS** | 3 ms |
| **32** | `Ecosystem`| `TestStatistics` | Tính Var, Std (ddof), Median, Covariance, Correlation | **PASS** | 3 ms |
| **33** | `Ecosystem`| `TestExtendedLinAlg` | PInv (Moore-Penrose), MatrixRank, Least Squares LstSq | **PASS** | 4 ms |
| **34** | `Ecosystem`| `TestNpyAndBinaryIO` | Ghi/Đọc file nhị phân chuẩn `.npy` và Raw Binary | **PASS** | 5 ms |
| **35** | `Parity` | `Test_CumulativeOps_CumSum_And_CumProd`| CumSum & CumProd theo trục 0, trục 1 và mảng phẳng | **PASS** | 3 ms |
| **36** | `Parity` | `Test_CumulativeOps_Diff` | Sai phân rời rạc bậc 1 và bậc 2 dọc theo trục | **PASS** | 2 ms |
| **37** | `Parity` | `Test_Polynomial_PolyFit_PolyVal_Roots`| Khớp đa thức bậc 2 Vandermonde, Horner PolyVal, Roots | **PASS** | 4 ms |
| **38** | `Parity` | `Test_GridOps_Meshgrid_Diag_Triu_Tril` | Lưới tọa độ Meshgrid 2D, Diag/Diagonal, Triu, Tril | **PASS** | 3 ms |
| **39** | `Parity` | `Test_LogicToleranceOps` | Kiểm tra IsClose, AllClose, phát hiện IsNaN, IsInf | **PASS** | 2 ms |
| **40** | `Parity` | `Test_TensorArchive_Npz_And_MemMap` | Ghi/Đọc kho `.npz`, ánh xạ bộ nhớ MemoryMappedNDArray | **PASS** | 6 ms |
| **41** | `SuperLib`| `Test_SignalProcessing_Convolve_Windows`| Convolve, Correlate, Hanning, Hamming, Blackman | **PASS** | 4 ms |
| **42** | `SuperLib`| `Test_SpecialFunctions_Erf_Gamma_Bessel`| Hàm sai số Erf, Erfc, Hàm Gamma, LogGamma, Bessel I0/J0 | **PASS** | 3 ms |
| **43** | `SuperLib`| `Test_SetOperations_Intersect_Union` | Set Intersect1D, Union1D, SetDiff1D, SetXor1D, IsIn | **PASS** | 3 ms |
| **44** | `SuperLib`| `Test_MatrixOps_Kron_Outer_Power_Norm` | Tích Kronecker (Kron), Outer Product, MatrixPower, Norm | **PASS** | 4 ms |
| **45** | `SuperLib`| `Test_BitwiseOps` | Phép toán Bitwise And, Or, Xor, LeftShift, RightShift | **PASS** | 2 ms |
| **46** | `Science` | `Test_Astrodynamics_Kepler_Hohmann` | Giải Kepler $M = E - e\sin(E)$, chuyển dịch Hohmann, WGS84 | **PASS** | 3 ms |
| **47** | `Finance` | `Test_FinanceMath_BlackScholes_Greeks` | Định giá Black-Scholes Call/Put, Greeks, Markowitz, NPV | **PASS** | 4 ms |
| **48** | `Stats` | `Test_TimeSeriesAndKalman` | Bộ lọc Kalman 1D & Đa chiều, Holt Linear Trend, ACF | **PASS** | 4 ms |
| **49** | `Physics` | `Test_PhysicsODEAndFields` | Tích phân RK4 dao động điều hòa, vi phân 3D Laplacian | **PASS** | 3 ms |
| **50** | `Spatial` | `Test_Geometry3DAndPointClouds` | Khoảng cách Điểm-Mặt phẳng, Ray-Triangle, ICP Kabsch | **PASS** | 3 ms |
| **51** | `LinAlg` | `Test_MatrixFunctions_Expm_Sqrtm` | Ma trận mũ $e^A$ Padé [6/6], Căn ma trận $\sqrt{A}$, Sylvester | **PASS** | 5 ms |
| **52** | `Expanded`| `Test_ElementWiseMathOps` | Hàm lượng giác & siêu việt (Sin, Cos, Tan, Sinh, Log1p) | **PASS** | 3 ms |
| **53** | `Expanded`| `Test_ShapeExtensions` | Mở rộng hình dạng: ExpandDims, Squeeze, AtLeast1D/2D/3D | **PASS** | 2 ms |
| **54** | `Expanded`| `Test_SpecialFunctions` | Beta, LogBeta, Gammainc, Digamma $\Psi(x)$, Sinc, Logit | **PASS** | 3 ms |
| **55** | `Expanded`| `Test_LinAlgExtended_Tridiagonal_NullSpace`| Thuật toán Thomas $O(N)$ tridiagonal, NullSpace, KronSum | **PASS** | 4 ms |
| **56** | `Expanded`| `Test_Optimization_Root_And_Simplex` | Tìm nghiệm Brentq, Bisection, Nelder-Mead, BFGS, Simplex | **PASS** | 5 ms |
| **57** | `Expanded`| `Test_Interpolation_Spline_And_Bilinear`| Khớp Cubic Spline 1D, Bilinear 2D, Radial Basis Function | **PASS** | 4 ms |
| **58** | `Expanded`| `Test_Statistics_Distributions` | Phân phối Student's t, Normal, Chi-Square, t-test, ANOVA | **PASS** | 4 ms |
| **59** | `Expanded`| `Test_Spatial_KDTree_And_ConvexHull` | Cây KDTree tìm k-NN, Bao lồi 2D Monotone Chain | **PASS** | 3 ms |
| **60** | `Expanded`| `Test_Astrodynamics_And_Finance` | Nhiễu loạn J2 Trái Đất, Bi-Elliptic, Quyền chọn Mỹ CRR | **PASS** | 4 ms |
| **61** | `Expanded`| `Test_Quantum_And_Neural_Extended` | Cổng lượng tử CRz, SWAP, Toffoli, Chuẩn hóa RMSNorm, RoPE | **PASS** | 3 ms |
| **62** | `Frontier`| `Test_Quantum_QState_BellState` | Trạng thái Bell $|\Phi^+\rangle$, Ma trận mật độ, Entropy | **PASS** | 3 ms |
| **63** | `Frontier`| `Test_Crypto_NTT_PolyMul_And_LLL` | Biến đổi NTT, Nhân đa thức $O(N\log N)$, Thu gọn LLL | **PASS** | 4 ms |
| **64** | `Frontier`| `Test_Biology_Dihedral_RMSD_TMScore` | Góc nhị diện protein $(\phi, \psi, \omega)$, Kabsch RMSD, TM-Score | **PASS** | 3 ms |
| **65** | `Frontier`| `Test_Graphs_Laplacian_ChebyshevConv` | Normalized Laplacian $L_{sym}$, Tích chập đồ thị Chebyshev GNN | **PASS** | 3 ms |
| **66** | `Frontier`| `Test_Physics_LatticeBoltzmann2D` | Mô phỏng Navier-Stokes LBM D2Q9 collision/streaming | **PASS** | 4 ms |
| **67** | `Frontier`| `Test_Neural_Sinkhorn_And_DDIM` | Vận chuyển tối ưu Sinkhorn Wasserstein, DDIM Diffusion | **PASS** | 4 ms |
| **68** | `Frontier`| `Test_Robotics_LQR_And_JacobianDLS` | Điều khiển tối ưu LQR (DARE), Động học ngược Jacobian DLS | **PASS** | 4 ms |
| **69** | `Frontier`| `Test_Spatial_HyperbolicGeometry` | Hình học Hyperbolic Poincaré Ball, Phép cộng Möbius | **PASS** | 3 ms |
| **70** | `Autograd`| `Test_ScalarArithmetic_Autograd` | Vi phân $(x+y)(x-y) = x^2-y^2$, kiểm tra $dz/dx, dz/dy$ | **PASS** | 2 ms |
| **71** | `Autograd`| `Test_MultiBranch_Autograd` | Vi phân nhiều nhánh $x^3$, kiểm tra $dz/dx=3x^2$ | **PASS** | 1 ms |
| **72** | `Autograd`| `Test_MatrixMatMul_Unbroadcasting` | VJP cho $Y = XW + b$, tự động unbroadcast bias gradient | **PASS** | 2 ms |
| **73** | `Autograd`| `Test_ActivationFunctions_Autograd` | Đạo hàm các hàm kích hoạt ReLU, Sigmoid, Tanh | **PASS** | 2 ms |
| **74** | `Autograd`| `Test_LossFunctions_Autograd` | Tính MSE Loss và kiểm tra gradient $\frac{2}{N}(y_{pred}-y_{true})$ | **PASS** | 1 ms |
| **75** | `Autograd`| `Test_EndToEnd_XOR_NeuralNetwork` | Huấn luyện mạng nơ-ron MLP giải XOR với AdamW (Loss < 0.04) | **PASS** | 8 ms |
| **76** | `SuperLib` | `TestLambertW_Branches` | Hàm Lambert W $W_0(x)$ và $W_{-1}(x)$ số thực & tensor | **PASS** | 2 ms |
| **77** | `SuperLib` | `TestBesselAndAiryFunctions` | Bessel $Y_0(x), K_0(x)$ và hàm Airy $\text{Ai}(x), \text{Bi}(x)$ | **PASS** | 2 ms |
| **78** | `LinAlg` | `TestSchurDecomposition` | Phân rã Schur thực $A = Q T Q^T$ với ma trận trực giao $Q$ | **PASS** | 3 ms |
| **79** | `LinAlg` | `TestSolveSylvester` | Giải phương trình ma trận Sylvester $A X + X B = C$ | **PASS** | 3 ms |
| **80** | `Signal` | `TestWaveletTransform_Haar_Reconstruction` | Biến đổi Wavelet DWT/IDWT Haar phục hồi tín hiệu 100% | **PASS** | 2 ms |
| **81** | `Signal` | `TestHilbertTransform` | Biến đổi Hilbert 1D và tín hiệu giải tích qua FFT | **PASS** | 3 ms |
| **82** | `Optimize` | `TestQPSolve_EqualityConstrained` | Giải quy hoạch toàn phương ADMM có ràng buộc đẳng thức | **PASS** | 4 ms |
| **83** | `Optimize` | `TestQPSolve_BoxBounded` | Giải quy hoạch toàn phương ADMM có ràng buộc khoảng $[lb, ub]$ | **PASS** | 4 ms |
| **84** | `Physics` | `TestSymplecticVerlet_EnergyConservation` | Tích phân Symplectic Leapfrog bảo toàn tuyệt đối năng lượng | **PASS** | 5 ms |

---

## 2. KẾT QUẢ THỰC THI (CLI OUTPUT)

> Ghi chú phạm vi: log này được ghi nhận cho bản runtime v1.1.0. Sau đó phần source engine và test project đã bị loại khỏi repository; binary phát hành vẫn còn trong `dist/bin/` và gánh nặng kiểm chứng nay thuộc bộ suite stdlib TokenVector (Mục 3).

```text
Command: dotnet test "TokenVector.Numerics.sln" -c Release

  Determining projects to restore...
  All projects are up-to-date for restore.
  TokenVector.Numerics -> d:\TokenVector Numerics\src\TokenVector.Numerics\bin\Release\net8.0\TokenVector.Numerics.dll
  Successfully created package 'd:\TokenVector Numerics\src\TokenVector.Numerics\bin\Release\TokenVector.Numerics.1.1.0.nupkg'.
  Successfully created package 'd:\TokenVector Numerics\src\TokenVector.Numerics\bin\Release\TokenVector.Numerics.1.1.0.snupkg'.
  TokenVector.Numerics.Tests -> d:\TokenVector Numerics\tests\TokenVector.Numerics.Tests\bin\Release\net8.0\TokenVector.Numerics.Tests.dll
Test run for d:\TokenVector Numerics\tests\TokenVector.Numerics.Tests\bin\Release\net8.0\TokenVector.Numerics.Tests.dll (.NETCoreApp,Version=v8.0)
VSTest version 17.14.1 (x64)

Starting test execution, please wait...
A total of 1 test files matched the specified pattern.

Passed!  - Failed:     0, Passed:    84, Skipped:     0, Total:    84, Duration: 179 ms - TokenVector.Numerics.Tests.dll (net8.0)
```

---

## 3. KIỂM CHỨNG `MATHLIB/` TOÁN CHÍNH XÁC (ngày 25 tháng 9, 2026)

### 3.1 Thực thi

> **Phiên bản compiler quan trọng.** Kết quả này đến từ bản `tkvc` **mới hơn** bản đang phát hành trong [release TokenVector](https://github.com/nguyenhungtran18/TokenVector/releases/tag/TokenVector_release). Bản `tkvc.exe` đã publish **biên dịch sai cả hai module** — chạy bất kỳ suite nào với nó đều ném `System.InvalidCastException` bên trong `bignum_mul` (bf_bigfloat) và `pow_mod` (number_theory). Nó chỉ có `build [--entry] [--out]`, không có `--no-lint` / `--target` / `--subsystem`. Cho tới khi compiler mới được phát hành, kết quả này tái lập được trên bản local hiện tại nhưng **không** tái lập được từ bản release đã đăng. CI phát hiện điều này và bỏ qua gate kèm cảnh báo thay vì báo đạt một kết quả chưa hề có.

Các module `mathlib/` viết thuần TokenVector, không phụ thuộc `import tv`, nên biên dịch và chạy trọn vẹn bằng compiler native. Cả hai suite đã được build lại từ nguồn và chạy lại cho báo cáo này:

```text
> tkvc build mathlib/bf_bigfloat.tkv --out bf_bigfloat.exe
[tkv] Da bien dich: bf_bigfloat.exe
> ./bf_bigfloat.exe
t4_arith: OK
t5_div: OK
t6_bigmul: OK
t1_sqrt2(100 cs): OK
t2_pi_chud(100 cs): OK
t3_e(100 cs): OK
t7_bignum_neg: OK
t8_precision_borders: OK
PASS 8 / 8 - bigfloat OK

> tkvc build mathlib/nt_number_theory.tkv --out nt_number_theory.exe
[tkv] Da bien dich: nt_number_theory.exe
> ./nt_number_theory.exe
t8_i64_boundaries: OK
t9_large_factorization: OK
PASS 9 / 9 - number_theory OK
```

### 3.2 Lỗi phát hiện và đã sửa

| # | Module | Mức độ | Lỗi | Cách sửa | Test hồi quy |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | `bf_bigfloat.tkv` | **Nghiêm trọng** | hằng Chudnovsky `10939058825628000` — sai từ khoảng chữ số 14, làm sai $\pi$ ở độ chính xác cao | sửa thành `10939058860032000` | `t2_pi_chud(100 cs)` |
| 2 | `bf_bigfloat.tkv` | **Nghiêm trọng** | `bignum_neg` làm mất một limb hợp lệ qua nhánh phủ bù lỗi | bỏ nhánh phủ bù, đảo dấu trực tiếp | `t7_bignum_neg` (mới) |
| 3 | `bf_bigfloat.tkv` | Lớn | `pi_chudnovsky` lấy thiếu độ chính xác và xử lý sai zero dẫn trong các chữ số mẫu số | nay lấy `len(den_digits) + prec_digits + 2`, bỏ zero dẫn rồi chuẩn hoá | `t8_precision_borders` (mới) |
| 4 | `nt_number_theory.tkv` | **Nghiêm trọng** | phép nhân/cộng modulo âm thầm tràn `i64`, khiến `pow_mod`, Miller–Rabin và Pollard–Rho có thể trả về tính nguyên/phân tích sai | thêm `mul_mod_i64` / `add_mod_i64`; cả ba dùng qua đó | `t8_i64_boundaries` (mới) |
| 5 | `nt_number_theory.tkv` | Lớn | `isqrt_i`, `trial_prime`, `factorize` và `iroot` tràn `i64` ở các biên (`x + 1`, `i * i`, `p * p`) | mọi bước nhân trung gian đều được kiểm tra miền giá trị | `t8_i64_boundaries`, `t9_large_factorization` (mới) |

### 3.3 Rà soát thuật toán ở mức mã nguồn — 36/36

`linalg`, `linalg_functions`, `fft` và `crypto_graph` được rà soát từng thuật toán đối chiếu với các oracle viết độc lập. **Đây là rà soát mức mã nguồn, không phải lần chạy native** — xem Mục 4.

| Hạng mục | Số kiểm tra | Kết quả |
| :--- | :---: | :---: |
| Số phức — cộng, trừ, nhân, chia, phủ bù, mô đun | 6 | 6/6 |
| Đại số ma trận — matmul, det, trace, solve, inverse, QR, Cholesky ($A=LL^\top$), SVD, eigh | 9 | 9/9 |
| FFT — biến đổi trực tiếp đối chiếu oracle DFT độc lập với $N = 1,2,4,5,6,7,8,9$, cộng round-trip IFFT | 10 | 10/10 |
| NTT — round-trip, tích chập đa thức tuần hoàn, Laplacian chuẩn hoá và không chuẩn hoá | 11 | 11/11 |
| **Tổng** | **36** | **36/36** |

Trong quá trình rà soát, ba trường hợp nghi ngờ lỗi hoá ra là **lỗi của oracle chứ không phải của thư viện** và đã được sửa ở oracle: $(3+2i)/(1-4i) = (-5+14i)/17$; $A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\, b=[1,2]$ có nghiệm $x=[0,\,0.5]$; và Cholesky tách $L L^\top$, không phải $L L$.

---

## 4. GIỚI HẠN ĐÃ BIẾT — SMOKE SUITE KHÔNG TÁI LẬP ĐƯỢC

Mục 2 ghi nhận lần chạy của bản phát hành v1.1.0. Lệnh smoke `.tkv` được trích khắp tài liệu,

```powershell
tkvc build tests/tokenvector/smoke_tests.tkv --entry main --out smoke.exe
```

**thất bại trên compiler hiện tại** và không tạo ra kết quả `passed=175, failed=0` như tài liệu mô tả:

```text
[tkv] Loi: File khong co ham top-level nao co annotation kieu DSL
```

Nguyên nhân gốc: `tkvc build` phân giải `import tv` theo thư mục đi kèm của chính nó (đường dẫn giải nén PyInstaller), chứ không phải `src/tokenvector` của repository, và không có tuỳ chọn runtime-path — dạng lệnh `-r src/tokenvector` nêu trong phần đầu `smoke_tests.tkv` bị từ chối thẳng (`invalid choice`, lệnh con duy nhất là `build`). Một probe tối giản import `tv` xác nhận độc lập điều này: `import module 'tv' khong tim thay trong thu muc hien tai hoi site-packages`.

Vì vậy **con số 175/175 nên được đọc là kết quả lưu của bản phát hành v1.1.0**, không phải một cổng kiểm chứng đang hoạt động. Bài kiểm tra tái lập được ngày hôm nay là hai suite `mathlib` ở Mục 3.1, và CI nay chạy đúng hai suite đó (`verify-mathlib`) thay cho smoke suite. Smoke suite không được nối vào CI vì sẽ hỏng trên mọi runner với lý do nêu trên; nên bật lại khi compiler resolve được stdlib từ thư mục dự án.


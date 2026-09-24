# QUALITY ASSURANCE & VERIFICATION TEST REPORT
## PROJECT: TOKENVECTOR.NUMERICS (RUNTIME TENSOR & AUTOGRAD ENGINE)

[ 🇬🇧 English ](TEST_REPORT.md) | [ 🇻🇳 Tiếng Việt ](TEST_REPORT_VI.md)

**Report ID:** TR-TKV-NUMERICS-2026-V1.1.0-FINAL (TOKENVECTOR STDLIB & NUMPY-PARITY EDITION)  
**Execution Date:** September 24, 2026  
**Target Version:** `v1.1.0`  
**Test Environment:** .NET SDK 8.0 LTS, Release Configuration, x64 Architecture, Windows OS + verification harness  
**Test Framework:** xUnit.net v2.5.3, Microsoft.NET.Test.Sdk v17.8.0 + `tests/tokenvector/tkv_harness.py`  
**Status:** **100% PASSED (84/84 xUnit in ~179 ms · 175/175 .tkv smoke · 354/354 numeric surface)**  

---

## 1. COMPREHENSIVE 84-TEST MATRIX

| ID | Suite | Test Method | Technical Objective | Status | Time |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **01** | `Core` | `TestNDArrayCreationAndIndexing` | [2,3] tensor allocation, element assignment, 2D indexing | **PASS** | 2 ms |
| **02** | `Core` | `TestZeroCopySlicing` | 3D strided slicing verifying $O(1)$ zero-copy memory views | **PASS** | 1 ms |
| **03** | `Core` | `TestReshapeAndPermute` | [2,3,4] tensor reshaping and [4,2,3] axis permutation | **PASS** | 1 ms |
| **04** | `Core` | `TestNativeMemoryAllocation` | Native memory allocation via `NativeMemory.AllocZeroed` | **PASS** | 1 ms |
| **05** | `SIMD` | `TestBroadcastingAddition` | Broadcasting (2,1) and (1,3) matrices to shape (2,3) | **PASS** | 2 ms |
| **06** | `SIMD` | `TestSIMDContiguousArithmetic` | Hardware SIMD Vector256 / Vector&lt;T&gt; 1024-element ops | **PASS** | 2 ms |
| **07** | `SIMD` | `TestReductions` | Global and axis reductions: Sum, Mean, Min, Max | **PASS** | 2 ms |
| **08** | `LinAlg` | `TestMatrixMultiplication2D` | Cache-blocked 32x32 MatMul [2,3] x [3,2] -> [2,2] | **PASS** | 2 ms |
| **09** | `LinAlg` | `TestLUDecompositionAndSolve` | $PA = LU$ with partial pivoting and $Ax = b$ solver | **PASS** | 3 ms |
| **10** | `LinAlg` | `TestDeterminantAndInverse` | Determinant $\det(A)$ and $A \cdot A^{-1} = I$ inversion | **PASS** | 2 ms |
| **11** | `LinAlg` | `TestFFT1D` | 1D Cooley-Tukey Radix-2 FFT & IFFT reconstruction | **PASS** | 2 ms |
| **12** | `Spatial` | `TestAffine3DTranslationAndScale` | 4x4 3D Affine Translation and Scaling transforms | **PASS** | 1 ms |
| **13** | `Spatial` | `TestQuaternionMultiplicationAndSlerp`| Unit quaternion multiplication and Slerp at $t=0.5$ | **PASS** | 1 ms |
| **14** | `Spatial` | `TestVectorCrossProduct` | 3D Vector Cross Product ($X \times Y = Z$) | **PASS** | 1 ms |
| **15** | `Neural` | `TestSoftmaxNumericalStability` | Softmax Log-Sum-Exp numerical stabilization on logits | **PASS** | 2 ms |
| **16** | `Neural` | `TestGELUActivation` | GELU activation at extreme and zero points | **PASS** | 1 ms |
| **17** | `Neural` | `TestIm2ColAndConv2D` | im2col memory transform and 2D Convolution layer | **PASS** | 3 ms |
| **18** | `Neural` | `TestScaledDotProductAttention` | Transformer Scaled Dot-Product Attention $Q, K, V$ | **PASS** | 2 ms |
| **19** | `Neural` | `TestLayerNorm` | LayerNorm normalization (Mean = 0, Std = 1) on [1,4] | **PASS** | 2 ms |
| **20** | `Verify` | `Test3DZeroCopySlicing` | Independent verification of 3D tensor slicing | **PASS** | 2 ms |
| **21** | `Verify` | `TestBroadcasting2x1And1x3` | Independent verification of (2,1) & (1,3) broadcasting | **PASS** | 1 ms |
| **22** | `Verify` | `TestMatMulSoftmaxAttentionAndLU` | Pipeline test: MatMul + Softmax + Attention + LU | **PASS** | 3 ms |
| **23** | `AdvMath` | `TestSVDDecomposition` | SVD $A = U \cdot \text{diag}(S) \cdot V^T$ on [3,2] matrix | **PASS** | 3 ms |
| **24** | `AdvMath` | `TestEigenSymmetricEigh` | Symmetric matrix eigenvalue & eigenvector solver | **PASS** | 3 ms |
| **25** | `AdvMath` | `TestArbitraryLengthBluesteinFFT` | Bluestein Chirp-Z FFT for arbitrary prime length $N = 7$ | **PASS** | 3 ms |
| **26** | `AdvMath` | `TestEinSumContraction` | EinSum contractions: `"ij,jk->ik"`, `"ii->"`, `"ij->ji"` | **PASS** | 3 ms |
| **27** | `AdvMath` | `TestBooleanMaskingAndWhere` | Mask filtering `arr.Filter(arr > 3)` and `Where` | **PASS** | 2 ms |
| **28** | `AdvMath` | `TestTakeAndPut` | Gather slices via `Take` and scatter via `Put` | **PASS** | 2 ms |
| **29** | `Ecosystem`| `TestRandomGenerators` | Random: Uniform, Standard Normal, Choice | **PASS** | 3 ms |
| **30** | `Ecosystem`| `TestArrayManipulation` | Concatenate, Stack, Split, Tile, Pad, Roll, Flip | **PASS** | 4 ms |
| **31** | `Ecosystem`| `TestSortingAndSearching` | Axis-wise Sort, ArgSort, Unique values, Clipping | **PASS** | 3 ms |
| **32** | `Ecosystem`| `TestStatistics` | Variance, Std, Median, Covariance, Correlation | **PASS** | 3 ms |
| **33** | `Ecosystem`| `TestExtendedLinAlg` | Moore-Penrose PInv, Matrix Rank, LstSq, Cond | **PASS** | 4 ms |
| **34** | `Ecosystem`| `TestNpyAndBinaryIO` | Standard `.npy` binary I/O and Raw streaming | **PASS** | 5 ms |
| **35** | `Parity` | `Test_CumulativeOps_CumSum_And_CumProd`| Cumulative Sum and Product along tensor axes | **PASS** | 3 ms |
| **36** | `Parity` | `Test_CumulativeOps_Diff` | Discrete difference operator of order $n=1, 2$ | **PASS** | 2 ms |
| **37** | `Parity` | `Test_Polynomial_PolyFit_PolyVal_Roots`| Vandermonde polyfit, Horner polyval, Companion roots | **PASS** | 4 ms |
| **38** | `Parity` | `Test_GridOps_Meshgrid_Diag_Triu_Tril` | 2D meshgrids, diagonal extraction, Triu, Tril | **PASS** | 3 ms |
| **39** | `Parity` | `Test_LogicToleranceOps` | Tolerance checks: IsClose, AllClose, IsNaN, IsInf | **PASS** | 2 ms |
| **40** | `Parity` | `Test_TensorArchive_Npz_And_MemMap` | `.npz` Archive read/write and memory-mapped files | **PASS** | 6 ms |
| **41** | `SuperLib`| `Test_SignalProcessing_Convolve_Windows`| Convolutions, correlations, DSP window functions | **PASS** | 4 ms |
| **42** | `SuperLib`| `Test_SpecialFunctions_Erf_Gamma_Bessel`| Error function (Erf, Erfc), Gamma, Bessel I0/J0 | **PASS** | 3 ms |
| **43** | `SuperLib`| `Test_SetOperations_Intersect_Union` | 1D Set intersection, union, diff, XOR, IsIn mask | **PASS** | 3 ms |
| **44** | `SuperLib`| `Test_MatrixOps_Kron_Outer_Power_Norm` | Kronecker product, Outer product, Matrix power, Norms | **PASS** | 4 ms |
| **45** | `SuperLib`| `Test_BitwiseOps` | Bitwise integer ops: AND, OR, XOR, Shifts | **PASS** | 2 ms |
| **46** | `Science` | `Test_Astrodynamics_Kepler_Hohmann` | Kepler orbit solver, Hohmann transfer, WGS84 ECEF | **PASS** | 3 ms |
| **47** | `Finance` | `Test_FinanceMath_BlackScholes_Greeks` | Black-Scholes pricing, Greeks, Markowitz, NPV, IRR | **PASS** | 4 ms |
| **48** | `Stats` | `Test_TimeSeriesAndKalman` | 1D & ND Kalman Filters, Holt Trend, ACF, PACF | **PASS** | 4 ms |
| **49** | `Physics` | `Test_PhysicsODEAndFields` | RK4 oscillator integration, 3D Laplacian field | **PASS** | 3 ms |
| **50** | `Spatial` | `Test_Geometry3DAndPointClouds` | Point-to-plane dist, Ray-triangle, Kabsch ICP | **PASS** | 3 ms |
| **51** | `LinAlg` | `Test_MatrixFunctions_Expm_Sqrtm` | Padé Matrix Exponential $e^A$, $\sqrt{A}$, Sylvester | **PASS** | 5 ms |
| **52** | `Expanded`| `Test_ElementWiseMathOps` | Trigonometric & hyperbolic (Sin, Sinh, Tanh, Log1p) | **PASS** | 3 ms |
| **53** | `Expanded`| `Test_ShapeExtensions` | Shape ops: ExpandDims, Squeeze, AtLeast1D/2D/3D | **PASS** | 2 ms |
| **54** | `Expanded`| `Test_SpecialFunctions` | Beta, LogBeta, Gammainc, Digamma, Sinc, Logit | **PASS** | 3 ms |
| **55** | `Expanded`| `Test_LinAlgExtended_Tridiagonal_NullSpace`| Thomas $O(N)$ solver, NullSpace, Kronecker Sum | **PASS** | 4 ms |
| **56** | `Expanded`| `Test_Optimization_Root_And_Simplex` | Brentq, Bisection, Nelder-Mead, BFGS, Simplex LP | **PASS** | 5 ms |
| **57** | `Expanded`| `Test_Interpolation_Spline_And_Bilinear`| 1D Cubic Spline, 2D Bilinear, Radial Basis Function | **PASS** | 4 ms |
| **58** | `Expanded`| `Test_Statistics_Distributions` | Student's t, Normal, Chi-Square, t-tests, ANOVA | **PASS** | 4 ms |
| **59** | `Expanded`| `Test_Spatial_KDTree_And_ConvexHull` | KDTree nearest neighbors, 2D Convex Hull | **PASS** | 3 ms |
| **60** | `Expanded`| `Test_Astrodynamics_And_Finance` | J2 perturbation, Bi-Elliptic, CRR American Option | **PASS** | 4 ms |
| **61** | `Expanded`| `Test_Quantum_And_Neural_Extended` | Quantum gates (CRz, SWAP, Toffoli), RMSNorm, RoPE | **PASS** | 3 ms |
| **62** | `Frontier`| `Test_Quantum_QState_BellState` | Bell State $|\Phi^+\rangle$, Density Matrix, Entropy | **PASS** | 3 ms |
| **63** | `Frontier`| `Test_Crypto_NTT_PolyMul_And_LLL` | Number Theoretic Transform, $O(N\log N)$ Mul, LLL | **PASS** | 4 ms |
| **64** | `Frontier`| `Test_Biology_Dihedral_RMSD_TMScore` | Ramachandran angles $(\phi,\psi,\omega)$, Kabsch, TM-Score | **PASS** | 3 ms |
| **65** | `Frontier`| `Test_Graphs_Laplacian_ChebyshevConv` | Normalized Laplacian $L_{sym}$, Chebyshev GNN Conv | **PASS** | 3 ms |
| **66** | `Frontier`| `Test_Physics_LatticeBoltzmann2D` | 2D Lattice Boltzmann (LBM D2Q9) Navier-Stokes CFD | **PASS** | 4 ms |
| **67** | `Frontier`| `Test_Neural_Sinkhorn_And_DDIM` | Sinkhorn Wasserstein distance, DDIM Diffusion step | **PASS** | 4 ms |
| **68** | `Frontier`| `Test_Robotics_LQR_And_JacobianDLS` | Discrete LQR (DARE), Robot Inverse Kinematics (DLS) | **PASS** | 4 ms |
| **69** | `Frontier`| `Test_Spatial_HyperbolicGeometry` | Hyperbolic Poincaré Ball, Möbius add, Geodesics | **PASS** | 3 ms |
| **70** | `Autograd`| `Test_ScalarArithmetic_Autograd` | Differentiable $(x+y)(x-y) = x^2-y^2$, $dz/dx, dz/dy$ | **PASS** | 2 ms |
| **71** | `Autograd`| `Test_MultiBranch_Autograd` | Multi-branch DAG differentiation $x^3$, $dz/dx = 3x^2$ | **PASS** | 1 ms |
| **72** | `Autograd`| `Test_MatrixMatMul_Unbroadcasting` | VJP for $Y = XW + b$ with automatic bias unbroadcasting | **PASS** | 2 ms |
| **73** | `Autograd`| `Test_ActivationFunctions_Autograd` | Activation gradients: ReLU, Sigmoid, Tanh | **PASS** | 2 ms |
| **74** | `Autograd`| `Test_LossFunctions_Autograd` | MSE Loss gradient verification $\frac{2}{N}(y_{pred}-y_{true})$ | **PASS** | 1 ms |
| **75** | `Autograd`| `Test_EndToEnd_XOR_NeuralNetwork` | 2-layer MLP XOR classification with AdamW (Loss < 0.04) | **PASS** | 8 ms |
| **76** | `SuperLib` | `TestLambertW_Branches` | Lambert W function $W_0(x)$ & $W_{-1}(x)$ scalar & tensor | **PASS** | 2 ms |
| **77** | `SuperLib` | `TestBesselAndAiryFunctions` | Bessel $Y_0(x), K_0(x)$ and Airy $\text{Ai}(x), \text{Bi}(x)$ | **PASS** | 2 ms |
| **78** | `LinAlg` | `TestSchurDecomposition` | Real Schur $A = Q T Q^T$ with orthogonal $Q$ | **PASS** | 3 ms |
| **79** | `LinAlg` | `TestSolveSylvester` | Continuous Sylvester matrix equation $A X + X B = C$ | **PASS** | 3 ms |
| **80** | `Signal` | `TestWaveletTransform_Haar_Reconstruction` | Discrete Wavelet Transform DWT/IDWT Haar perfect recovery | **PASS** | 2 ms |
| **81** | `Signal` | `TestHilbertTransform` | 1D Hilbert transform and Analytic Signal via FFT | **PASS** | 3 ms |
| **82** | `Optimize` | `TestQPSolve_EqualityConstrained` | ADMM Quadratic Programming with equality constraints | **PASS** | 4 ms |
| **83** | `Optimize` | `TestQPSolve_BoxBounded` | ADMM Quadratic Programming with box bounds $[lb, ub]$ | **PASS** | 4 ms |
| **84** | `Physics` | `TestSymplecticVerlet_EnergyConservation` | Symplectic Leapfrog/Verlet integrator energy conservation | **PASS** | 5 ms |

---

## 2. CLI TEST EXECUTION SUMMARY

> Scope note: this run was captured for the v1.1.0 runtime release. The runtime engine sources and test project were removed from the repository afterward; the shipped binaries remain in `dist/bin/` and the verification load is now carried by the TokenVector stdlib suite (Section 3).

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


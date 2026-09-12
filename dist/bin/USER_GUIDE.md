# TOKENVECTOR.NUMERICS - TECHNICAL HANDBOOK & USER GUIDE
### (Comprehensive Technical Handbook & API Guide for TokenVector.Numerics)

[ 🇬🇧 English ](USER_GUIDE.md) | [ 🇻🇳 Tiếng Việt ](USER_GUIDE_VI.md)

**Document Code:** TKV-NUMERICS-GUIDE-2026-V6 (GRAND UNIFIED EDITION)  
**Target Platform:** C# 12 / .NET 8 LTS / TokenVector Compiler AOT  
**Copyright:** TokenVector Compiler Team & Antigravity AI Team  

---

## TABLE OF CONTENTS
1. [Chapter 1: Multidimensional Array Architecture & Memory Management](#chapter-1-multidimensional-array-architecture--memory-management)
2. [Chapter 2: Broadcasting Engine & Hardware SIMD Optimization](#chapter-2-broadcasting-engine--hardware-simd-optimization)
3. [Chapter 3: Linear Algebra & Matrix Decompositions (SVD, Eigen, EinSum, Kron)](#chapter-3-linear-algebra--matrix-decompositions-svd-eigen-einsum-kron)
4. [Chapter 4: Analytical Matrix Functions (Padé Expm, Sqrtm, Sylvester)](#chapter-4-analytical-matrix-functions-padé-expm-sqrtm-sylvester)
5. [Chapter 5: Astrodynamics & Space Mechanics](#chapter-5-astrodynamics--space-mechanics)
6. [Chapter 6: Quantitative Finance & Option Pricing (FinanceMath)](#chapter-6-quantitative-finance--option-pricing-financemath)
7. [Chapter 7: Time Series Forecasting & Kalman Filtering](#chapter-7-time-series-forecasting--kalman-filtering)
8. [Chapter 8: Computational Physics & Differential Equations (PhysicsODEAndFields)](#chapter-8-computational-physics--differential-equations-physicsodeandfields)
9. [Chapter 9: 3D Spatial Geometry & Point Clouds (Geometry3DAndPointClouds)](#chapter-9-3d-spatial-geometry--point-clouds-geometry3dandpointclouds)
10. [Chapter 10: Signal Processing & DSP Filtering (Bluestein FFT, Windows)](#chapter-10-signal-processing--dsp-filtering-bluestein-fft-windows)
11. [Chapter 11: Special Mathematical Functions (Erf, Gamma, Bessel)](#chapter-11-special-mathematical-functions-erf-gamma-bessel)
12. [Chapter 12: Polynomials, Cumulative Ops, Grids & Sets](#chapter-12-polynomials-cumulative-ops-grids--sets)
13. [Chapter 13: AI, Deep Learning & Transformer Kernels (Neural)](#chapter-13-ai-deep-learning--transformer-kernels-neural)
14. [Chapter 14: Automatic Differentiation & Neural Network Training (Autograd Engine)](#chapter-14-automatic-differentiation--neural-network-training-autograd-engine)

---

## CHAPTER 1: MULTIDIMENSIONAL ARRAY ARCHITECTURE & MEMORY MANAGEMENT

`NDArray<T>` is the core data structure managing unmanaged native memory or GC pinned arrays via `TensorBuffer<T>`, featuring $O(1)$ zero-copy slicing.

```csharp
using TokenVector.Numerics.Core;

// 1. Initialize tensor from flat array
var a = NDArray<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

// 2. Allocate Native Unmanaged Memory (Zero-GC pressure)
using var nativeArr = NDArray<float>.AllocateNative(1024, 1024);

// 3. Zero-Copy Slicing (View)
var slice = a.Slice(Slice.Range(0, 2), Slice.Range(1, 3)); // Shape [2, 2]

// 4. Shape Transformations (Reshape & Permute)
var reshaped = a.Reshape(3, 2);
var permuted = a.Permute(1, 0); // 2D Transpose
```

---

## CHAPTER 2: BROADCASTING ENGINE & HARDWARE SIMD OPTIMIZATION

`TokenVector.Numerics` incorporates standard NumPy right-aligned broadcasting powered by **Stride-0 tricking** and **AVX2/FMA** hardware vectorization.

```csharp
using TokenVector.Numerics.Core;

var mat = NDArray<double>.Zeros(4, 3);
var bias = NDArray<double>.FromArray([10.0, 20.0, 30.0], 1, 3);

// Automatic broadcasting of bias (1, 3) across matrix (4, 3) with Stride-0 trick
var res = mat + bias;

// Hardware SIMD Vector256<double> execution on contiguous memory
var sum = res.Sum();
var mean = res.Mean(axis: 0);
```

---

## CHAPTER 3: LINEAR ALGEBRA & MATRIX DECOMPOSITIONS (SVD, EIGEN, EINSUM, KRON)

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

var A = NDArray<double>.FromArray([4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0], 3, 3);
var b = NDArray<double>.FromArray([7.0, 4.0, 7.0], 3, 1);

// 1. Solve linear system Ax = b with LU Partial Pivoting
var x = Decomposition.Solve(A, b);

// 2. Singular Value Decomposition: A = U * S * V^T
var (U, S, Vt) = SVD.Decompose(A);

// 3. Eigenvalues and Eigenvectors for symmetric matrices (Eigh)
var (eigenValues, eigenVectors) = Eigen.Eigh(A);

// 4. Einstein Summation Contraction
var C = EinSum.Evaluate("ij,jk->ik", A, A);

// 5. Kronecker Product
var K = MatrixOps.Kron(A, NDArray<double>.Eye(2));
```

---

## CHAPTER 4: ANALYTICAL MATRIX FUNCTIONS (PADÉ EXPM, SQRTM, SYLVESTER)

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

// 1. Matrix Exponential e^A via Padé [6/6] with Scaling and Squaring
var a = NDArray<double>.FromArray([0.0, 1.0, -1.0, 0.0], 2, 2);
var expmA = MatrixFunctions.Expm(a); // Rotation matrix cos(1), sin(1)

// 2. Matrix Square Root S = sqrt(A) such that S * S = A
var s = MatrixFunctions.Sqrtm(a);

// 3. Solve Sylvester Matrix Equation: AX + XB = C
var solX = MatrixFunctions.SolveSylvester(A, B, C);
```

---

## CHAPTER 5: ASTRODYNAMICS & SPACE MECHANICS

```csharp
using TokenVector.Numerics.Science;

// 1. Solve Kepler's Equation: M = E - e*sin(E)
double e = 0.05; // Eccentricity
double M = 1.25; // Mean anomaly (rad)
double E = Astrodynamics.SolveKepler(M, e);

// 2. Convert 6 Keplerian orbital elements to 3D Cartesian position (r) and velocity (v) in ECI
var (r, v) = Astrodynamics.KeplerianToCartesian(
    a: 7000.0, e: 0.01, i: 0.9, raan: 1.2, omega: 0.5, nu: 0.8
);

// 3. Calculate Hohmann Orbit Transfer (LEO to GEO)
var (dv1, dv2, totalDv, tof) = Astrodynamics.HohmannTransfer(6678.137, 42164.0);

// 4. Convert WGS84 Geodetic Coordinates to ECEF Cartesian
var ecef = Astrodynamics.GeodeticToEcef(latDeg: 21.0285, lonDeg: 105.8542, altKm: 0.02);
```

---

## CHAPTER 6: QUANTITATIVE FINANCE & OPTION PRICING (FINANCEMATH)

```csharp
using TokenVector.Numerics.Finance;

// 1. Black-Scholes-Merton Option Pricing & The Greeks
double callPrice = FinanceMath.BlackScholesCall(s: 100.0, k: 100.0, t: 1.0, r: 0.05, sigma: 0.20);
double putPrice = FinanceMath.BlackScholesPut(s: 100.0, k: 100.0, t: 1.0, r: 0.05, sigma: 0.20);

var (delta, gamma, vega, theta, rho) = FinanceMath.OptionGreeks(100.0, 100.0, 1.0, 0.05, 0.20);

// 2. Markowitz Modern Portfolio Theory & Sharpe Ratio
double expReturn = FinanceMath.PortfolioReturn(weights, assetReturns);
double expVol = FinanceMath.PortfolioVolatility(weights, covMatrix);
double sharpe = FinanceMath.SharpeRatio(weights, assetReturns, covMatrix, riskFreeRate: 0.02);

// 3. Discounted Cash Flows: NPV and IRR
double npv = FinanceMath.NPV(0.08, cashFlows);
double irr = FinanceMath.IRR(cashFlows);
```

---

## CHAPTER 7: TIME SERIES FORECASTING & KALMAN FILTERING

```csharp
using TokenVector.Numerics.Statistics;

// 1. 1D Scalar Kalman Filter
var kf = new TimeSeriesAndKalman.KalmanFilter1D(initialState: 0.0, initialVariance: 1.0, processNoise: 0.01, measurementNoise: 0.1);
kf.Predict();
double filteredState = kf.Update(measuredValue);

// 2. Multidimensional Kalman Filter (ND State Space)
var kfNd = new TimeSeriesAndKalman.KalmanFilterND(x0, P0, F, H, Q, R);
kfNd.Predict();
var state = kfNd.Update(measurement);

// 3. Holt Linear Trend Forecasting
var (fitted, forecast) = TimeSeriesAndKalman.HoltLinearTrend(series, alpha: 0.8, beta: 0.2, forecastSteps: 5);
```

---

## CHAPTER 8: COMPUTATIONAL PHYSICS & DIFFERENTIAL EQUATIONS (PHYSICSODEANDFIELDS)

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Physics;

// 1. Integrate Ordinary Differential Equations via 4th-order Runge-Kutta (RK4)
Func<double, NDArray<double>, NDArray<double>> harmonicOscillator = (t, y) =>
    NDArray<double>.FromArray([y[1], -y[0]], 2);

var (times, trajectory) = PhysicsODEAndFields.SolveRK4(harmonicOscillator, 0.0, 10.0, y0, numSteps: 200);

// 2. Gravitational N-Body dynamics via Symplectic Velocity Verlet
var (newPos, newVel) = PhysicsODEAndFields.NBodyVerletStep(positions, velocities, masses, dt: 0.01);

// 3. 3D Vector Differential Operators
var (gx, gy, gz) = PhysicsODEAndFields.Gradient3D(scalarField);
var div = PhysicsODEAndFields.Divergence3D(fx, fy, fz);
var (cx, cy, cz) = PhysicsODEAndFields.Curl3D(fx, fy, fz);
var laplacian = PhysicsODEAndFields.Laplacian3D(scalarField);
```

---

## CHAPTER 9: 3D SPATIAL GEOMETRY & POINT CLOUDS (GEOMETRY3DANDPOINTCLOUDS)

```csharp
using TokenVector.Numerics.Spatial;

// 1. Point Cloud Alignment & Rigid Registration (Kabsch / ICP)
var (rotationMatrix, translationVec) = Geometry3DAndPointClouds.AlignPointCloudsKabsch(sourceCloud, targetCloud);

// 2. Möller-Trumbore Ray-Triangle Intersection (Ray Tracing)
var (hit, dist, u, v) = Geometry3DAndPointClouds.RayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2);

// 3. Point-to-Plane Distance
double distPlane = Geometry3DAndPointClouds.PointToPlaneDistance(point, planePt, planeNormal);

// 4. 4x4 Affine Transforms and Quaternions
var transform = Affine3D.LookAt(eye, target, up);
var q = Quaternion<double>.FromAxisAngle(axis, angleRad);
```

---

## CHAPTER 10: SIGNAL PROCESSING & DSP FILTERING (BLUESTEIN FFT, WINDOWS)

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

// 1. Fast Fourier Transform (Bluestein Chirp-Z FFT) for arbitrary prime length N = 1009
var signal = NDArray<double>.FromArray(rawData, 1009);
var spectrum = FFT.FFT1D(signal); // Complex NDArray

// 2. DSP Windowing Functions (Blackman, Hanning, Hamming)
var win = SignalProcessing.Blackman(1024);
var filtered = SignalProcessing.Convolve(signal, win, mode: "same");
```

---

## CHAPTER 11: SPECIAL MATHEMATICAL FUNCTIONS (ERF, GAMMA, BESSEL)

```csharp
using TokenVector.Numerics.LinAlg;

double erfVal = SpecialFunctions.Erf(1.5);
double gammaVal = SpecialFunctions.Gamma(5.0); // 4! = 24.0
double logGamma = SpecialFunctions.LogGamma(10.0);
double digamma = SpecialFunctions.Digamma(2.5);
double besselJ0 = SpecialFunctions.BesselJ0(2.4048); // ~0.0 (First zero)
```

---

## CHAPTER 12: POLYNOMIALS, CUMULATIVE OPS, GRIDS & SETS

```csharp
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;
using TokenVector.Numerics.Ops;

// 1. Polynomial Fitting (PolyFit) and Root Finding (Roots)
var x = NDArray<double>.FromArray([0, 1, 2, 3], 4);
var y = NDArray<double>.FromArray([1, 3, 7, 13], 4);
var coeffs = Polynomial.PolyFit(x, y, degree: 2);
var roots = Polynomial.Roots(coeffs);

// 2. Cumulative Operations (CumSum) and Discrete Differences (Diff)
var cum = x.CumSum();
var d = CumulativeOps.Diff(y, n: 1);

// 3. Coordinate Grids (Meshgrid) and Set Operations
var (XGrid, YGrid) = GridOps.Meshgrid(x, y);
var common = SetOperations.Intersect1D(arr1, arr2);
```

---

## CHAPTER 13: AI, DEEP LEARNING & TRANSFORMER KERNELS (NEURAL)

```csharp
using TokenVector.Numerics.Neural;

// 1. Rotary Positional Embedding (RoPE) for LLMs (Llama 3 / Mistral)
var ropeQ = AttentionEngine.ApplyRoPE(qTensor, cosCache, sinCache);

// 2. Scaled Dot-Product Attention (FlashAttention compatible)
var attnOut = AttentionEngine.ScaledDotProductAttention(Q, K, V, mask: causalMask);

// 3. RMSNorm & Sinkhorn Optimal Transport
var normed = Normalization.RMSNorm(hiddenStates, weightGamma);
var distW = OptimalTransportAndDiffusion.Sinkhorn(sourceDist, targetDist, costMatrix, reg: 0.1);
```

---

## CHAPTER 14: AUTOMATIC DIFFERENTIATION & NEURAL NETWORK TRAINING (AUTOGRAD ENGINE)

The **Autograd Engine** constructs a Dynamic Directed Acyclic Graph (DAG) and calculates exact gradients via Reverse-Mode Backpropagation.

```csharp
using TokenVector.Numerics.Autograd;
using TokenVector.Numerics.Autograd.NN;
using TokenVector.Numerics.Autograd.Optim;
using TokenVector.Numerics.Autograd.Nodes;

// 1. Scalar Arithmetic Autograd
var x = new Tensor<double>(3.0, requiresGrad: true);
var y = new Tensor<double>(2.0, requiresGrad: true);
var z = (x + y) * (x - y); // z = x^2 - y^2
z.Backward();

Console.WriteLine($"dz/dx = {x.Grad!.Buffer[0]}"); // 6.0 (2*x)
Console.WriteLine($"dz/dy = {y.Grad!.Buffer[0]}"); // -4.0 (-2*y)

// 2. Matrix Multiplication & Automatic Unbroadcasting
var X = new Tensor<double>(new double[] { 1, 2, 3, 4 }, new[] { 2, 2 }, requiresGrad: true);
var W = new Tensor<double>(new double[] { 0.5, -0.5, 1.0, 2.0 }, new[] { 2, 2 }, requiresGrad: true);
var b = new Tensor<double>(new double[] { 0.1, 0.2 }, new[] { 1, 2 }, requiresGrad: true);

var Y = X.MatMul(W) + b;
var loss = Y.Sum();
loss.Backward(); // Automatically unbroadcasts bias gradient to [1, 2]

// 3. Train a Multi-Layer Perceptron (MLP) with AdamW
var inputs = new Tensor<double>(new double[] { 0,0, 0,1, 1,0, 1,1 }, new[] { 4, 2 });
var targets = new Tensor<double>(new double[] { 0, 1, 1, 0 }, new[] { 4, 1 });

var l1 = new Linear<double>(inFeatures: 2, outFeatures: 8);
var l2 = new Linear<double>(inFeatures: 8, outFeatures: 1);
var model = new Sequential<double>(l1, l2);
var optimizer = new AdamW<double>(model.Parameters(), lr: 0.1);

for (int epoch = 0; epoch < 200; epoch++)
{
    optimizer.ZeroGrad();
    var h = l1.Forward(inputs).Tanh();
    var preds = l2.Forward(h).Sigmoid();
    var mse = ActivationAndReductionNodes<double>.MSELoss(preds, targets);
    
    mse.Backward();
    optimizer.Step();
}
```

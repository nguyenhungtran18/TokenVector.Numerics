# 🚀 TokenVector.Numerics v1.0.0 Release Notes

[🇻🇳 Xem bản Tiếng Việt](RELEASE_NOTES_VI.md)

---

**Release Version:** `v1.0.0`  
**Release Date:** September 13, 2026  
**Target Framework:** .NET 8.0 LTS (C# 12)  
**License:** [MIT License](LICENSE)  
**Repository:** [https://github.com/nguyenhungtran18/TokenVector.Numerics](https://github.com/nguyenhungtran18/TokenVector.Numerics)  
**NuGet Package:** `TokenVector.Numerics` (v1.0.0)

---

## 🌟 Overview

We are thrilled to announce the official **v1.0.0 production release** of **TokenVector.Numerics** — an enterprise-grade, hardware-accelerated multidimensional tensor, scientific computing, and reverse-mode automatic differentiation (Autograd) engine written entirely from scratch in C# 12 and .NET 8 LTS with zero third-party dependencies.

`TokenVector.Numerics` is designed as the high-performance runtime core for the **TokenVector** compiled programming language and as a standalone high-throughput numerical foundation for AI, robotics, and scientific simulations in .NET.

---

## 🚀 Key Highlights & Major Features

### 1. Dynamic Reverse-Mode Autograd Engine
* **Full DAG Computational Graph:** Dynamic reverse-mode graph construction with topological sorting backward execution.
* **Automatic Gradient Unbroadcasting:** Automatically performs multi-dimensional sum-reductions during backpropagation across broadcasted axes, guaranteeing gradient-to-weight shape consistency.
* **Tensor Lifecycle Management:** Supports `.backward()`, `.zero_grad()`, `.detach()`, and in-place gradient accumulation.
* **Operator Overloading:** Full operator overloading (`+`, `-`, `*`, `/`, `^`, unary `-`, `@` matrix multiplication).

### 2. Neural Modules & Advanced Optimizers
* **Layer Architectures:** `Linear<T>` (with Kaiming Uniform weight initialization), `Sequential<T>`, and `RMSNorm<T>` (Root Mean Square Normalization).
* **Activation Functions & Loss Kernels:** Vectorized Vector-Jacobian Products (VJP) for `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`, `MSELoss`, and numerically stable Log-Sum-Exp `CrossEntropyLoss`.
* **First-Class Optimizers:**
  * **SGD:** Stochastic Gradient Descent with classical Momentum and $L_2$ Weight Decay.
  * **AdamW:** Adaptive Moment Estimation with decoupled weight decay and complete first/second moment bias correction ($m_t, v_t$).

### 3. Hardware SIMD & Zero-GC Memory Architecture
* **Unmanaged Memory Management:** Employs `NativeMemory.AllocZeroed` and pointer arithmetic to guarantee zero-GC allocations during high-frequency matrix operations.
* **SIMD Intrinsics:** Accelerated via `Vector256<float>`, `Vector256<double>`, AVX2, and FMA hardware instructions for maximum multi-core CPU throughput.

### 4. 34 Comprehensive Scientific Domains
* **Linear Algebra (`LinAlg`):** Fast Matrix Multiplication ($O(N^3)$ optimized), Singular Value Decomposition (SVD), Eigenvalue/Eigenvector solver, Cholesky Decomposition, and Padé Approximation Matrix Exponential ($\exp(A)$).
* **Signal Processing:** 1D/2D Fast Fourier Transform (Cooley-Tukey Radix-2 FFT) and Discrete Cosine Transform (DCT).
* **Astrodynamics & Aerospace:** Keplerian orbital element propagation, Lambert problem solver, and orbital state vector conversions.
* **Robotics & Dynamics:** 6-DoF Quadrotor rigid-body dynamics, aerodynamic drag force calculation, and Quaternion 3D spatial rotations.
* **Financial Engineering:** Black-Scholes European option pricing, Greeks sensitivity calculation ($\Delta, \Gamma, \Theta, \text{Vega}, \rho$), and Monte Carlo path simulations.
* **Quantum Computing:** Qubit statevector simulators and quantum logic gates ($H$, $X$, $Y$, $Z$, $\text{CNOT}$, Phase).

---

## 📊 Verification & Test Metrics

The entire release has been validated through an extensive unit test suite with 100% pass rate:

* **Total Tests:** **75 / 75 Tests Passed**
* **Failures:** **0**
* **Execution Time:** **~146 ms**
* **Verification Suite:** `TokenVector.Numerics.Tests` (xUnit, Release x64)
* **Full Test Matrix:** See [`TEST_REPORT.md`](TEST_REPORT.md).

---

## 📦 Distribution Packages (`dist/`)

The compiled release artifacts are available in the `dist/` directory:

| Artifact | Path | Size | Description |
| :--- | :--- | :---: | :--- |
| **Release Zip** | `dist/TokenVector.Numerics-v1.0.0-Release.zip` | `174 KB` | Complete standalone release bundle |
| **NuGet Package** | `dist/nuget/TokenVector.Numerics.1.0.0.nupkg` | `123 KB` | Official NuGet package |
| **Binary DLL** | `dist/bin/TokenVector.Numerics.dll` | `234 KB` | Optimized standalone .NET 8 assembly |

---

## 💻 Quick Code Example (C# / XOR Neural Convergence)

```csharp
using TokenVector.Numerics.Autograd;
using TokenVector.Numerics.Autograd.NN;
using TokenVector.Numerics.Autograd.Optim;

// 1. Build a 2-layer MLP (2 -> 8 -> 1)
var model = new Sequential<float>(
    new Linear<float>(2, 8),
    new Linear<float>(8, 1)
);
var optimizer = new AdamW<float>(model.Parameters(), lr: 0.05f);

// 2. Training Data (XOR)
var x = Tensor<float>.From(new float[,] { {0,0}, {0,1}, {1,0}, {1,1} }, requiresGrad: false);
var y = Tensor<float>.From(new float[,] { {0}, {1}, {1}, {0} }, requiresGrad: false);

// 3. Training Loop
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

## 🤝 Acknowledgments & Contributing

We would like to express our gratitude to the contributors and the scientific computing community. Feedback, bug reports, and contributions are welcome on our [GitHub Repository](https://github.com/nguyenhungtran18/TokenVector.Numerics).

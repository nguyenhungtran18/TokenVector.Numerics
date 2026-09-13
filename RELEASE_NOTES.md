# 🚀 TokenVector.Numerics v1.0.1 Release Notes

[🇻🇳 Xem bản Tiếng Việt](RELEASE_NOTES_VI.md)

---

**Release Version:** `v1.0.1`  
**Release Date:** September 13, 2026  
**Target Framework:** .NET 8.0 LTS (C# 12)  
**License:** [MIT License](LICENSE)  
**Repository:** [https://github.com/nguyenhungtran18/TokenVector.Numerics](https://github.com/nguyenhungtran18/TokenVector.Numerics)  
**NuGet Package:** `TokenVector.Numerics` (v1.0.1)

---

## 🌟 Overview & What's New in v1.0.1

The **v1.0.1 release** introduces critical high-order mathematical, optimization, and scientific signal processing capabilities to `TokenVector.Numerics`. This update expands linear algebra decomposition, advanced special transcendental functions, discrete wavelet analysis, quadratic programming solvers for drone/robotics control, and symplectic Hamiltonian integrators.

---

## 🚀 Key Additions in v1.0.1

### 1. Advanced Special Functions (`SpecialFunctions`)
* **Lambert W Function (`LambertW`):** Computes $W_k(x)$ where $W(x) e^{W(x)} = x$ using Halley's 3rd-order root-finding method. Supports both principal branch $k=0$ (for $x \ge -1/e$) and secondary branch $k=-1$ (for $-1/e \le x < 0$) with full scalar and tensor support.
* **Bessel Functions of the Second Kind (`BesselY0`, `BesselK0`):**
  * $Y_0(x)$ (Neumann function of order 0).
  * $K_0(x)$ (Modified Bessel function of the second kind of order 0).
* **Airy Functions (`AiryAi`, `AiryBi`):** Computes solutions to the differential equation $y'' - x y = 0$ using Maclaurin series around zero and asymptotic expansions for large $|x|$.

### 2. Matrix Decompositions & Matrix Equations (`LinAlg`)
* **Real Schur Decomposition (`Decomposition.Schur`):** Decomposes square matrix $A = Q T Q^T$, where $Q$ is an orthogonal matrix and $T$ is quasi-upper triangular, via Hessenberg reduction and shifted QR iteration.
* **Sylvester Equation Solver (`Decomposition.SolveSylvester`):** Solves the continuous Sylvester matrix equation $A X + X B = C$ (and Lyapunov equations when $B = A^T$) using Kronecker vectorization.

### 3. Wavelet & Analytic Signal Processing (`SignalProcessing`)
* **Discrete Wavelet Transform (`DWT` & `IDWT`):** 1D single-level forward and inverse wavelet transform supporting Haar and Daubechies-4 (`db4`) filter banks with exact signal reconstruction.
* **Hilbert Transform & Analytic Signal (`Hilbert`, `AnalyticSignal`):** Computes the analytic signal $x_a(t) = x(t) + i \mathcal{H}[x(t)]$ in $O(N \log N)$ via FFT.

### 4. Quadratic Programming Solver (`Optimize.QPSolve`)
* **Convex QP Solver:** Solves general convex quadratic programs:
  $$\min_x \frac{1}{2} x^T P x + q^T x \quad \text{subject to} \quad G x \le h, \quad A x = b, \quad lb \le x \le ub$$
* **ADMM Operator-Splitting Engine:** Matrix factorization with proximal operator projections, optimized for real-time model predictive control (NMPC), robotics trajectory optimization, and drone swarms.

### 5. Symplectic Hamiltonian Integrator (`PhysicsODEAndFields.SolveSymplecticVerlet`)
* **Energy-Preserving Symplectic Leapfrog / Velocity-Verlet:** Integrates Hamiltonian dynamics $H(q, p) = \frac{1}{2m} p^T p + V(q)$ preserving phase-space volume and total energy over long trajectories without secular energy drift.

---

## 📊 Verification & Test Metrics

* **Total Tests:** **84 / 84 Tests Passed (100% Pass)**
* **Failures:** **0**
* **Execution Time:** **~138 ms**
* **Verification Suite:** `TokenVector.Numerics.Tests` (xUnit, Release x64)

---

## 📦 Distribution Packages (`dist/`)

| Artifact | Path | Description |
| :--- | :--- | :--- |
| **Release Zip** | `dist/TokenVector.Numerics-v1.0.1-Release.zip` | Standalone v1.0.1 release bundle |
| **NuGet Package** | `dist/nuget/TokenVector.Numerics.1.0.1.nupkg` | Official v1.0.1 NuGet package |
| **Binary DLL** | `dist/bin/TokenVector.Numerics.dll` | Optimized standalone .NET 8 assembly |

---

## 🤝 Contributing & Community

Contributions, issue reports, and discussions are welcome on our [GitHub Repository](https://github.com/nguyenhungtran18/TokenVector.Numerics).

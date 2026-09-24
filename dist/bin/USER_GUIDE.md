# TOKENVECTOR.NUMERICS - TECHNICAL HANDBOOK & USER GUIDE
### (Comprehensive Technical Handbook & API Guide for TokenVector.Numerics)

[ 🇬🇧 English ](USER_GUIDE.md) | [ 🇻🇳 Tiếng Việt ](USER_GUIDE_VI.md)

**Document Code:** TKV-NUMERICS-GUIDE-2026-V6 (GRAND UNIFIED EDITION)  
**Target Platform:** .NET 8 LTS / TokenVector Compiler AOT  
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
15. [Chapter 15: TokenVector Standard Library (.tkv) — 100% Numeric Surface Parity](#chapter-15-tokenvector-standard-library-tkv--100-numeric-surface-parity)

---

## CHAPTER 1: MULTIDIMENSIONAL ARRAY ARCHITECTURE & MEMORY MANAGEMENT

`NDArray<T>` is the core data structure managing unmanaged native memory or GC pinned arrays via `TensorBuffer<T>`, featuring $O(1)$ zero-copy slicing.

```tkv
import tv
from tv.core import from_array

# 1. Initialize tensor from flat array
a = from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

# 2. Allocate native unmanaged memory (zero-GC pressure)
native_arr = tv.core.allocate_native([1024, 1024], dtype=tv.f32)

# 3. Zero-copy slicing (view)
view = a.slice([[0, 2, 1], [1, 3, 1]])  # Shape [2, 2]

# 4. Shape transformations (Reshape & Permute)
reshaped = a.reshape([3, 2])
permuted = a.permute([1, 0])  # 2D transpose
```

---

## CHAPTER 2: BROADCASTING ENGINE & HARDWARE SIMD OPTIMIZATION

`TokenVector.Numerics` incorporates right-aligned broadcasting powered by **Stride-0 tricking** and **AVX2/FMA** hardware vectorization.

```tkv
import tv
from tv.core import from_array, zeros

mat = zeros([4, 3])
bias = from_array([10.0, 20.0, 30.0], [1, 3])

# Automatic broadcasting of bias (1, 3) across matrix (4, 3) with the Stride-0 trick
res = tv.ops.add(mat, bias)

# SIMD/AVX2 kernels execute on contiguous memory
total = tv.ops.sum(res)
avg = tv.ops.mean_axis(res, 0)
```

---

## CHAPTER 3: LINEAR ALGEBRA & MATRIX DECOMPOSITIONS (SVD, EIGEN, EINSUM, KRON)

```tkv
import tv
from tv.core import from_array
from tv import linalg

A = from_array([4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0], [3, 3])
b = from_array([7.0, 4.0, 7.0], [3, 1])

# 1. Solve the linear system Ax = b with LU partial pivoting
x = linalg.solve(A, b)

# 2. Singular Value Decomposition: A = U * S * V^T
U, S, Vt = linalg.svd(A)

# 3. Eigenvalues and eigenvectors for symmetric matrices (Eigh)
eigen_values, eigen_vectors = linalg.eigh(A)

# 4. Einstein summation contraction
C = linalg.einsum("ij,jk->ik", [A, A])

# 5. Kronecker product
K = linalg.kron(A, tv.core.eye(2))
```

---

## CHAPTER 4: ANALYTICAL MATRIX FUNCTIONS (PADÉ EXPM, SQRTM, SYLVESTER)

```tkv
import tv
from tv.core import from_array
import tv.linalg_functions as lf

# 1. Matrix exponential e^A via Padé [6/6] with scaling and squaring
a = from_array([0.0, 1.0, -1.0, 0.0], [2, 2])
expm_a = lf.expm(a)  # Rotation matrix cos(1), sin(1)

# 2. Matrix square root S = sqrt(A) such that S * S = A
s = lf.sqrtm(a)

# 3. Solve the Sylvester matrix equation: AX + XB = C
sol_x = lf.solve_sylvester_cm(A, B, C)
```

---

## CHAPTER 5: ASTRODYNAMICS & SPACE MECHANICS

```tkv
import tv
import tv.astro

# 1. Solve Kepler's equation: M = E - e*sin(E)
e = 0.05      # Eccentricity
m_anom = 1.25 # Mean anomaly (rad)
ecc_anom = tv.astro.solve_kepler(m_anom, e)

# 2. Convert 6 Keplerian orbital elements to 3D Cartesian position (r) and velocity (v) in ECI
r_vec, v_vec = tv.astro.keplerian_to_cartesian(7000.0, 0.01, 0.9, 1.2, 0.5, 0.8)

# 3. Hohmann orbit transfer (LEO to GEO)
dv1, dv2, total_dv, tof = tv.astro.hohmann_transfer(6678.137, 42164.0)

# 4. Convert WGS84 geodetic coordinates to ECEF Cartesian
ecef = tv.astro.geodetic_to_ecef(21.0285, 105.8542, 0.02)
```

---

## CHAPTER 6: QUANTITATIVE FINANCE & OPTION PRICING (FINANCEMATH)

```tkv
import tv.finance

# 1. Black-Scholes-Merton option pricing & the Greeks
call_price = tv.finance.black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.20)
put_price = tv.finance.black_scholes_put(100.0, 100.0, 1.0, 0.05, 0.20)

delta, gamma, vega, theta, rho = tv.finance.option_greeks(100.0, 100.0, 1.0, 0.05, 0.20)

# 2. Markowitz modern portfolio theory & Sharpe ratio
exp_return = tv.finance.portfolio_return(weights, asset_returns)
exp_vol = tv.finance.portfolio_volatility(weights, cov_matrix)
sharpe = tv.finance.sharpe_ratio(weights, asset_returns, cov_matrix, 0.02)

# 3. Discounted cash flows: NPV and IRR
npv_val = tv.finance.npv(0.08, cash_flows)
irr_val = tv.finance.irr(cash_flows)
```

---

## CHAPTER 7: TIME SERIES FORECASTING & KALMAN FILTERING

```tkv
import tv.core
import tv.statistics as stats

# 1. 1D scalar Kalman filter
kf = stats.KalmanFilter1D(0.0, 1.0, 0.01, 0.1)
kf.predict()
filtered_state = kf.update(measured_value)

# 2. Multidimensional Kalman filter (ND state space)
kf_nd = stats.KalmanFilterND(x0, p0, f_mat, h_mat, q_mat, r_mat)
kf_nd.predict()
state = kf_nd.update(measurement)

# 3. Holt linear trend forecasting
fitted, forecast = stats.holt_linear_trend(series, 0.8, 0.2, 5)
```

---

## CHAPTER 8: COMPUTATIONAL PHYSICS & DIFFERENTIAL EQUATIONS (PHYSICSODEANDFIELDS)

```tkv
import tv.core
import tv.physics

# 1. Integrate ordinary differential equations via 4th-order Runge-Kutta (RK4)
def harmonic_oscillator(t, y):
    return tv.core.from_array([y.get([1]), -y.get([0])], [2])

times, trajectory = tv.physics.solve_rk4(harmonic_oscillator, 0.0, 10.0, y0, 200)

# 2. Gravitational N-body dynamics via symplectic velocity Verlet
new_pos, new_vel = tv.physics.nbody_verlet_step(positions, velocities, masses, 0.01)

# 3. 3D vector differential operators
gx, gy, gz = tv.physics.gradient_3d(scalar_field)
div = tv.physics.divergence_3d(fx, fy, fz)
cx, cy, cz = tv.physics.curl_3d(fx, fy, fz)
lap = tv.physics.laplacian_3d(scalar_field)
```

---

## CHAPTER 9: 3D SPATIAL GEOMETRY & POINT CLOUDS (GEOMETRY3DANDPOINTCLOUDS)

```tkv
import tv.core
import tv.geometry3d
import tv.spatial

# 1. Point cloud alignment & rigid registration (Kabsch)
rotation_matrix, translation_vec = tv.geometry3d.align_point_clouds_kabsch(source_cloud, target_cloud)

# 2. Möller-Trumbore ray-triangle intersection (ray tracing)
has_hit, dist, u_coord, v_coord = tv.geometry3d.ray_triangle_intersect(ray_origin, ray_dir, v0, v1, v2)

# 3. Point-to-plane distance
dist_plane = tv.geometry3d.point_to_plane_distance(point, plane_pt, plane_normal)

# 4. Quaternions & 4x4 transforms
q = tv.spatial.quaternion_from_axis_angle(ax, ay, az, angle_rad)
transform = tv.spatial.quaternion_to_rotation_matrix_4x4(q)
```

---

## CHAPTER 10: SIGNAL PROCESSING & DSP FILTERING (BLUESTEIN FFT, WINDOWS)

```tkv
import tv.fft
import tv.signal

# 1. Fast Fourier transform (Bluestein chirp-Z FFT) for arbitrary prime length N = 1009
spectrum = tv.fft.fft1d(raw_data)  # list of Complex in / out

# 2. DSP windowing functions (Blackman, Hanning, Hamming)
win = tv.signal.blackman(1024)
filtered = tv.signal.convolve(signal_t, win, "same")
```

---

## CHAPTER 11: SPECIAL MATHEMATICAL FUNCTIONS (ERF, GAMMA, BESSEL)

```tkv
import tv.special

erf_val = tv.special.erf_scalar(1.5)
gamma_val = tv.special.gamma_scalar(5.0)      # 4! = 24.0
log_gamma_val = tv.special.log_gamma_scalar(10.0)
digamma_val = tv.special.digamma(2.5)
bessel_j0_val = tv.special.bessel_j0_scalar(2.4048)  # ~0.0 (first zero)
```

---

## CHAPTER 12: POLYNOMIALS, CUMULATIVE OPS, GRIDS & SETS

```tkv
import tv.core
from tv.core import from_array
import tv.compare
import tv.grid
import tv.poly

# 1. Polynomial fitting (poly_fit) and root finding (roots)
x = from_array([0, 1, 2, 3], [4])
y = from_array([1, 3, 7, 13], [4])
coeffs = tv.poly.poly_fit(x, y, 2)
roots = tv.poly.roots(coeffs)

# 2. Cumulative operations (cumsum) and discrete differences (diff)
cum = tv.grid.cumsum(x)
d = tv.grid.diff(y)

# 3. Coordinate grids (meshgrid) and set operations
x_grid, y_grid = tv.grid.meshgrid([x, y])
common = tv.compare.intersect1d(arr1, arr2)
```

---

## CHAPTER 13: AI, DEEP LEARNING & TRANSFORMER KERNELS (NEURAL)

```tkv
import tv.core
import tv.neural

# 1. Rotary positional embedding (RoPE) for LLMs (Llama 3 / Mistral)
rope_q = tv.neural.apply_rope(q_tensor, 0, 10000.0)

# 2. Scaled dot-product attention (FlashAttention compatible)
attn_out, attn_weights = tv.neural.scaled_dot_product_attention(q, k, v, causal_mask)

# 3. RMSNorm & Sinkhorn optimal transport
normed = tv.neural.rms_norm(hidden_states, weight_gamma)
dist_w, plan = tv.neural.sinkhorn(source_dist, target_dist, cost_matrix, 0.1)
```

---

## CHAPTER 14: AUTOMATIC DIFFERENTIATION & NEURAL NETWORK TRAINING (AUTOGRAD ENGINE)

The **Autograd Engine** constructs a Dynamic Directed Acyclic Graph (DAG) and calculates exact gradients via Reverse-Mode Backpropagation.

```tkv
import tv.core
from tv.core import from_array
import tv.autograd as ag

# 1. Scalar arithmetic autograd
x = ag.Tensor.full(3.0, [1])
x.requires_grad = True
y = ag.Tensor.full(2.0, [1])
y.requires_grad = True
z = (x + y) * (x - y)  # z = x^2 - y^2
z.backward()

print(x.grad.get([0]))  # dz/dx = 6.0 (2*x)
print(y.grad.get([0]))  # dz/dy = -4.0 (-2*y)

# 2. Matrix multiplication & automatic unbroadcasting
x_mat = ag.Tensor.from_ndarray(from_array([1, 2, 3, 4], [2, 2]), requires_grad=True)
w_mat = ag.Tensor.from_ndarray(from_array([0.5, -0.5, 1.0, 2.0], [2, 2]), requires_grad=True)
b_vec = ag.Tensor.from_ndarray(from_array([0.1, 0.2], [1, 2]), requires_grad=True)

y_mat = x_mat.matmul(w_mat) + b_vec
loss = y_mat.sum()
loss.backward()  # Automatically unbroadcasts the bias gradient to [1, 2]

# 3. Train a multi-layer perceptron (MLP) with AdamW
inputs = ag.Tensor.from_ndarray(from_array([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]))
targets = ag.Tensor.from_ndarray(from_array([0, 1, 1, 0], [4, 1]))

model = ag.Sequential([ag.Linear(2, 8), ag.Linear(8, 1)])
optimizer = ag.AdamW(model.parameters(), lr=0.1)

epoch = 0
while epoch < 200:
    optimizer.zero_grad()
    h = ag.tanh(model.modules[0].forward(inputs))
    preds = ag.sigmoid(model.modules[1].forward(h))
    mse = ag.mse_loss(preds, targets)
    mse.backward()
    optimizer.step()
    epoch = epoch + 1
```

---

## CHAPTER 15: TOKENVECTOR STANDARD LIBRARY (.TKV) — 100% NUMERIC SURFACE PARITY

*New in v1.1.0.* The entire library is now also published as a **TokenVector language** standard library: 27 `.tkv` modules (~12.1k lines) under `src/tokenvector/`, built per the `TKV-SPEC-SYNTAX-2026-V1` grammar (TV-1001). Every module carries a "Source of truth" header mapping each source section to its `.tkv` function.

### 15.1 Module Map

| Group | Modules |
| :--- | :--- |
| Core tensor engine | `core` (NDArray, TensorBuffer, BoolNDArray, shape/broadcast helpers), `engine` (Stride-0 broadcast, SIMD kernels), `io` (.npy/.npz/raw/CSV, MemoryMappedNDArray) |
| Math surface | `ops`, `manipulation`, `grid`, `compare`, `linalg`, `linalg_functions`, `poly`, `special`, `fft`, `signal`, `random`, `statistics`, `optimize`, `interpolation` |
| AI | `autograd` (Tensor, reverse-mode AD, SGD/AdamW, Module/Linear/Sequential/RMSNorm), `neural` (activations, attention, conv2d, Sinkhorn, DDIM) |
| Domains | `quantum`, `physics`, `astro`, `finance`, `spatial`, `geometry3d`, `crypto_graph`, `biology_robotics` |

### 15.2 Numeric Surface Coverage — 100%

The audit tool cross-checks the full numeric-library public surface (595 names audited) and matches each library function against the `.tkv` stdlib (module functions + methods verified on the real `NDArray`/`BoolNDArray`/`Tensor` classes):

| Bucket | Count |
| :--- | ---: |
| Language-layer N/A (dtype objects, constants, RNG machinery, packaging) | 241 |
| **Library surface audited** | **354** |
| **Matched in .tkv stdlib** | **354 (100%)** |
| Truly missing | **0** |

The 241 language-layer names are handled by the TokenVector language/compiler itself (`tv.f64`, runtime constants, `tkvc`) by design — not by the library.

### 15.3 Verification Toolchain

```powershell
# 1. Smoke suite: 175 checks across all 27 modules
python tests/tokenvector/tkv_harness.py
#    Syntax gate: all .tkv modules parse, TV-1001 constructs only.
#    TokenVector stdlib smoke tests: passed=175, failed=0

# 2. Coverage audit (reproducible, prints per-bucket detail)
python tests/tokenvector/numpy_coverage_audit.py

# 3. Numeric parity + performance benchmark
python tests/tokenvector/benchmark_vs_numpy.py
#    matmul / broadcast add: max|diff| = 0.0, SVD: 1.1e-14, FFT: ~5e-12

# Or compile natively with tkvc (compiler repo):
./tkvc.exe tests/tokenvector/smoke_tests.tkv -r src/tokenvector -o smoke.exe
```

### 15.4 Quick Start in TokenVector

```tkv
import tv
from tv.core import from_array
from tv import linalg
import tv.autograd as ag

# Math with verified parity (see benchmark)
A = from_array([4.0, 1.0, 1.0, 3.0], [2, 2])
U, S, Vt = linalg.svd(A)

# Fancy indexing (standard semantics)
mask = tv.compare.greater_than(A, tv.core.full(2.0, [2, 2]))
picked = tv.grid.boolean_select(A, mask)      # arr[mask]
tv.grid.boolean_assign(A, mask, 0.0)          # arr[mask] = 0.0

# Autograd built into the stdlib
x = ag.Tensor.full(3.0, [1])
x.requires_grad = True
y = x * x
y.backward()                                  # dy/dx = 2x = 6.0
print(x.grad.get([0]))
```

**Parity & performance context:** numeric results match the reference implementation on every benchmarked kernel (max|diff| 0.0 → 1.1e-14). Interpreter-level timings run 400–2800× slower; `tkvc -O parallel -O simd` lowers the same loops to native SIMD/parallel code. See `src/tokenvector/README.md` §5–§6 for the full tables.

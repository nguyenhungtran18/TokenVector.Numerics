# TOKENVECTOR LANGUAGE - NATIVE NUMERICS & TENSOR SYNTAX SPECIFICATION

[ 🇬🇧 English ](TOKENVECTOR_SYNTAX_SPEC.md) | [ 🇻🇳 Tiếng Việt ](TOKENVECTOR_SYNTAX_SPEC_VI.md)

**Document Code:** TKV-SPEC-SYNTAX-2026-V1  
**Target:** Formal native mathematics, multidimensional tensor, and automatic differentiation syntax specification for the **TokenVector** programming language—100% standalone, with zero dependencies on or borrowing from `numpy` / `np`.

---

## 1. Namespaces & Module Imports

In TokenVector, the sole native top-level prefix is **`tv`** (or the full module name **`tokenvector`**):

```tokenvector
# 1. Import core module
import tv

# 2. Import specific submodules and constructs
from tv import array, tensor, zeros, ones, full, linspace
from tv import linalg, autograd, nn, optim
from tv import science, finance, physics, quantum, spatial
```

---

## 2. Multidimensional Arrays & Memory Allocation (NDArray)

```tokenvector
# Create multidimensional arrays
a = tv.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
b = tv.zeros([4, 4])
c = tv.ones([2, 3, 4])
d = tv.linspace(start=0.0, stop=1.0, num=100)

# Zero-Copy Slicing $O(1)$
view = a[0:2, 1:3]

# Allocate Unmanaged Native Memory via `with` block
with tv.allocate_native([1024, 1024], dtype=tv.f32) as native_buf:
    # High-throughput computation with Zero Garbage Collection pressure
    pass
# RAM is automatically freed immediately upon exiting the `with` block
```

---

## 3. Linear Algebra & Matrix Computing (`tv.linalg`)

```tokenvector
import tv.linalg as la

A = tv.array([[4.0, 1.0], [1.0, 3.0]])
B = tv.array([[2.0, 0.0], [0.0, 5.0]])

# Matrix Multiplication using the `@` operator or `la.matmul`
C = A @ B
C = la.matmul(A, B)

# Solve linear systems: Ax = b
x = la.solve(A, b)

# Matrix Decompositions
U, S, Vt = la.svd(A)
eigen_vals, eigen_vecs = la.eigen(A)

# Einstein Summation Contraction
result = la.einsum("ij,jk->ik", A, B)

# Analytical Padé Matrix Exponential and Square Root
expA = la.expm(A)
sqrtA = la.sqrtm(A)
```

---

## 4. Automatic Differentiation & Deep Learning (`tv.autograd` & `tv.nn`)

```tokenvector
from tv import tensor
import tv.nn as nn
import tv.optim as optim

# 1. Dynamic Graph Reverse-Mode Autograd
x = tv.tensor(3.0, requires_grad=True)
y = tv.tensor(2.0, requires_grad=True)

z = (x + y) * (x - y)  # z = x^2 - y^2
z.backward()

print(x.grad)  # 6.0 (2*x)
print(y.grad)  # -4.0 (-2*y)

# 2. Neural Network Model Architecture
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=16)
        self.norm = nn.RMSNorm(dim=16)
        self.fc2 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x: tv.Tensor) -> tv.Tensor:
        x = self.fc1(x).tanh()
        x = self.norm(x)
        return self.fc2(x).sigmoid()

# 3. Standard TokenVector Training Loop
model = MLP()
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    predictions = model.forward(inputs)
    loss = nn.mse_loss(predictions, targets)
    loss.backward()
    optimizer.step()
```

---

## 5. Domain-Specific Mathematics Modules

### A. Astrodynamics & Space Mechanics (`tv.astro`)
```tokenvector
import tv.astro as astro

# Solve Kepler's equation and orbit propagation
E = astro.solve_kepler(mean_anomaly=1.25, eccentricity=0.05)
r, v = astro.kepler_to_cartesian(a=7000.0, e=0.01, i=0.9, raan=1.2, omega=0.5, nu=0.8)
dv1, dv2, total_dv, tof = astro.hohmann_transfer(r_leo=6678.0, r_geo=42164.0)
```

### B. Quantitative Finance (`tv.finance`)
```tokenvector
import tv.finance as fin

# Black-Scholes-Merton option pricing & Greeks
call_price = fin.black_scholes_call(s=100.0, k=100.0, t=1.0, r=0.05, sigma=0.20)
delta, gamma, vega, theta, rho = fin.option_greeks(s=100.0, k=100.0, t=1.0, r=0.05, sigma=0.20)
sharpe = fin.sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.02)
```

### C. Computational Physics & LBM Fluid Dynamics (`tv.physics`)
```tokenvector
import tv.physics as phys

# 4th-order Runge-Kutta ODE integration & LBM D2Q9 Navier-Stokes
trajectory = phys.solve_rk4(harmonic_oscillator, t0=0.0, t1=10.0, y0=init_state, steps=200)
lbm_step = phys.lattice_boltzmann_2d(density, velocity, tau=0.6)
```

### D. Quantum Computing (`tv.quantum`)
```tokenvector
import tv.quantum as qtm

# Quantum Bell state simulation and circuit QFT
state = qtm.QState(num_qubits=2)
state.h(0)
state.cnot(control=0, target=1)
entropy = state.von_neumann_entropy()
```

---

## 6. Syntax Mapping Table (TokenVector Language $\leftrightarrow$ .NET CIL Runtime)

| TokenVector Language Syntax | Native .NET CIL Mapping (`TokenVector.Numerics.dll`) |
| :--- | :--- |
| `import tv` | `using TokenVector.Numerics.Core;` |
| `tv.array([1, 2, 3])` | `NDArray<double>.FromArray([1, 2, 3])` |
| `tv.tensor(x, requires_grad=True)` | `new Tensor<double>(x, requiresGrad: true)` |
| `a @ b` | `MatrixMultiplication.MatMul(a, b)` |
| `loss.backward()` | `loss.Backward()` |
| `optim.AdamW(params, lr=0.01)` | `new AdamW<double>(params, lr: 0.01)` |
| `nn.Linear(2, 8)` | `new Linear<double>(2, 8)` |
| `with tv.allocate_native(...) as buf:` | `using var buf = NDArray.AllocateNative(...)` |

---

## 7. Compiler Toolchain & Building with `tkvc.exe`

### 1. Download Compiler & Clone Ecosystem Libraries
To obtain the standalone compiler **`tkvc.exe`**, standard libraries (`stdlib`), and language toolchains, clone or visit the official repository:
```powershell
# Clone official TokenVector Compiler & Ecosystem Repository
git clone https://github.com/nguyenhungtran18/TokenVector.git
```

### 2. Compiling TokenVector Programs
You can compile your native TokenVector source files (`.tkv` or `.tv`) with direct linking to `TokenVector.Numerics.dll` into standalone native executables:
```powershell
# Compile source file to native Windows PE executable
./tkvc.exe main.tkv -r TokenVector.Numerics.dll -o app.exe

# Execute the native standalone binary
./app.exe
```


# TOKENVECTOR NATIVE MATHEMATICS & TENSOR SYNTAX SPECIFICATION

[ 🇬🇧 English ](TOKENVECTOR_SYNTAX_SPEC.md) | [ 🇻🇳 Tiếng Việt ](TOKENVECTOR_SYNTAX_SPEC_VI.md)

**Document code:** TKV-SPEC-SYNTAX-2026-V1  
**Purpose:** Defines the native numerical, multidimensional-tensor and automatic-differentiation syntax for the **TokenVector** language (TV-1001), 100% independent — it borrows no `numpy` / `np` prefix.

> **Source of truth.** Every symbol below was checked against the standard library in [`src/tokenvector/`](src/tokenvector/) (27 `.tkv` modules). The runtime is the `.tkv` stdlib itself; there is no separate C# engine. See the [stdlib module map](src/tokenvector/README.md) for per-module coverage.

---

## 1. Namespaces & Imports

The single native root prefix is **`tv`**. The 27 stdlib modules are re-exported into flat assembly scope by the package index `tv.tkv`.

```tokenvector
# 1. Import the package index and the core module
import tv
from tv.core import from_array, zeros, ones, full, linspace, arange, eye_rc

# 2. Import submodules by their real names (note: neural, not nn)
import tv.linalg as la
import tv.linalg_functions as lf
import tv.autograd as ag
import tv.neural as nn          # the module is `neural`; `nn` is only a local alias
import tv.optimize as optim

# 3. Domain modules (there is no umbrella `tv.science`)
import tv.astro
import tv.finance
import tv.physics
import tv.quantum
import tv.biology_robotics as bio
import tv.geometry3d
```

**Module names that do not exist** — use the real ones:

| ❌ Not a module | ✅ Real module |
| :--- | :--- |
| `tv.nn` | `tv.neural` |
| `tv.optim` | `tv.optimize` |
| `tv.science` | `tv.astro`, `tv.physics`, `tv.quantum`, `tv.biology_robotics`, `tv.geometry3d` |

---

## 2. Multidimensional Array Syntax & Memory (`NDArray`)

`dtype` is a **plain string literal** (`"f64"`, `"f32"`, `"i32"`, `"i64"`) — not a `tv.*` constant.

```tokenvector
# Construct n-dimensional arrays
a = tv.core.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
b = tv.core.zeros([4, 4])
c = tv.core.ones([2, 3, 4])
d = tv.core.full(7.5, [3, 3])
e = tv.core.linspace(0.0, 1.0, 100)
f = tv.core.arange(0.0, 10.0, 0.5)
g = tv.core.eye_rc(3, 4)                # rectangular identity

# Zero-copy O(1) slicing — a view shares the parent buffer
view = a.slice([[0, 2, 1], [1, 3, 1]])

# Unmanaged native allocation
native_buf = tv.core.allocate_native([1024, 1024], dtype="f32")
```

> **There is no `with` statement in TV-1001.** Call `allocate_native(...)` directly and bind the result. The `with tv.allocate_native(...) as buf:` form shown in earlier revisions of this document is not valid TV-1001 and does not appear anywhere in the stdlib.

---

## 3. Linear Algebra & Matrix Syntax (`tv.linalg`)

```tokenvector
import tv.linalg as la
import tv.linalg_functions as lf

A = tv.core.from_array([4.0, 1.0, 1.0, 3.0], [2, 2])
B = tv.core.from_array([2.0, 0.0, 0.0, 5.0], [2, 2])
b = tv.core.from_array([1.0, 2.0], [2])

# Matrix multiplication — call matmul(); there is no `@` operator
C = la.matmul(A, B)

# Solve A x = b
x = la.solve(A, b)

# Decompositions
p, l, u = la.lu(A)
q, r = la.qr(A)
L = la.cholesky(A)                      # A = L * L^T, lower triangular
U, S, Vt = la.svd(A)
vals, vecs = la.eigh(A)                 # symmetric eigen — `eigh`, not `eigen`
d = la.det(A)
tr = la.trace(A)
Ainv = la.inverse(A)

# Einstein summation
result = la.einsum("ij,jk->ik", A, B)

# Matrix functions live in `linalg_functions`, not `linalg`
expA = lf.expm(A)
sqrtA = lf.sqrtm(A)
```

> **Two names that changed.** The symmetric eigensolver is `eigh(a)`, not `eigen(a)` — `eigen` does not exist in the stdlib. And `expm` / `sqrtm` are in `tv.linalg_functions`; `tv.linalg` has no such names.

---

## 4. Automatic Differentiation & Deep Learning (`tv.autograd`, `tv.neural`)

Differentiable scalars are `Tensor` values built with the class-level factories.

```tokenvector
import tv.autograd as ag
import tv.neural as nn

# 1. Dynamic compute-graph autograd
x = ag.Tensor.full(3.0, [1])
x.requires_grad = True
y = (x * x) + (2.0 * x) + 1.0
y.backward()
print(x.grad.get([0]))                  # dy/dx = 2*3 + 2 = 8.0

# 2. Compose layers with Sequential / Linear (both live in `autograd`)
model = ag.Sequential([ag.Linear(2, 8), ag.Linear(8, 1)])
optimizer = ag.AdamW(model.parameters(), lr=0.05)

# 3. Activations and losses are plain functions in `tv.neural`
h = nn.relu(x)                          # or nn.sigmoid / nn.tanh / nn.gelu
loss = nn.mse_loss(predictions, targets)
loss.backward()
optimizer.step()
```

> **Where things live.** `Tensor`, `Module`, `Linear`, `Sequential`, `SGD` and `AdamW` are all in `tv.autograd`. `tv.neural` holds functional ops — `relu`, `sigmoid`, `tanh`, `gelu`, `softmax`, `mse_loss`, `cross_entropy_loss`, `layer_norm`, `rms_norm`, `scaled_dot_product_attention`, `im2col`. `tv.optimize` is classical numerical optimisation (`minimize_brent`, `minimize_bfgs`, `minimize_nelder_mead`, `root_brentq`, `bisection`) and does not define `AdamW`.

---

## 5. Advanced Multidisciplinary Mathematics

### A. Orbital Mechanics & Astrodynamics (`tv.astro`)

```tokenvector
import tv.astro as astro

E = astro.solve_kepler(1.25, 0.05)                       # eccentric anomaly
pos = astro.keplerian_to_cartesian(7000.0, 0.01, 0.9, 1.2, 0.5, 0.8)
dv1, dv2, total_dv, tof = astro.hohmann_transfer(6678.0, 42164.0)
xyz = astro.geodetic_to_ecef(10.0, 106.0, 0.0)          # WGS84
```

> Function is `keplerian_to_cartesian` (not `kepler_to_cartesian`), and the sixth argument is `nu=0.8` — an `=`, not a `:`. The `:` form in earlier revisions is a parse error.

### B. Quantitative Finance (`tv.finance`)

```tokenvector
import tv.finance as fin

call_price = fin.black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.20)
put_price = fin.black_scholes_put(100.0, 100.0, 1.0, 0.05, 0.20)
greeks = fin.option_greeks(100.0, 100.0, 1.0, 0.05, 0.20)   # delta..rho
sharpe = fin.sharpe_ratio(weights, expected_returns, cov_matrix, 0.02)
```

### C. Computational Physics (`tv.physics`)

```tokenvector
import tv.physics as phys

trajectory = phys.solve_rk4(force, 0.0, 10.0, init_state, 200)
div = phys.divergence_3d(fx, fy, fz, 1.0, 1.0, 1.0)
```

### D. Quantum Computing (`tv.quantum`)

```tokenvector
import tv.quantum as qtm

state = qtm.QState(2)
state.h(0)
state.cnot(0, 1)                         # control, target
entropy = qtm.von_neumann_entropy(density_matrix)
```

### E. Lattice Cryptography (`tv.crypto_graph`)

```tokenvector
import tv.crypto_graph as cg

coeffs = cg.forward_ntt(values, q, root_of_unity)
back = cg.inverse_ntt(coeffs, q, root_of_unity)
product = cg.poly_mul_ntt(poly_a, poly_b, q, root_of_unity)
```

---

## 6. Syntax Mapping (TokenVector → TokenVector stdlib)

The runtime *is* the `.tkv` standard library, so this table maps language surface to stdlib module rather than to a separate backend.

| TokenVector surface | stdlib implementation |
| :--- | :--- |
| `import tv` | `src/tokenvector/tv.tkv` (package index re-exporting 27 modules) |
| `tv.core.from_array([...], [2, 3])` | `core.tkv :: from_array` |
| `tv.core.zeros / ones / full / linspace` | `core.tkv` |
| `ndarray.slice([[...], [...]])` | `core.tkv :: NDArray.slice` (zero-copy view) |
| `tv.core.allocate_native(shape, dtype="f32")` | `core.tkv :: allocate_native` |
| `la.matmul(a, b)` | `linalg.tkv :: matmul` → `batch_matmul` (32×32 cache-tiled) |
| `la.solve / inverse / det / trace` | `linalg.tkv` |
| `la.qr / cholesky / svd / eigh` | `linalg.tkv` (Householder / A=LLᵀ / one-sided Jacobi / symmetric) |
| `ag.Tensor.full(3.0, [1])` | `autograd.tkv :: Tensor.full` |
| `loss.backward()` | `autograd.tkv :: Tensor.backward` (reverse-mode VJP) |
| `ag.Linear(2, 8)`, `ag.Sequential([...])` | `autograd.tkv :: Linear / Sequential` |
| `ag.AdamW(params, lr=...)` | `autograd.tkv :: AdamW` |
| `nn.relu / gelu / mse_loss` | `neural.tkv` |
| `lf.expm / sqrtm` | `linalg_functions.tkv` |

---

## 7. Compiler & Building Applications with `tkvc.exe`

### 1. Get the compiler

`tkvc.exe`, the standard library and the language tooling live in the official TokenVector repository:

```powershell
git clone https://github.com/nguyenhungtran18/TokenVector.git
```

### 2. Compile TokenVector source

`tkvc` exposes a single subcommand, `build`:

```powershell
# Compile a .tkv source file into a standalone executable
./tkvc.exe build main.tkv --out app.exe

# Run it
./app.exe
```

Useful flags (`tkvc build --help`):

| Flag | Meaning |
| :--- | :--- |
| `--entry NAME` | Entry-point function; defaults to the only function, or one named `main` |
| `--out PATH` | Output `.exe` path (default: source name with `.exe`) |
| `--target exe\|library` | Standalone executable (default) or managed DLL |
| `--debug` | Also emit a Windows `.pdb` |
| `--no-lint` | Disable the pre-flight TV-1001 syntax linter |

> **A stale invocation.** Earlier revisions of this document used `./tkvc.exe main.tkv -r TokenVector.Numerics.dll -o app.exe`. That form is **rejected** by the current compiler — `build` is the only subcommand and there is no `-r` / `-o` flag. Use `build --out`.

### 3. Resolving the `tv` package

For a file that does `import tv`, the compiler must be able to find the stdlib on its module search path. See [src/tokenvector/README.md](src/tokenvector/README.md) §3 for the current status of stdlib resolution and which suites build today.

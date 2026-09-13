# ĐẶC TẢ CÚ PHÁP TOÁN HỌC & TENSOR BẢN ĐỊA CHO NGÔN NGỮ TOKENVECTOR
### (TokenVector Native Mathematics & Tensor Syntax Specification)

[ 🇬🇧 English ](TOKENVECTOR_SYNTAX_SPEC.md) | [ 🇻🇳 Tiếng Việt ](TOKENVECTOR_SYNTAX_SPEC_VI.md)

**Mã tài liệu:** TKV-SPEC-SYNTAX-2026-V1  
**Mục tiêu:** Định nghĩa chuẩn cú pháp toán học số học, tensor đa chiều và tự động vi phân hoàn toàn bản địa cho ngôn ngữ **TokenVector**, độc lập 100%, không phụ thuộc hay vay mượn tiền tố `numpy` / `np`.

---

## 1. Hệ thống Không gian tên & Khởi tạo (Namespaces & Imports)

Trong TokenVector, tiền tố gốc bản địa duy nhất là **`tv`** (hoặc tên đầy đủ **`tokenvector`**):

```tokenvector
# 1. Nhập module cốt lõi
import tv

# 2. Nhập các submodule chuyên biệt theo chuẩn TokenVector
from tv import array, tensor, zeros, ones, full, linspace
from tv import linalg, autograd, nn, optim
from tv import science, finance, physics, quantum, spatial
```

---

## 2. Cú pháp Mảng Đa Chiều & Bộ Nhớ (NDArray & Memory)

```tokenvector
# Khởi tạo mảng n-chiều
a = tv.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
b = tv.zeros([4, 4])
c = tv.ones([2, 3, 4])
d = tv.linspace(start=0.0, stop=1.0, num=100)

# Cắt lát Zero-Copy $O(1)$
view = a[0:2, 1:3]

# Cấp phát bộ nhớ Unmanaged (Native Memory) với cú pháp `with`
with tv.allocate_native([1024, 1024], dtype=tv.f32) as native_buf:
    # Xử lý ma trận lớn với Zero GC Pressure
    pass
# Tự động giải phóng bộ nhớ RAM ngay khi thoát khối `with`
```

---

## 3. Cú pháp Đại Số Tuyến Tính & Ma Trận (`tv.linalg`)

```tokenvector
import tv.linalg as la

A = tv.array([[4.0, 1.0], [1.0, 3.0]])
B = tv.array([[2.0, 0.0], [0.0, 5.0]])

# Nhân ma trận bằng toán tử `@` hoặc hàm `la.matmul`
C = A @ B
C = la.matmul(A, B)

# Giải hệ phương trình Ax = b
x = la.solve(A, b)

# Phân rã ma trận
U, S, Vt = la.svd(A)
eigen_vals, eigen_vecs = la.eigen(A)

# Co rút tensor Einstein Summation
result = la.einsum("ij,jk->ik", A, B)

# Ma trận mũ giải tích Padé và Căn bậc hai ma trận
expA = la.expm(A)
sqrtA = la.sqrtm(A)
```

---

## 4. Cú pháp Tự Động Vi Phân & Deep Learning (`tv.autograd` & `tv.nn`)

```tokenvector
from tv import tensor
import tv.nn as nn
import tv.optim as optim

# 1. Tự động vi phân (Dynamic Graph Autograd)
x = tv.tensor(3.0, requires_grad=True)
y = tv.tensor(2.0, requires_grad=True)

z = (x + y) * (x - y)  # z = x^2 - y^2
z.backward()

print(x.grad)  # 6.0 (2*x)
print(y.grad)  # -4.0 (-2*y)

# 2. Định nghĩa mô hình Mạng Nơ-ron (Neural Network)
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

# 3. Vòng lặp huấn luyện chuẩn TokenVector
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

## 5. Cú pháp Toán Học Đa Ngành Chuyên Sâu

### A. Cơ Học Không Gian & Thiên Văn (`tv.astro`)
```tokenvector
import tv.astro as astro

# Giải Kepler và chuyển đổi quỹ đạo
E = astro.solve_kepler(mean_anomaly=1.25, eccentricity=0.05)
r, v = astro.kepler_to_cartesian(a=7000.0, e=0.01, i=0.9, raan=1.2, omega=0.5, nu: 0.8)
dv1, dv2, total_dv, tof = astro.hohmann_transfer(r_leo=6678.0, r_geo=42164.0)
```

### B. Toán Tài Chính Định Lượng (`tv.finance`)
```tokenvector
import tv.finance as fin

# Định giá quyền chọn Black-Scholes & The Greeks
call_price = fin.black_scholes_call(s=100.0, k=100.0, t=1.0, r=0.05, sigma=0.20)
delta, gamma, vega, theta, rho = fin.option_greeks(s=100.0, k=100.0, t=1.0, r=0.05, sigma=0.20)
sharpe = fin.sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.02)
```

### C. Vật Lý Tính Toán & LBM CFD (`tv.physics`)
```tokenvector
import tv.physics as phys

# Tích phân vi phân Runge-Kutta 4 & Thủy động lực học Navier-Stokes LBM
trajectory = phys.solve_rk4(harmonic_oscillator, t0=0.0, t1=10.0, y0=init_state, steps=200)
lbm_step = phys.lattice_boltzmann_2d(density, velocity, tau=0.6)
```

### D. Điện Toán Lượng Tử (`tv.quantum`)
```tokenvector
import tv.quantum as qtm

# Trạng thái Bell lượng tử và mạch QFT
state = qtm.QState(num_qubits=2)
state.h(0)
state.cnot(control=0, target=1)
entropy = state.von_neumann_entropy()
```

---

## 6. Bảng Ánh Xạ Cú Pháp (TokenVector Language $\leftrightarrow$ .NET CIL Runtime)

| Cú pháp Ngôn ngữ TokenVector | Ánh xạ Native .NET CIL (`TokenVector.Numerics.dll`) |
| :--- | :--- |
| `import tv` | `using TokenVector.Numerics.Core;` |
| `tv.array([1, 2, 3])` | `NDArray<double>.FromArray([1, 2, 3])` |
| `tv.tensor(x, requires_grad=True)` | `new Tensor<double>(x, requiresGrad: true)` |
| `a @ b` | `MatrixMultiplication.MatMul(a, b)` |
| `loss.backward()` | `loss.Backward()` |
| `optim.AdamW(params, lr=0.01)` | `new AdamW<double>(params, lr: 0.01)` |
| `nn.Linear(2, 8)` | `new Linear<double>(2, 8)` |
| `with tv.allocate_native(...) as buf:` | `using var buf = NDArray.AllocateNative(...)` |

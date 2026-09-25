# ĐẶC TẢ CÚ PHÁP TOÁN HỌC & TENSOR BẢN ĐỊA CHO NGÔN NGỮ TOKENVECTOR

[ 🇬🇧 English ](TOKENVECTOR_SYNTAX_SPEC.md) | [ 🇻🇳 Tiếng Việt ](TOKENVECTOR_SYNTAX_SPEC_VI.md)

**Mã tài liệu:** TKV-SPEC-SYNTAX-2026-V1  
**Mục tiêu:** Định nghĩa chuẩn cú pháp toán học số học, tensor đa chiều và tự động vi phân hoàn toàn bản địa cho ngôn ngữ **TokenVector** (TV-1001), độc lập 100%, không vay mượn tiền tố `numpy` / `np`.

> **Nguồn sự thật.** Mọi ký hiệu dưới đây đã được đối chiếu với thư viện chuẩn trong [`src/tokenvector/`](src/tokenvector/) (27 module `.tkv`). Runtime chính là bộ stdlib `.tkv`; không còn engine C# riêng. Xem [bản đồ module stdlib](src/tokenvector/README.md) để biết độ phủ theo từng module.

---

## 1. Hệ thống Không gian tên & Khởi tạo (Namespaces & Imports)

Tiền tố gốc bản địa duy nhất là **`tv`**. Bản chỉ mục gói `tv.tkv` re-export 27 module stdlib vào cùng một phạm vi.

```tokenvector
# 1. Nhập chỉ mục gói và module cốt lõi
import tv
from tv.core import from_array, zeros, ones, full, linspace, arange, eye_rc

# 2. Nhập submodule theo tên thật (lưu ý: neural, không phải nn)
import tv.linalg as la
import tv.linalg_functions as lf
import tv.autograd as ag
import tv.neural as nn          # module tên là `neural`; `nn` chỉ là bí danh cục bộ
import tv.optimize as optim

# 3. Module theo lĩnh vực (không có module gộp `tv.science`)
import tv.astro
import tv.finance
import tv.physics
import tv.quantum
import tv.biology_robotics as bio
import tv.geometry3d
```

**Những tên module không tồn tại** — hãy dùng tên thật:

| ❌ Không phải module | ✅ Module thật |
| :--- | :--- |
| `tv.nn` | `tv.neural` |
| `tv.optim` | `tv.optimize` |
| `tv.science` | `tv.astro`, `tv.physics`, `tv.quantum`, `tv.biology_robotics`, `tv.geometry3d` |

---

## 2. Cú pháp Mảng Đa Chiều & Bộ Nhớ (`NDArray`)

`dtype` là **chuỗi ký tự thuần** (`"f64"`, `"f32"`, `"i32"`, `"i64"`) — không phải hằng `tv.*`.

```tokenvector
# Khởi tạo mảng n-chiều
a = tv.core.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
b = tv.core.zeros([4, 4])
c = tv.core.ones([2, 3, 4])
d = tv.core.full(7.5, [3, 3])
e = tv.core.linspace(0.0, 1.0, 100)
f = tv.core.arange(0.0, 10.0, 0.5)
g = tv.core.eye_rc(3, 4)                # ma trận đơn vị chữ nhật

# Cắt lát zero-copy O(1) — view dùng chung buffer với mảng gốc
view = a.slice([[0, 2, 1], [1, 3, 1]])

# Cấp phát bộ nhớ unmanaged
native_buf = tv.core.allocate_native([1024, 1024], dtype="f32")
```

> **TV-1001 không có câu lệnh `with`.** Hãy gọi trực tiếp `allocate_native(...)` rồi gán kết quả. Dạng `with tv.allocate_native(...) as buf:` xuất hiện ở các bản sửa đổi trước của tài liệu này không phải cú pháp TV-1001 hợp lệ và không xuất hiện ở bất kỳ đâu trong stdlib.

---

## 3. Cú pháp Đại Số Tuyến Tính & Ma Trận (`tv.linalg`)

```tokenvector
import tv.linalg as la
import tv.linalg_functions as lf

A = tv.core.from_array([4.0, 1.0, 1.0, 3.0], [2, 2])
B = tv.core.from_array([2.0, 0.0, 0.0, 5.0], [2, 2])
b = tv.core.from_array([1.0, 2.0], [2])

# Nhân ma trận — gọi matmul(); không có toán tử `@`
C = la.matmul(A, B)

# Giải hệ phương trình A x = b
x = la.solve(A, b)

# Các phân rã
p, l, u = la.lu(A)
q, r = la.qr(A)
L = la.cholesky(A)                      # A = L * L^T, tam giác dưới
U, S, Vt = la.svd(A)
vals, vecs = la.eigh(A)                 # eigen đối xứng — `eigh`, không phải `eigen`
d = la.det(A)
tr = la.trace(A)
Ainv = la.inverse(A)

# Co rút Einstein
result = la.einsum("ij,jk->ik", A, B)

# Hàm ma trận nằm ở `linalg_functions`, không phải `linalg`
expA = lf.expm(A)
sqrtA = lf.sqrtm(A)
```

> **Hai tên đã thay đổi.** Bộ giải eigen đối xứng là `eigh(a)`, không phải `eigen(a)` — `eigen` không tồn tại trong stdlib. Và `expm` / `sqrtm` nằm ở `tv.linalg_functions`; `tv.linalg` không có các tên này.

---

## 4. Cú pháp Tự Động Vi Phân & Học Sâu (`tv.autograd`, `tv.neural`)

Các giá trị vô hướng khả vi là giá trị `Tensor`, tạo bằng các hàm tạo ở cấp lớp.

```tokenvector
import tv.autograd as ag
import tv.neural as nn

# 1. Autograd đồ thị tính toán động
x = ag.Tensor.full(3.0, [1])
x.requires_grad = True
y = (x * x) + (2.0 * x) + 1.0
y.backward()
print(x.grad.get([0]))                  # dy/dx = 2*3 + 2 = 8.0

# 2. Ghép lớp bằng Sequential / Linear (cả hai nằm ở `autograd`)
model = ag.Sequential([ag.Linear(2, 8), ag.Linear(8, 1)])
optimizer = ag.AdamW(model.parameters(), lr=0.05)

# 3. Hàm kích hoạt và hàm mất mát là hàm thuần ở `tv.neural`
h = nn.relu(x)                          # hoặc nn.sigmoid / nn.tanh / nn.gelu
loss = nn.mse_loss(predictions, targets)
loss.backward()
optimizer.step()
```

> **Mọi thứ nằm ở đâu.** `Tensor`, `Module`, `Linear`, `Sequential`, `SGD` và `AdamW` đều ở `tv.autograd`. `tv.neural` chứa các hàm — `relu`, `sigmoid`, `tanh`, `gelu`, `softmax`, `mse_loss`, `cross_entropy_loss`, `layer_norm`, `rms_norm`, `scaled_dot_product_attention`, `im2col`. `tv.optimize` là tối ưu hoá số học cổ điển (`minimize_brent`, `minimize_bfgs`, `minimize_nelder_mead`, `root_brentq`, `bisection`) và không định nghĩa `AdamW`.

---

## 5. Toán Học Đa Ngành Chuyên Sâu

### A. Cơ Học Quỹ Đạo & Thiên Văn (`tv.astro`)

```tokenvector
import tv.astro as astro

E = astro.solve_kepler(1.25, 0.05)                       # dịch dương
pos = astro.keplerian_to_cartesian(7000.0, 0.01, 0.9, 1.2, 0.5, 0.8)
dv1, dv2, total_dv, tof = astro.hohmann_transfer(6678.0, 42164.0)
xyz = astro.geodetic_to_ecef(10.0, 106.0, 0.0)          # WGS84
```

> Tên hàm là `keplerian_to_cartesian` (không phải `kepler_to_cartesian`), và tham số thứ sáu là `nu=0.8` — dùng `=`, không phải `:`. Dạng `:` ở các bản sửa đổi trước là lỗi cú pháp.

### B. Toán Tài Chính Định Lượng (`tv.finance`)

```tokenvector
import tv.finance as fin

call_price = fin.black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.20)
put_price = fin.black_scholes_put(100.0, 100.0, 1.0, 0.05, 0.20)
greeks = fin.option_greeks(100.0, 100.0, 1.0, 0.05, 0.20)   # delta..rho
sharpe = fin.sharpe_ratio(weights, expected_returns, cov_matrix, 0.02)
```

### C. Vật Lý Tính Toán (`tv.physics`)

```tokenvector
import tv.physics as phys

trajectory = phys.solve_rk4(force, 0.0, 10.0, init_state, 200)
div = phys.divergence_3d(fx, fy, fz, 1.0, 1.0, 1.0)
```

### D. Điện Toán Lượng Tử (`tv.quantum`)

```tokenvector
import tv.quantum as qtm

state = qtm.QState(2)
state.h(0)
state.cnot(0, 1)                         # qubit điều khiển, qubit đích
entropy = qtm.von_neumann_entropy(density_matrix)
```

### E. Mật Mã Lattice (`tv.crypto_graph`)

```tokenvector
import tv.crypto_graph as cg

coeffs = cg.forward_ntt(values, q, root_of_unity)
back = cg.inverse_ntt(coeffs, q, root_of_unity)
product = cg.poly_mul_ntt(poly_a, poly_b, q, root_of_unity)
```

---

## 6. Bảng Ánh Xạ Cú Pháp (TokenVector → stdlib TokenVector)

Runtime *chính là* thư viện chuẩn `.tkv`, nên bảng này ánh xạ bề mặt ngôn ngữ sang module stdlib, chứ không phải sang một backend riêng.

| Bề mặt TokenVector | Cài đặt trong stdlib |
| :--- | :--- |
| `import tv` | `src/tokenvector/tv.tkv` (chỉ mục gói re-export 27 module) |
| `tv.core.from_array([...], [2, 3])` | `core.tkv :: from_array` |
| `tv.core.zeros / ones / full / linspace` | `core.tkv` |
| `ndarray.slice([[...], [...]])` | `core.tkv :: NDArray.slice` (view zero-copy) |
| `tv.core.allocate_native(shape, dtype="f32")` | `core.tkv :: allocate_native` |
| `la.matmul(a, b)` | `linalg.tkv :: matmul` → `batch_matmul` (cache-tiling 32×32) |
| `la.solve / inverse / det / trace` | `linalg.tkv` |
| `la.qr / cholesky / svd / eigh` | `linalg.tkv` (Householder / A=LLᵀ / Jacobi một phía / đối xứng) |
| `ag.Tensor.full(3.0, [1])` | `autograd.tkv :: Tensor.full` |
| `loss.backward()` | `autograd.tkv :: Tensor.backward` (VJP chế độ đảo) |
| `ag.Linear(2, 8)`, `ag.Sequential([...])` | `autograd.tkv :: Linear / Sequential` |
| `ag.AdamW(params, lr=...)` | `autograd.tkv :: AdamW` |
| `nn.relu / gelu / mse_loss` | `neural.tkv` |
| `lf.expm / sqrtm` | `linalg_functions.tkv` |

---

## 7. Trình Biên Dịch & Biên Dịch Ứng Dụng với `tkvc.exe`

### 1. Tải trình biên dịch

`tkvc.exe`, thư viện chuẩn và công cụ ngôn ngữ nằm trong repository chính thức của TokenVector:

```powershell
git clone https://github.com/nguyenhungtran18/TokenVector.git
```

### 2. Biên dịch mã nguồn TokenVector

`tkvc` chỉ có một lệnh con duy nhất, `build`:

```powershell
# Biên dịch file nguồn .tkv thành file thực thi độc lập
./tkvc.exe build main.tkv --out app.exe

# Chạy
./app.exe
```

Các tuỳ chọn thường dùng (`tkvc build --help`):

| Tuỳ chọn | Ý nghĩa |
| :--- | :--- |
| `--entry NAME` | Hàm làm entry point; mặc định là hàm duy nhất, hoặc hàm tên `main` |
| `--out PATH` | Đường dẫn `.exe` đầu ra (mặc định: tên nguồn đổi đuôi `.exe`) |
| `--target exe\|library` | File thực thi độc lập (mặc định) hoặc DLL managed |
| `--debug` | Sinh thêm file `.pdb` cho Windows |
| `--no-lint` | Tắt bộ lint cú pháp TV-1001 |

> **Một cách gọi đã lỗi thời.** Các bản sửa đổi trước của tài liệu này dùng `./tkvc.exe main.tkv -r TokenVector.Numerics.dll -o app.exe`. Dạng đó **bị từ chối** bởi compiler hiện tại — `build` là lệnh con duy nhất và không có cờ `-r` / `-o`. Hãy dùng `build --out`.

### 3. Phân giải gói `tv`

Với một file dùng `import tv`, compiler phải tìm được stdlib trên đường dẫn module của nó. Xem [src/tokenvector/README.md](src/tokenvector/README.md) §3 để biết hiện trạng phân giải stdlib và những suite nào build được ngày hôm nay.

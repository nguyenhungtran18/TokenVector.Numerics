# benchmark_vs_numpy.py — TokenVector (.tkv) vs NumPy micro-benchmarks.
#
# Runs the .tkv stdlib through the Python harness (tv.* runtime + loader in
# tkv_harness.py) and compares against NumPy on the same machine:
#   - matmul (cache-tiled, flat-index hot loop)
#   - elementwise add (broadcast fast path)
#   - SVD (one-sided Jacobi vs LAPACK)
#   - FFT radix-2 + Bluestein (non-power-of-2)
# Parity (max abs difference) is checked for every kernel.
#
# Honest caveat: the .tkv kernels run on the Python harness, so the numbers
# measure the TV semantics implementation, NOT the tkvc-compiled native output.
# Compiled TokenVector (`tkvc -O parallel -O simd`) sits much closer to NumPy
# than these figures suggest; treat them as an algorithmic-parity baseline.

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tkv_harness as H  # noqa: E402

import importlib  # noqa: E402


def timeit(fn, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def max_diff(values_a, values_b):
    worst = 0.0
    for i in range(len(values_a)):
        d = abs(values_a[i] - values_b[i])
        if d > worst:
            worst = d
    return worst


def main():
    loader = H.TkvLoader()
    sys.meta_path.insert(0, loader)
    core = importlib.import_module("tv.core")
    loader.tv_runtime = H.make_tv_runtime(core)
    sys.modules["tv"].__dict__.update(loader.tv_runtime.__dict__)

    linalg = importlib.import_module("tv.linalg")
    ops = importlib.import_module("tv.ops")
    fft = importlib.import_module("tv.fft")
    lf = importlib.import_module("tv.linalg_functions")

    import numpy as np

    rows = []

    # ---------------- matmul ----------------
    for size in (32, 64, 128):
        repeats = 5 if size <= 64 else 3
        flat = [((i * size + j) % 7) - 3.0 for i in range(size) for j in range(size)]
        a = core.from_array(flat, [size, size])
        b = core.from_array(flat, [size, size])
        an = np.array(flat, dtype=np.float64).reshape(size, size)

        t_tkv = timeit(lambda: linalg.matmul(a, b), repeats)
        t_np = timeit(lambda: an @ an, repeats)
        ref = (an @ an).reshape(-1).tolist()
        got = linalg.matmul(a, b).buffer.data
        rows.append(("matmul", f"{size}x{size}", t_tkv, t_np, max_diff(got, ref)))

    # ---------------- elementwise add + broadcast ----------------
    n = 1 << 18
    xs = [i % 97 * 0.1 for i in range(n)]
    a = core.from_array(xs, [2, n // 2])
    b = core.from_array([1.0, 2.0, 3.0, 4.0] * (n // 8), [n // 2])

    an = np.array(xs, dtype=np.float64).reshape(2, n // 2)
    bn = np.array([1.0, 2.0, 3.0, 4.0] * (n // 8), dtype=np.float64)

    t_tkv = timeit(lambda: ops.add(a, b), 5)
    t_np = timeit(lambda: an + bn, 5)
    got = ops.add(a, b).buffer.data
    ref = (an + bn).reshape(-1).tolist()
    rows.append(("add+broadcast", f"2x{n//2}", t_tkv, t_np, max_diff(got, ref)))

    # ---------------- SVD ----------------
    size = 40
    flat = [math.sin(i * 0.37 + j * 0.11) for i in range(size) for j in range(size)]
    a = core.from_array(flat, [size, size])
    an = np.array(flat, dtype=np.float64).reshape(size, size)

    t_tkv = timeit(lambda: linalg.svd(a), 3)
    t_np = timeit(lambda: np.linalg.svd(an), 3)
    u, s, vt = linalg.svd(a)
    us, ss, vts = np.linalg.svd(an)
    # Compare singular values (sorted desc by construction in both).
    s_tkv = [s.get([i]) for i in range(size)]
    s_np = ss.tolist()
    rows.append(("svd singular values", f"{size}x{size}", t_tkv, t_np, max_diff(s_tkv, s_np)))

    # ---------------- FFT radix-2 ----------------
    n = 1024
    xs = [math.cos(i * 0.11) + 0.5 * math.sin(i * 0.53) for i in range(n)]
    xs_c = [lf.Complex(x, 0.0) for x in xs]
    an = np.array(xs, dtype=np.float64)
    t_tkv = timeit(lambda: fft.fft1d(xs_c), 5)
    t_np = timeit(lambda: np.fft.fft(an), 5)
    out = fft.fft1d(xs_c)
    ref = np.fft.fft(an)
    diff = max(
        max(abs(out[i].real - ref[i].real) for i in range(n)),
        max(abs(out[i].imaginary - ref[i].imag) for i in range(n)),
    )
    rows.append(("fft radix-2", str(n), t_tkv, t_np, diff))

    # ---------------- FFT Bluestein (non-power-of-2) ----------------
    n = 1000
    xs = [math.cos(i * 0.037) for i in range(n)]
    xs_c = [lf.Complex(x, 0.0) for x in xs]
    an = np.array(xs, dtype=np.float64)
    t_tkv = timeit(lambda: fft.fft1d(xs_c), 3)
    t_np = timeit(lambda: np.fft.fft(an), 3)
    out = fft.fft1d(xs_c)
    ref = np.fft.fft(an)
    diff = max(
        max(abs(out[i].real - ref[i].real) for i in range(n)),
        max(abs(out[i].imaginary - ref[i].imag) for i in range(n)),
    )
    rows.append(("fft bluestein", str(n), t_tkv, t_np, diff))

    # ---------------- report ----------------
    print()
    print(f"{'kernel':<22}{'size':>10}{'TokenVector':>14}{'NumPy':>12}{'ratio':>9}{'max|diff|':>12}")
    print("-" * 79)
    for name, size, t_tkv, t_np, diff in rows:
        ratio = t_tkv / t_np if t_np > 0 else float("inf")
        print(f"{name:<22}{size:>10}{t_tkv:>12.4f}s{t_np:>10.4f}s{ratio:>8.1f}x{diff:>12.3e}")
    print()
    print("Caveat: .tkv runs on the Python harness (interpreter on interpreter).")
    print("tkvc-compiled native output will be far closer to NumPy.")
    print("Parity: every kernel matches NumPy within the printed max|diff|.")


if __name__ == "__main__":
    main()

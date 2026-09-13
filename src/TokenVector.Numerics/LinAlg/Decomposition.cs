using System;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.LinAlg;

/// <summary>
/// Numerical linear algebra matrix decompositions: LU, QR, Cholesky, Inverse, Determinant, Trace, and System Solver.
/// </summary>
public static class Decomposition
{
    /// <summary>
    /// Computes LU decomposition with partial pivoting: PA = LU.
    /// </summary>
    /// <returns>(P: Permutation matrix, L: Unit lower triangular, U: Upper triangular)</returns>
    public static (NDArray<T> P, NDArray<T> L, NDArray<T> U) LU<T>(NDArray<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1])
        {
            throw new ArgumentException("LU decomposition requires a square 2D matrix.");
        }

        int n = a.Shape[0];
        var u = a.Clone();
        var l = NDArray<T>.Eye(n);
        var p = NDArray<T>.Eye(n);

        int[] piv = new int[n];
        for (int i = 0; i < n; i++) piv[i] = i;

        for (int k = 0; k < n - 1; k++)
        {
            // Find pivot
            int maxRow = k;
            T maxVal = T.Abs(u[k, k]);

            for (int i = k + 1; i < n; i++)
            {
                T val = T.Abs(u[i, k]);
                if (val > maxVal)
                {
                    maxVal = val;
                    maxRow = i;
                }
            }

            if (maxRow != k)
            {
                // Swap rows in U
                for (int j = 0; j < n; j++)
                {
                    (u[k, j], u[maxRow, j]) = (u[maxRow, j], u[k, j]);
                    (p[k, j], p[maxRow, j]) = (p[maxRow, j], p[k, j]);
                }
                // Swap sub-diagonal elements in L
                for (int j = 0; j < k; j++)
                {
                    (l[k, j], l[maxRow, j]) = (l[maxRow, j], l[k, j]);
                }
                (piv[k], piv[maxRow]) = (piv[maxRow], piv[k]);
            }

            T pivot = u[k, k];
            if (T.Abs(pivot) < T.CreateChecked(1e-15)) continue;

            for (int i = k + 1; i < n; i++)
            {
                T factor = u[i, k] / pivot;
                l[i, k] = factor;
                u[i, k] = T.Zero;

                for (int j = k + 1; j < n; j++)
                {
                    u[i, j] -= factor * u[k, j];
                }
            }
        }

        return (p, l, u);
    }

    /// <summary>
    /// Computes QR decomposition using Householder reflections: A = QR.
    /// </summary>
    /// <returns>(Q: Orthogonal matrix, R: Upper triangular matrix)</returns>
    public static (NDArray<T> Q, NDArray<T> R) QR<T>(NDArray<T> a) where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        if (a.Rank != 2) throw new ArgumentException("QR decomposition requires a 2D matrix.");
        int m = a.Shape[0];
        int n = a.Shape[1];

        var q = NDArray<T>.Eye(m);
        var r = a.Clone();

        int kLimit = Math.Min(m, n);
        for (int k = 0; k < kLimit; k++)
        {
            // Compute norm of column subvector
            T normX2 = T.Zero;
            for (int i = k; i < m; i++) normX2 += r[i, k] * r[i, k];
            T normX = T.Sqrt(normX2);
            if (normX < T.CreateChecked(1e-15)) continue;

            T alpha = (r[k, k] >= T.Zero) ? -normX : normX;
            T u1 = r[k, k] - alpha;

            T[] v = new T[m - k];
            v[0] = T.One;
            for (int i = k + 1; i < m; i++) v[i - k] = r[i, k] / u1;

            T beta = -u1 / alpha;

            // Apply Householder reflection to R: R = (I - beta * v * v^T) * R
            for (int j = k; j < n; j++)
            {
                T dot = T.Zero;
                for (int i = k; i < m; i++) dot += v[i - k] * r[i, j];
                T factor = beta * dot;
                for (int i = k; i < m; i++) r[i, j] -= factor * v[i - k];
            }

            // Accumulate Q: Q = Q * (I - beta * v * v^T)
            for (int i = 0; i < m; i++)
            {
                T dot = T.Zero;
                for (int j = k; j < m; j++) dot += q[i, j] * v[j - k];
                T factor = beta * dot;
                for (int j = k; j < m; j++) q[i, j] -= factor * v[j - k];
            }
        }

        return (q, r);
    }

    /// <summary>
    /// Computes Cholesky decomposition of a symmetric positive-definite matrix: A = L * L^T.
    /// </summary>
    public static NDArray<T> Cholesky<T>(NDArray<T> a) where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1])
        {
            throw new ArgumentException("Cholesky decomposition requires a square symmetric matrix.");
        }

        int n = a.Shape[0];
        var l = new NDArray<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                T sum = T.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum += l[i, k] * l[j, k];
                }

                if (i == j)
                {
                    T diag = a[i, i] - sum;
                    if (diag <= T.Zero)
                    {
                        throw new InvalidOperationException($"Matrix is not positive definite at index ({i}, {i}).");
                    }
                    l[i, j] = T.Sqrt(diag);
                }
                else
                {
                    l[i, j] = (a[i, j] - sum) / l[j, j];
                }
            }
        }

        return l;
    }

    /// <summary>
    /// Solves linear system Ax = b using LU decomposition.
    /// </summary>
    public static NDArray<T> Solve<T>(NDArray<T> a, NDArray<T> b) where T : unmanaged, IFloatingPoint<T>
    {
        var (p, l, u) = LU(a);
        int n = a.Shape[0];

        // Apply permutation to b: Pb
        var pb = MatrixMultiplication.MatMul(p, b.Rank == 1 ? b.Reshape(n, 1) : b);
        var y = new NDArray<T>(n, 1);

        // Forward substitution: Ly = Pb
        for (int i = 0; i < n; i++)
        {
            T sum = T.Zero;
            for (int j = 0; j < i; j++) sum += l[i, j] * y[j, 0];
            y[i, 0] = pb[i, 0] - sum;
        }

        // Back substitution: Ux = y
        var x = new NDArray<T>(n, 1);
        for (int i = n - 1; i >= 0; i--)
        {
            T sum = T.Zero;
            for (int j = i + 1; j < n; j++) sum += u[i, j] * x[j, 0];
            x[i, 0] = (y[i, 0] - sum) / u[i, i];
        }

        return b.Rank == 1 ? x.Reshape(n) : x;
    }

    /// <summary>
    /// Computes matrix inverse A^-1.
    /// </summary>
    public static NDArray<T> Inverse<T>(NDArray<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1])
        {
            throw new ArgumentException("Matrix inverse requires a square 2D matrix.");
        }

        int n = a.Shape[0];
        var identity = NDArray<T>.Eye(n);
        var (p, l, u) = LU(a);

        var inv = new NDArray<T>(n, n);
        for (int col = 0; col < n; col++)
        {
            // Solve A * x_col = e_col
            var eCol = new NDArray<T>(n, 1);
            for (int i = 0; i < n; i++) eCol[i, 0] = p[i, col];

            // Ly = eCol
            var y = new NDArray<T>(n, 1);
            for (int i = 0; i < n; i++)
            {
                T sum = T.Zero;
                for (int j = 0; j < i; j++) sum += l[i, j] * y[j, 0];
                y[i, 0] = eCol[i, 0] - sum;
            }

            // Ux = y
            for (int i = n - 1; i >= 0; i--)
            {
                T sum = T.Zero;
                for (int j = i + 1; j < n; j++) sum += u[i, j] * inv[j, col];
                inv[i, col] = (y[i, 0] - sum) / u[i, i];
            }
        }

        return inv;
    }

    /// <summary>
    /// Computes determinant of a square matrix.
    /// </summary>
    public static T Det<T>(NDArray<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1])
        {
            throw new ArgumentException("Determinant requires a square 2D matrix.");
        }

        var (p, l, u) = LU(a);
        int n = a.Shape[0];
        T det = T.One;

        for (int i = 0; i < n; i++)
        {
            det *= u[i, i];
        }

        // Check parity of permutation P
        int swaps = 0;
        bool[] visited = new bool[n];
        for (int i = 0; i < n; i++)
        {
            if (visited[i]) continue;
            int len = 0;
            int curr = i;
            while (!visited[curr])
            {
                visited[curr] = true;
                // Find where 1 is in row curr of P
                int next = 0;
                for (int j = 0; j < n; j++)
                {
                    if (p[curr, j] == T.One) { next = j; break; }
                }
                curr = next;
                len++;
            }
            if (len > 0) swaps += len - 1;
        }

        return (swaps % 2 == 1) ? -det : det;
    }

    /// <summary>
    /// Computes trace (sum of diagonal elements) of a 2D matrix.
    /// </summary>
    public static T Trace<T>(NDArray<T> a) where T : unmanaged, INumber<T>
    {
        if (a.Rank != 2) throw new ArgumentException("Trace requires a 2D matrix.");
        int min = Math.Min(a.Shape[0], a.Shape[1]);
        T sum = T.Zero;
        for (int i = 0; i < min; i++) sum += a[i, i];
        return sum;
    }

    /// <summary>
    /// Computes the Real Schur Decomposition of a square matrix A = Q * T * Q^T.
    /// Q is an orthogonal matrix and T is quasi-upper triangular.
    /// </summary>
    /// <returns>(Q: Orthogonal matrix, T: Quasi-triangular Schur matrix)</returns>
    public static (NDArray<T> Q, NDArray<T> TMat) Schur<T>(NDArray<T> a, int maxIter = 300)
        where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1])
        {
            throw new ArgumentException("Schur decomposition requires a square 2D matrix.");
        }

        int n = a.Shape[0];
        if (n == 1)
        {
            return (NDArray<T>.Eye(1), a.Clone());
        }

        // 1. Hessenberg reduction: A = Q0 * H * Q0^T
        var (h, q) = ReduceToHessenberg(a);

        // 2. Francis double-shift / shifted QR iteration on Hessenberg matrix H
        T eps = T.CreateChecked(1e-14);
        int iter = 0;
        int m = n;

        while (m > 1 && iter < maxIter)
        {
            iter++;

            // Deflation check
            int l = m - 1;
            while (l > 0)
            {
                if (T.Abs(h[l, l - 1]) <= eps * (T.Abs(h[l - 1, l - 1]) + T.Abs(h[l, l])))
                {
                    h[l, l - 1] = T.Zero;
                    break;
                }
                l--;
            }

            if (l == m - 1)
            {
                // Single eigenvalue deflated
                m--;
                continue;
            }
            if (l == m - 2)
            {
                // 2x2 block deflated
                m -= 2;
                continue;
            }

            // Rayleigh / Wilkinson shift from trailing 2x2 block
            T a11 = h[m - 2, m - 2];
            T a12 = h[m - 2, m - 1];
            T a21 = h[m - 1, m - 2];
            T a22 = h[m - 1, m - 1];

            T tr = a11 + a22;
            T det = a11 * a22 - a12 * a21;
            T disc = tr * tr - T.CreateChecked(4.0) * det;

            T shift;
            if (disc >= T.Zero)
            {
                T sqrtDisc = T.Sqrt(disc);
                T s1 = (tr + sqrtDisc) * T.CreateChecked(0.5);
                T s2 = (tr - sqrtDisc) * T.CreateChecked(0.5);
                shift = (T.Abs(s1 - a22) < T.Abs(s2 - a22)) ? s1 : s2;
            }
            else
            {
                shift = tr * T.CreateChecked(0.5);
            }

            // QR step on active submatrix h[l..m-1, l..m-1] with shift
            int subDim = m - l;
            var subH = new NDArray<T>(subDim, subDim);
            for (int r = 0; r < subDim; r++)
            {
                for (int c = 0; c < subDim; c++)
                {
                    subH[r, c] = h[l + r, l + c];
                    if (r == c) subH[r, c] -= shift;
                }
            }

            var (subQ, subR) = QR(subH);
            var subHNext = MatrixMultiplication.MatMul(subR, subQ);

            for (int r = 0; r < subDim; r++)
            {
                for (int c = 0; c < subDim; c++)
                {
                    h[l + r, l + c] = subHNext[r, c] + (r == c ? shift : T.Zero);
                }
            }

            // Update remaining rows and columns outside the submatrix
            // Update left columns
            for (int r = 0; r < l; r++)
            {
                for (int c = 0; c < subDim; c++)
                {
                    T sum = T.Zero;
                    for (int k = 0; k < subDim; k++) sum += h[r, l + k] * subQ[k, c];
                    // Temporary buffer applied after
                }
            }

            // Accumulate into Q: Q[:, l..m-1] = Q[:, l..m-1] * subQ
            var qSub = new NDArray<T>(n, subDim);
            for (int r = 0; r < n; r++)
            {
                for (int c = 0; c < subDim; c++)
                {
                    T sum = T.Zero;
                    for (int k = 0; k < subDim; k++) sum += q[r, l + k] * subQ[k, c];
                    qSub[r, c] = sum;
                }
            }
            for (int r = 0; r < n; r++)
            {
                for (int c = 0; c < subDim; c++) q[r, l + c] = qSub[r, c];
            }
        }

        // Clean small sub-diagonals
        for (int i = 2; i < n; i++)
        {
            for (int j = 0; j < i - 1; j++) h[i, j] = T.Zero;
        }

        return (q, h);
    }

    private static (NDArray<T> H, NDArray<T> Q) ReduceToHessenberg<T>(NDArray<T> a)
        where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        int n = a.Shape[0];
        var h = a.Clone();
        var q = NDArray<T>.Eye(n);

        for (int k = 0; k < n - 2; k++)
        {
            int m = n - k - 1;
            T normX2 = T.Zero;
            for (int i = k + 1; i < n; i++) normX2 += h[i, k] * h[i, k];
            T normX = T.Sqrt(normX2);
            if (normX < T.CreateChecked(1e-15)) continue;

            T alpha = (h[k + 1, k] >= T.Zero) ? -normX : normX;
            T u1 = h[k + 1, k] - alpha;

            T[] v = new T[m];
            v[0] = T.One;
            for (int i = 1; i < m; i++) v[i] = h[k + 1 + i, k] / u1;

            T beta = -u1 / alpha;

            // Apply Householder from left: H = (I - beta * v * v^T) * H
            for (int j = k; j < n; j++)
            {
                T dot = T.Zero;
                for (int i = 0; i < m; i++) dot += v[i] * h[k + 1 + i, j];
                T factor = beta * dot;
                for (int i = 0; i < m; i++) h[k + 1 + i, j] -= factor * v[i];
            }

            // Apply Householder from right: H = H * (I - beta * v * v^T)
            for (int i = 0; i < n; i++)
            {
                T dot = T.Zero;
                for (int j = 0; j < m; j++) dot += v[j] * h[i, k + 1 + j];
                T factor = beta * dot;
                for (int j = 0; j < m; j++) h[i, k + 1 + j] -= factor * v[j];
            }

            // Accumulate into Q: Q = Q * (I - beta * v * v^T)
            for (int i = 0; i < n; i++)
            {
                T dot = T.Zero;
                for (int j = 0; j < m; j++) dot += q[i, k + 1 + j] * v[j];
                T factor = beta * dot;
                for (int j = 0; j < m; j++) q[i, k + 1 + j] -= factor * v[j];
            }
        }

        return (h, q);
    }

    /// <summary>
    /// Solves the Sylvester equation: A * X + X * B = C.
    /// Uses vectorized Kronecker transformation: (I_n (x) A + B^T (x) I_m) * vec(X) = vec(C).
    /// </summary>
    /// <param name="a">Matrix A of shape [m, m]</param>
    /// <param name="b">Matrix B of shape [n, n]</param>
    /// <param name="c">Matrix C of shape [m, n]</param>
    /// <returns>Solution matrix X of shape [m, n]</returns>
    public static NDArray<T> SolveSylvester<T>(NDArray<T> a, NDArray<T> b, NDArray<T> c)
        where T : unmanaged, IFloatingPoint<T>
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1])
            throw new ArgumentException("Matrix A must be square.");
        if (b.Rank != 2 || b.Shape[0] != b.Shape[1])
            throw new ArgumentException("Matrix B must be square.");
        if (c.Rank != 2 || c.Shape[0] != a.Shape[0] || c.Shape[1] != b.Shape[0])
            throw new ArgumentException("Matrix C must have shape [m, n] matching A [m,m] and B [n,n].");

        int m = a.Shape[0];
        int n = b.Shape[0];
        int mn = m * n;

        // Form Kronecker system: M = (I_n (x) A) + (B^T (x) I_m)
        var inMat = NDArray<T>.Eye(n);
        var imMat = NDArray<T>.Eye(m);
        var bT = b.MatrixTranspose();

        var inKronA = MatrixOps.Kron(inMat, a);
        var btKronIm = MatrixOps.Kron(bT, imMat);
        var M = inKronA + btKronIm;

        // vec(C): column-major vectorization of C
        var vecC = new NDArray<T>(mn, 1);
        for (int col = 0; col < n; col++)
        {
            for (int row = 0; row < m; row++)
            {
                vecC[col * m + row, 0] = c[row, col];
            }
        }

        // Solve M * vec(X) = vec(C)
        var vecX = Solve(M, vecC);

        // Unflatten vec(X) back into [m, n] matrix X
        var x = new NDArray<T>(m, n);
        for (int col = 0; col < n; col++)
        {
            for (int row = 0; row < m; row++)
            {
                x[row, col] = vecX[col * m + row, 0];
            }
        }

        return x;
    }
}

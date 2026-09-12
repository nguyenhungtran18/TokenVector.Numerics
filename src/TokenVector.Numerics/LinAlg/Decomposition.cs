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
}

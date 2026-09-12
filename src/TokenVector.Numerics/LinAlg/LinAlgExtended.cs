using System;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.LinAlg;

/// <summary>
/// Extended Linear Algebra routines: Pseudoinverse (PInv), Matrix Rank, Least Squares (LstSq), and Condition Number.
/// </summary>
public static class LinAlgExtended
{
    /// <summary>
    /// Computes the Moore-Penrose Pseudoinverse of a matrix using SVD: A^+ = V * Sigma^+ * U^T.
    /// </summary>
    public static NDArray<T> PInv<T>(NDArray<T> a, T? rcond = null) where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        if (a.Rank != 2) throw new ArgumentException("PInv requires a 2D matrix.");

        var (u, s, vt) = SVD.Decompose(a);
        int m = a.Shape[0];
        int n = a.Shape[1];
        int k = s.Shape[0];

        T maxS = s[0];
        T cutoff = rcond ?? (maxS * T.CreateChecked(Math.Max(m, n)) * T.CreateChecked(1e-15));

        // Construct Sigma^+ of shape [k, k] for thin SVD
        var sigmaPlus = new NDArray<T>(k, k);
        for (int i = 0; i < k; i++)
        {
            if (s[i] > cutoff)
            {
                sigmaPlus[i, i] = T.One / s[i];
            }
        }

        // A^+ = V * Sigma^+ * U^T (shape [n, m])
        var v = vt.MatrixTranspose(); // [n, k]
        var uT = u.MatrixTranspose(); // [k, m]

        var vSigma = MatrixMultiplication.MatMul(v, sigmaPlus); // [n, k]
        return MatrixMultiplication.MatMul(vSigma, uT); // [n, m]
    }

    /// <summary>
    /// Computes the numerical rank of a matrix using SVD.
    /// </summary>
    public static int MatrixRank<T>(NDArray<T> a, T? tol = null) where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        if (a.Rank != 2) throw new ArgumentException("MatrixRank requires a 2D matrix.");

        var (_, s, _) = SVD.Decompose(a);
        int m = a.Shape[0];
        int n = a.Shape[1];

        T cutoff = tol ?? (s[0] * T.CreateChecked(Math.Max(m, n)) * T.CreateChecked(1e-15));
        int rank = 0;

        foreach (var sigma in s)
        {
            if (sigma > cutoff) rank++;
        }

        return rank;
    }

    /// <summary>
    /// Computes the minimum-norm least-squares solution to linear equation Ax = b via Pseudoinverse.
    /// Returns 1D vector if b is 1D, or 2D matrix if b is 2D.
    /// </summary>
    public static NDArray<T> LstSq<T>(NDArray<T> a, NDArray<T> b) where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        var pinvA = PInv(a);
        if (b.Rank == 1)
        {
            var sol2D = MatrixMultiplication.MatMul(pinvA, b.Reshape(b.Shape[0], 1));
            return sol2D.Reshape(a.Shape[1]);
        }
        return MatrixMultiplication.MatMul(pinvA, b);
    }

    /// <summary>
    /// Computes 2-norm condition number of matrix: kappa(A) = sigma_max / sigma_min.
    /// </summary>
    public static T Cond<T>(NDArray<T> a) where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        var (_, s, _) = SVD.Decompose(a);
        T maxS = s[0];
        T minS = s[s.TotalLength - 1];

        if (minS <= T.CreateChecked(1e-15)) return T.CreateChecked(double.PositiveInfinity);
        return maxS / minS;
    }

    /// <summary>
    /// Solves tridiagonal linear system A x = d in O(N) using the Thomas Algorithm.
    /// lower: subdiagonal [n-1], diag: main diagonal [n], upper: superdiagonal [n-1], rhs: right-hand side [n].
    /// </summary>
    public static NDArray<double> SolveBandTridiagonal(
        NDArray<double> lower, NDArray<double> diag, NDArray<double> upper, NDArray<double> rhs)
    {
        int n = diag.TotalLength;
        double[] cPrime = new double[n - 1];
        double[] dPrime = new double[n];

        cPrime[0] = upper[0] / diag[0];
        dPrime[0] = rhs[0] / diag[0];

        for (int i = 1; i < n - 1; i++)
        {
            double m = 1.0 / (diag[i] - lower[i - 1] * cPrime[i - 1]);
            cPrime[i] = upper[i] * m;
            dPrime[i] = (rhs[i] - lower[i - 1] * dPrime[i - 1]) * m;
        }

        dPrime[n - 1] = (rhs[n - 1] - lower[n - 2] * dPrime[n - 2]) / (diag[n - 1] - lower[n - 2] * cPrime[n - 2]);

        var x = new NDArray<double>(n);
        x[n - 1] = dPrime[n - 1];
        for (int i = n - 2; i >= 0; i--)
        {
            x[i] = dPrime[i] - cPrime[i] * x[i + 1];
        }

        return x;
    }

    /// <summary>
    /// Computes the orthonormal basis for the Null Space (Kernel) of matrix A using SVD.
    /// </summary>
    public static NDArray<T> NullSpace<T>(NDArray<T> a, T? rcond = null) where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        var (_, s, vt) = SVD.Decompose(a);
        int n = a.Shape[1];
        int k = s.Shape[0];

        T maxS = s[0];
        T cutoff = rcond ?? (maxS * T.CreateChecked(Math.Max(a.Shape[0], n)) * T.CreateChecked(1e-15));

        int rank = 0;
        for (int i = 0; i < k; i++) if (s[i] > cutoff) rank++;

        int nullDim = n - rank;
        if (nullDim <= 0) return new NDArray<T>(n, 0);

        var nullBasis = new NDArray<T>(n, nullDim);
        var v = vt.MatrixTranspose(); // [n, k]

        for (int col = 0; col < nullDim; col++)
        {
            int vCol = rank + col;
            for (int r = 0; r < n; r++)
            {
                nullBasis[r, col] = (vCol < v.Shape[1]) ? v[r, vCol] : T.Zero;
            }
        }

        return nullBasis;
    }

    /// <summary>
    /// Computes Kronecker Sum: A (+) B = A (x) I_b + I_a (x) B.
    /// </summary>
    public static NDArray<T> KroneckerSum<T>(NDArray<T> a, NDArray<T> b) where T : unmanaged, INumber<T>
    {
        int na = a.Shape[0];
        int nb = b.Shape[0];
        var ia = NDArray<T>.Eye(na);
        var ib = NDArray<T>.Eye(nb);

        var aKronIb = MatrixOps.Kron(a, ib);
        var iaKronB = MatrixOps.Kron(ia, b);

        return aKronIb + iaKronB;
    }
}

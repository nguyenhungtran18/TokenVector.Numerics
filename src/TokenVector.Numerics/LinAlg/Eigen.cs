using System;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.LinAlg;

/// <summary>
/// Spectral theory engine computing Eigenvalues and Eigenvectors (Eigh for symmetric matrices and Eig for general matrices).
/// </summary>
public static class Eigen
{
    private const int MaxIterations = 100;

    /// <summary>
    /// Computes eigenvalues and eigenvectors of a real symmetric matrix A using the Jacobi eigenvalue algorithm.
    /// A * v_i = lambda_i * v_i.
    /// </summary>
    /// <returns>(Eigenvalues: 1D array, Eigenvectors: Column-matrix V where column i is eigenvector i)</returns>
    public static (NDArray<T> Eigenvalues, NDArray<T> Eigenvectors) Eigh<T>(NDArray<T> a)
        where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1])
        {
            throw new ArgumentException("Eigh requires a square 2D matrix.");
        }

        int n = a.Shape[0];
        var d = a.Clone();
        var v = NDArray<T>.Eye(n);
        T eps = T.CreateChecked(1e-15);

        for (int iter = 0; iter < MaxIterations; iter++)
        {
            // Find off-diagonal element with maximum absolute value
            T maxOffDiag = T.Zero;
            int p = 0, q = 1;

            for (int i = 0; i < n - 1; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    T val = T.Abs(d[i, j]);
                    if (val > maxOffDiag)
                    {
                        maxOffDiag = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if (maxOffDiag < eps) break;

            // Compute Jacobi rotation angle
            T dPP = d[p, p];
            T dQQ = d[q, q];
            T dPQ = d[p, q];

            T theta = (dQQ - dPP) / (T.CreateChecked(2.0) * dPQ);
            T t = (theta >= T.Zero ? T.One : -T.One) / (T.Abs(theta) + T.Sqrt(T.One + theta * theta));
            T c = T.One / T.Sqrt(T.One + t * t);
            T s = t * c;
            T tau = s / (T.One + c);

            // Update diagonal elements
            d[p, p] = dPP - t * dPQ;
            d[q, q] = dQQ + t * dPQ;
            d[p, q] = T.Zero;
            d[q, p] = T.Zero;

            // Update off-diagonal elements
            for (int k = 0; k < n; k++)
            {
                if (k != p && k != q)
                {
                    T dKP = d[k, p];
                    T dKQ = d[k, q];

                    d[k, p] = dKP - s * (dKQ + tau * dKP);
                    d[p, k] = d[k, p];

                    d[k, q] = dKQ + s * (dKP - tau * dKQ);
                    d[q, k] = d[k, q];
                }
            }

            // Update eigenvector matrix V
            for (int k = 0; k < n; k++)
            {
                T vKP = v[k, p];
                T vKQ = v[k, q];

                v[k, p] = c * vKP - s * vKQ;
                v[k, q] = s * vKP + c * vKQ;
            }
        }

        // Extract eigenvalues from diagonal
        var eigvals = new NDArray<T>(n);
        for (int i = 0; i < n; i++) eigvals[i] = d[i, i];

        // Sort eigenvalues in ascending order
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (i1, i2) => eigvals[i1].CompareTo(eigvals[i2]));

        var sortedEigvals = new NDArray<T>(n);
        var sortedV = new NDArray<T>(n, n);

        for (int j = 0; j < n; j++)
        {
            int srcIdx = order[j];
            sortedEigvals[j] = eigvals[srcIdx];
            for (int i = 0; i < n; i++) sortedV[i, j] = v[i, srcIdx];
        }

        return (sortedEigvals, sortedV);
    }
}

using System;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.LinAlg;

/// <summary>
/// Singular Value Decomposition (SVD) engine computing A = U * S * V^T for arbitrary rectangular matrices.
/// </summary>
public static class SVD
{
    private const int MaxSweeps = 100;

    /// <summary>
    /// Computes Singular Value Decomposition: A = U * diag(S) * V^T.
    /// </summary>
    /// <returns>(U: Left singular vectors, S: 1D Singular values, Vt: Right singular vectors transposed)</returns>
    public static (NDArray<T> U, NDArray<T> S, NDArray<T> Vt) Decompose<T>(NDArray<T> a)
        where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        if (a.Rank != 2) throw new ArgumentException("SVD requires a 2D matrix.");

        int m = a.Shape[0];
        int n = a.Shape[1];

        if (m >= n)
        {
            return ComputeOneSidedJacobi(a, m, n);
        }
        else
        {
            // For wide matrices (m < n), compute SVD of A^T: A^T = V * S * U^T -> A = U * S * V^T
            var aT = a.MatrixTranspose();
            var (v, s, ut) = ComputeOneSidedJacobi(aT, n, m);
            return (ut.MatrixTranspose(), s, v.MatrixTranspose());
        }
    }

    private static (NDArray<T> U, NDArray<T> S, NDArray<T> Vt) ComputeOneSidedJacobi<T>(NDArray<T> a, int m, int n)
        where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        var b = a.Clone();
        var v = NDArray<T>.Eye(n);
        T eps = T.CreateChecked(1e-15);

        for (int sweep = 0; sweep < MaxSweeps; sweep++)
        {
            T maxGamma = T.Zero;

            for (int j = 0; j < n - 1; j++)
            {
                for (int k = j + 1; k < n; k++)
                {
                    T alpha = T.Zero;
                    T beta = T.Zero;
                    T gamma = T.Zero;

                    for (int i = 0; i < m; i++)
                    {
                        T bj = b[i, j];
                        T bk = b[i, k];
                        alpha += bj * bj;
                        beta += bk * bk;
                        gamma += bj * bk;
                    }

                    if (T.Abs(gamma) > maxGamma) maxGamma = T.Abs(gamma);

                    if (T.Abs(gamma) < eps * T.Sqrt(alpha * beta + eps)) continue;

                    T zeta = (beta - alpha) / (T.CreateChecked(2.0) * gamma);
                    T t = (zeta >= T.Zero ? T.One : -T.One) / (T.Abs(zeta) + T.Sqrt(T.One + zeta * zeta));
                    T c = T.One / T.Sqrt(T.One + t * t);
                    T s = c * t;

                    // Rotate columns j and k of B
                    for (int i = 0; i < m; i++)
                    {
                        T bj = b[i, j];
                        T bk = b[i, k];
                        b[i, j] = c * bj - s * bk;
                        b[i, k] = s * bj + c * bk;
                    }

                    // Rotate columns j and k of V
                    for (int i = 0; i < n; i++)
                    {
                        T vj = v[i, j];
                        T vk = v[i, k];
                        v[i, j] = c * vj - s * vk;
                        v[i, k] = s * vj + c * vk;
                    }
                }
            }

            if (maxGamma < eps) break;
        }

        // Compute singular values and normalize columns of B to form U
        var sVec = new NDArray<T>(n);
        var u = new NDArray<T>(m, n);

        for (int j = 0; j < n; j++)
        {
            T normSq = T.Zero;
            for (int i = 0; i < m; i++) normSq += b[i, j] * b[i, j];
            T sigma = T.Sqrt(normSq);
            sVec[j] = sigma;

            if (sigma > eps)
            {
                for (int i = 0; i < m; i++) u[i, j] = b[i, j] / sigma;
            }
            else
            {
                for (int i = 0; i < m; i++) u[i, j] = T.Zero;
            }
        }

        // Sort singular values in descending order
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (i1, i2) => sVec[i2].CompareTo(sVec[i1]));

        var sortedS = new NDArray<T>(n);
        var sortedU = new NDArray<T>(m, n);
        var sortedV = new NDArray<T>(n, n);

        for (int j = 0; j < n; j++)
        {
            int srcIdx = order[j];
            sortedS[j] = sVec[srcIdx];
            for (int i = 0; i < m; i++) sortedU[i, j] = u[i, srcIdx];
            for (int i = 0; i < n; i++) sortedV[i, j] = v[i, srcIdx];
        }

        return (sortedU, sortedS, sortedV.MatrixTranspose());
    }
}

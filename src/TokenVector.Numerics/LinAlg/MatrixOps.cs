// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;
using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.LinAlg;

/// <summary>
/// Provides advanced matrix operations: Kronecker product (Kron), Outer product (Outer), Matrix Power (MatrixPower), and Matrix/Vector Norms (np.kron, np.outer, np.linalg.matrix_power, np.linalg.norm).
/// </summary>
public static class MatrixOps
{
    /// <summary>
    /// Computes the Kronecker product of two arrays (np.kron).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Kron<T>(NDArray<T> a, NDArray<T> b) where T : unmanaged, INumber<T>
    {
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("Kron currently expects 2D matrices.");

        int ra = a.Shape[0], ca = a.Shape[1];
        int rb = b.Shape[0], cb = b.Shape[1];

        int outR = ra * rb;
        int outC = ca * cb;

        var result = new NDArray<T>(outR, outC);
        var aContig = a.Contiguous();
        var bContig = b.Contiguous();

        Parallel.For(0, ra, i =>
        {
            for (int j = 0; j < ca; j++)
            {
                T aVal = aContig[i * ca + j];
                for (int k = 0; k < rb; k++)
                {
                    for (int l = 0; l < cb; l++)
                    {
                        T bVal = bContig[k * cb + l];
                        result[(i * rb + k) * outC + (j * cb + l)] = aVal * bVal;
                    }
                }
            }
        });

        return result;
    }

    /// <summary>
    /// Computes the outer product of two vectors (np.outer).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Outer<T>(NDArray<T> a, NDArray<T> b) where T : unmanaged, INumber<T>
    {
        var aFlat = a.Contiguous();
        var bFlat = b.Contiguous();

        int lenA = aFlat.TotalLength;
        int lenB = bFlat.TotalLength;

        var result = new NDArray<T>(lenA, lenB);

        Parallel.For(0, lenA, i =>
        {
            T ai = aFlat[i];
            for (int j = 0; j < lenB; j++)
            {
                result[i * lenB + j] = ai * bFlat[j];
            }
        });

        return result;
    }

    /// <summary>
    /// Raises a square matrix to the (integer) power n (np.linalg.matrix_power).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> MatrixPower<T>(NDArray<T> a, int n) where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1])
            throw new ArgumentException("MatrixPower requires a square 2D matrix.");

        int dim = a.Shape[0];
        if (n == 0)
        {
            return NDArray<T>.Eye(dim);
        }

        NDArray<T> baseMat;
        int exp;

        if (n < 0)
        {
            baseMat = Decomposition.Inverse(a);
            exp = -n;
        }
        else
        {
            baseMat = a.Clone();
            exp = n;
        }

        var result = NDArray<T>.Eye(dim);
        while (exp > 0)
        {
            if ((exp & 1) == 1)
            {
                result = MatrixMultiplication.MatMul(result, baseMat);
            }
            baseMat = MatrixMultiplication.MatMul(baseMat, baseMat);
            exp >>= 1;
        }

        return result;
    }

    /// <summary>
    /// Computes matrix or vector norm (np.linalg.norm).
    /// Supported ord: "fro" (Frobenius), "1" (L1 norm), "2" (L2 norm), "inf" (Max norm).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static double Norm<T>(NDArray<T> x, string ord = "fro") where T : unmanaged, INumber<T>
    {
        var contig = x.Contiguous();
        int total = contig.TotalLength;

        switch (ord.ToLowerInvariant())
        {
            case "fro":
            case "2":
                double sumSq = 0.0;
                for (int i = 0; i < total; i++)
                {
                    double v = double.CreateTruncating(contig[i]);
                    sumSq += v * v;
                }
                return Math.Sqrt(sumSq);

            case "1":
                double sumAbs = 0.0;
                for (int i = 0; i < total; i++)
                {
                    double v = double.CreateTruncating(contig[i]);
                    sumAbs += Math.Abs(v);
                }
                return sumAbs;

            case "inf":
                double maxAbs = 0.0;
                for (int i = 0; i < total; i++)
                {
                    double v = double.CreateTruncating(contig[i]);
                    maxAbs = Math.Max(maxAbs, Math.Abs(v));
                }
                return maxAbs;

            default:
                throw new NotSupportedException($"Norm order '{ord}' is not supported.");
        }
    }
}

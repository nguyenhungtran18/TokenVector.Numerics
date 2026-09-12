// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Crypto;

/// <summary>
/// Provides post-quantum cryptography mathematics: Number Theoretic Transform (NTT), fast polynomial arithmetic, and LLL (Lenstra–Lenstra–Lovász) lattice basis reduction.
/// </summary>
public static class LatticeMath
{
    #region Number Theoretic Transform (NTT) for Post-Quantum Rings (e.g. CRYSTALS-Kyber, Dilithium)

    /// <summary>
    /// Computes in-place forward Number Theoretic Transform (NTT) on a polynomial of length N (power of 2) modulo prime q.
    /// rootOfUnity: primitive N-th root of unity modulo q (omega^N = 1 mod q).
    /// </summary>
    public static void ForwardNTT(Span<long> a, long q, long rootOfUnity)
    {
        int n = a.Length;
        BitReverse(a);

        for (int len = 2; len <= n; len <<= 1)
        {
            long wlen = ModPow(rootOfUnity, n / len, q);
            int half = len >> 1;

            for (int i = 0; i < n; i += len)
            {
                long w = 1;
                for (int j = 0; j < half; j++)
                {
                    long u = a[i + j];
                    long v = (a[i + j + half] * w) % q;

                    a[i + j] = (u + v) % q;
                    a[i + j + half] = (u - v + q) % q;

                    w = (w * wlen) % q;
                }
            }
        }
    }

    /// <summary>
    /// Computes in-place Inverse Number Theoretic Transform (INTT) modulo prime q.
    /// </summary>
    public static void InverseNTT(Span<long> a, long q, long rootOfUnity)
    {
        int n = a.Length;
        long invRoot = ModInverse(rootOfUnity, q);
        ForwardNTT(a, q, invRoot);

        long invN = ModInverse(n, q);
        for (int i = 0; i < n; i++)
        {
            a[i] = (a[i] * invN) % q;
        }
    }

    /// <summary>
    /// Computes Forward Number Theoretic Transform (NTT) on NDArray of length N.
    /// </summary>
    public static NDArray<long> ForwardNTT(NDArray<long> a, long q, long rootOfUnity)
    {
        var res = a.Clone();
        ForwardNTT(res.Buffer.AsSpan(), q, rootOfUnity);
        return res;
    }

    /// <summary>
    /// Computes Inverse Number Theoretic Transform (INTT) on NDArray of length N.
    /// </summary>
    public static NDArray<long> InverseNTT(NDArray<long> a, long q, long rootOfUnity)
    {
        var res = a.Clone();
        InverseNTT(res.Buffer.AsSpan(), q, rootOfUnity);
        return res;
    }

    /// <summary>
    /// Fast cyclic/negacyclic polynomial multiplication in Z_q[X] / (X^N - 1) via NTT in O(N log N).
    /// </summary>
    public static NDArray<long> PolyMulNTT(NDArray<long> polyA, NDArray<long> polyB, long q, long rootOfUnity)
    {
        int n = polyA.TotalLength;
        long[] a = new long[n];
        long[] b = new long[n];

        polyA.Contiguous().AsSpan().CopyTo(a);
        polyB.Contiguous().AsSpan().CopyTo(b);

        ForwardNTT(a, q, rootOfUnity);
        ForwardNTT(b, q, rootOfUnity);

        for (int i = 0; i < n; i++)
        {
            a[i] = (a[i] * b[i]) % q;
        }

        InverseNTT(a, q, rootOfUnity);
        return NDArray<long>.FromArray(a, n);
    }

    private static void BitReverse(Span<long> data)
    {
        int n = data.Length;
        int j = 0;
        for (int i = 0; i < n - 1; i++)
        {
            if (i < j) (data[i], data[j]) = (data[j], data[i]);
            int k = n >> 1;
            while (k <= j) { j -= k; k >>= 1; }
            j += k;
        }
    }

    private static long ModPow(long baseVal, long exp, long mod)
    {
        long res = 1;
        baseVal %= mod;
        while (exp > 0)
        {
            if ((exp & 1) == 1) res = (res * baseVal) % mod;
            baseVal = (baseVal * baseVal) % mod;
            exp >>= 1;
        }
        return res;
    }

    private static long ModInverse(long a, long m)
    {
        long m0 = m;
        long y = 0, x = 1;
        if (m == 1) return 0;

        while (a > 1)
        {
            long q = a / m;
            long t = m;
            m = a % m;
            a = t;
            t = y;
            y = x - q * y;
            x = t;
        }
        if (x < 0) x += m0;
        return x;
    }

    #endregion

    #region LLL (Lenstra–Lenstra–Lovász) Lattice Reduction

    /// <summary>
    /// Reduces a lattice basis matrix B [n, d] using the LLL algorithm with Lovász parameter delta (typically 0.75).
    /// </summary>
    public static NDArray<double> LLLReduction(NDArray<double> basis, double delta = 0.75) => LLL_Reduction(basis, delta);

    /// <summary>
    /// Reduces a lattice basis matrix B [n, d] using the LLL algorithm with Lovász parameter delta (typically 0.75).
    /// Returns the reduced basis matrix.
    /// </summary>
    public static NDArray<double> LLL_Reduction(NDArray<double> basis, double delta = 0.75)
    {
        int n = basis.Shape[0];
        int d = basis.Shape[1];

        var b = basis.Clone();
        var bStar = new NDArray<double>(n, d);
        var mu = new NDArray<double>(n, n);

        // Gram-Schmidt orthogonalization
        void UpdateGramSchmidt()
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < d; j++) bStar[i, j] = b[i, j];
                for (int j = 0; j < i; j++)
                {
                    double bStarJNormSq = DotRow(bStar, j, bStar, j, d);
                    double dot = DotRow(b, i, bStar, j, d);
                    mu[i, j] = (bStarJNormSq > 1e-15) ? (dot / bStarJNormSq) : 0.0;
                    for (int k = 0; k < d; k++)
                    {
                        bStar[i, k] -= mu[i, j] * bStar[j, k];
                    }
                }
            }
        }

        UpdateGramSchmidt();
        int kIdx = 1;

        while (kIdx < n)
        {
            // Size reduction condition
            for (int j = kIdx - 1; j >= 0; j--)
            {
                if (Math.Abs(mu[kIdx, j]) > 0.5)
                {
                    double r = Math.Round(mu[kIdx, j]);
                    for (int col = 0; col < d; col++)
                    {
                        b[kIdx, col] -= r * b[j, col];
                    }
                    UpdateGramSchmidt();
                }
            }

            // Lovász condition: ||b^*_k + mu_{k, k-1} * b^*_{k-1}||^2 >= delta * ||b^*_{k-1}||^2
            double bStarKNormSq = DotRow(bStar, kIdx, bStar, kIdx, d);
            double bStarKMinus1NormSq = DotRow(bStar, kIdx - 1, bStar, kIdx - 1, d);
            double muVal = mu[kIdx, kIdx - 1];

            if (bStarKNormSq >= (delta - muVal * muVal) * bStarKMinus1NormSq)
            {
                kIdx++;
            }
            else
            {
                // Swap basis vectors b[kIdx] and b[kIdx-1]
                for (int col = 0; col < d; col++)
                {
                    double tmp = b[kIdx, col];
                    b[kIdx, col] = b[kIdx - 1, col];
                    b[kIdx - 1, col] = tmp;
                }
                UpdateGramSchmidt();
                kIdx = Math.Max(1, kIdx - 1);
            }
        }

        return b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double DotRow(NDArray<double> matA, int rowA, NDArray<double> matB, int rowB, int cols)
    {
        double sum = 0.0;
        for (int c = 0; c < cols; c++) sum += matA[rowA, c] * matB[rowB, c];
        return sum;
    }

    #endregion
}

// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;
using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.LinAlg;

public enum ConvMode
{
    Full,
    Same,
    Valid
}

/// <summary>
/// Provides 1D discrete convolution, correlation, and DSP window functions (np.convolve, np.correlate, np.hanning, np.hamming, np.blackman, np.bartlett).
/// </summary>
public static class SignalProcessing
{
    /// <summary>
    /// Computes discrete 1D linear convolution of two 1D sequences (np.convolve).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Convolve<T>(NDArray<T> a, NDArray<T> v, ConvMode mode = ConvMode.Full) where T : unmanaged, INumber<T>
    {
        if (a.Rank != 1 || v.Rank != 1)
            throw new ArgumentException("Convolve requires 1D input tensors.");

        int na = a.Shape[0];
        int nv = v.Shape[0];
        if (na == 0 || nv == 0) return new NDArray<T>(0);

        var aContig = a.Contiguous();
        var vContig = v.Contiguous();

        int nFull = na + nv - 1;
        var full = new NDArray<T>(nFull);

        Parallel.For(0, nFull, k =>
        {
            T sum = T.Zero;
            int minJ = Math.Max(0, k - nv + 1);
            int maxJ = Math.Min(na - 1, k);

            for (int j = minJ; j <= maxJ; j++)
            {
                sum += aContig[j] * vContig[k - j];
            }
            full[k] = sum;
        });

        return SliceConvMode(full, na, nv, mode);
    }

    /// <summary>
    /// Computes discrete 1D cross-correlation of two 1D sequences (np.correlate).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Correlate<T>(NDArray<T> a, NDArray<T> v, ConvMode mode = ConvMode.Full) where T : unmanaged, INumber<T>
    {
        if (a.Rank != 1 || v.Rank != 1)
            throw new ArgumentException("Correlate requires 1D input tensors.");

        int na = a.Shape[0];
        int nv = v.Shape[0];
        if (na == 0 || nv == 0) return new NDArray<T>(0);

        var aContig = a.Contiguous();
        var vContig = v.Contiguous();

        // Correlation is convolution with v reversed and conjugated
        var vRev = new NDArray<T>(nv);
        for (int i = 0; i < nv; i++)
        {
            vRev[i] = vContig[nv - 1 - i];
        }

        return Convolve(aContig, vRev, mode);
    }

    private static NDArray<T> SliceConvMode<T>(NDArray<T> full, int na, int nv, ConvMode mode) where T : unmanaged, INumber<T>
    {
        switch (mode)
        {
            case ConvMode.Full:
                return full;

            case ConvMode.Same:
                int sameLen = Math.Max(na, nv);
                int startSame = (full.TotalLength - sameLen) / 2;
                var same = new NDArray<T>(sameLen);
                for (int i = 0; i < sameLen; i++)
                {
                    same[i] = full[startSame + i];
                }
                return same;

            case ConvMode.Valid:
                int validLen = Math.Max(0, Math.Max(na, nv) - Math.Min(na, nv) + 1);
                if (validLen == 0) return new NDArray<T>(0);
                int startValid = Math.Min(na, nv) - 1;
                var valid = new NDArray<T>(validLen);
                for (int i = 0; i < validLen; i++)
                {
                    valid[i] = full[startValid + i];
                }
                return valid;

            default:
                throw new ArgumentOutOfRangeException(nameof(mode));
        }
    }

    /// <summary>
    /// Computes the Hanning window: w(n) = 0.5 - 0.5 * cos(2*pi*n / (M - 1))
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<double> Hanning(int m)
    {
        if (m <= 0) return new NDArray<double>(0);
        if (m == 1) return NDArray<double>.FromArray([1.0], 1);

        var w = new NDArray<double>(m);
        double factor = 2.0 * Math.PI / (m - 1);
        for (int n = 0; n < m; n++)
        {
            w[n] = 0.5 - 0.5 * Math.Cos(factor * n);
        }
        return w;
    }

    /// <summary>
    /// Computes the Hamming window: w(n) = 0.54 - 0.46 * cos(2*pi*n / (M - 1))
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<double> Hamming(int m)
    {
        if (m <= 0) return new NDArray<double>(0);
        if (m == 1) return NDArray<double>.FromArray([1.0], 1);

        var w = new NDArray<double>(m);
        double factor = 2.0 * Math.PI / (m - 1);
        for (int n = 0; n < m; n++)
        {
            w[n] = 0.54 - 0.46 * Math.Cos(factor * n);
        }
        return w;
    }

    /// <summary>
    /// Computes the Blackman window: w(n) = 0.42 - 0.5*cos(2*pi*n/(M-1)) + 0.08*cos(4*pi*n/(M-1))
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<double> Blackman(int m)
    {
        if (m <= 0) return new NDArray<double>(0);
        if (m == 1) return NDArray<double>.FromArray([1.0], 1);

        var w = new NDArray<double>(m);
        double factor = 2.0 * Math.PI / (m - 1);
        for (int n = 0; n < m; n++)
        {
            w[n] = 0.42 - 0.5 * Math.Cos(factor * n) + 0.08 * Math.Cos(2.0 * factor * n);
        }
        return w;
    }

    /// <summary>
    /// Computes the Bartlett triangular window.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<double> Bartlett(int m)
    {
        if (m <= 0) return new NDArray<double>(0);
        if (m == 1) return NDArray<double>.FromArray([1.0], 1);

        var w = new NDArray<double>(m);
        double halfM = (m - 1) / 2.0;
        for (int n = 0; n < m; n++)
        {
            w[n] = 1.0 - Math.Abs((n - halfM) / halfM);
        }
        return w;
    }
}

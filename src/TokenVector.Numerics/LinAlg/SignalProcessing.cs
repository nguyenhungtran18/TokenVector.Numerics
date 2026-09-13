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

    #region Wavelet Transforms (DWT, IDWT)

    private static readonly double Sqrt2 = Math.Sqrt(2.0);
    private static readonly double InvSqrt2 = 1.0 / Math.Sqrt(2.0);

    // Daubechies 4 (db4) coefficients
    private static readonly double Db4H0 = (1.0 + Math.Sqrt(3.0)) / (4.0 * Math.Sqrt(2.0));
    private static readonly double Db4H1 = (3.0 + Math.Sqrt(3.0)) / (4.0 * Math.Sqrt(2.0));
    private static readonly double Db4H2 = (3.0 - Math.Sqrt(3.0)) / (4.0 * Math.Sqrt(2.0));
    private static readonly double Db4H3 = (1.0 - Math.Sqrt(3.0)) / (4.0 * Math.Sqrt(2.0));

    /// <summary>
    /// Computes 1D Single-level Discrete Wavelet Transform (DWT).
    /// Returns (cA: Approximation coefficients, cD: Detail coefficients).
    /// Supported wavelets: "haar" (default), "db4".
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static (NDArray<double> cA, NDArray<double> cD) DWT(NDArray<double> signal, string wavelet = "haar")
    {
        if (signal.Rank != 1) throw new ArgumentException("DWT requires a 1D signal.");
        int n = signal.TotalLength;
        if (n == 0) return (new NDArray<double>(0), new NDArray<double>(0));

        // Pad if odd
        int padN = (n % 2 == 0) ? n : n + 1;
        var padded = new double[padN];
        for (int i = 0; i < n; i++) padded[i] = signal[i];
        if (n < padN) padded[n] = signal[n - 1]; // Periodic / border repeat

        int half = padN / 2;
        var cA = new NDArray<double>(half);
        var cD = new NDArray<double>(half);

        string wl = wavelet.ToLowerInvariant();
        if (wl is "haar" or "db1")
        {
            Parallel.For(0, half, i =>
            {
                cA[i] = (padded[2 * i] + padded[2 * i + 1]) * InvSqrt2;
                cD[i] = (padded[2 * i] - padded[2 * i + 1]) * InvSqrt2;
            });
        }
        else if (wl is "db4" or "d4")
        {
            double[] h = [Db4H0, Db4H1, Db4H2, Db4H3];
            double[] g = [Db4H3, -Db4H2, Db4H1, -Db4H0];

            Parallel.For(0, half, i =>
            {
                double a = 0.0, d = 0.0;
                for (int j = 0; j < 4; j++)
                {
                    int idx = (2 * i + j) % padN;
                    a += h[j] * padded[idx];
                    d += g[j] * padded[idx];
                }
                cA[i] = a;
                cD[i] = d;
            });
        }
        else
        {
            throw new NotSupportedException($"Wavelet '{wavelet}' is not supported. Use 'haar' or 'db4'.");
        }

        return (cA, cD);
    }

    /// <summary>
    /// Computes 1D Inverse Discrete Wavelet Transform (IDWT) reconstructing signal from cA and cD.
    /// Supported wavelets: "haar" (default), "db4".
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<double> IDWT(NDArray<double> cA, NDArray<double> cD, string wavelet = "haar")
    {
        if (cA.Rank != 1 || cD.Rank != 1 || cA.TotalLength != cD.TotalLength)
            throw new ArgumentException("cA and cD must be 1D tensors of identical length.");

        int half = cA.TotalLength;
        int n = half * 2;
        var result = new NDArray<double>(n);

        string wl = wavelet.ToLowerInvariant();
        if (wl is "haar" or "db1")
        {
            Parallel.For(0, half, i =>
            {
                result[2 * i] = (cA[i] + cD[i]) * InvSqrt2;
                result[2 * i + 1] = (cA[i] - cD[i]) * InvSqrt2;
            });
        }
        else if (wl is "db4" or "d4")
        {
            double[] h = [Db4H0, Db4H1, Db4H2, Db4H3];
            double[] g = [Db4H3, -Db4H2, Db4H1, -Db4H0];

            for (int i = 0; i < half; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    int targetIdx = (2 * i + j) % n;
                    result[targetIdx] += h[j] * cA[i] + g[j] * cD[i];
                }
            }
        }
        else
        {
            throw new NotSupportedException($"Wavelet '{wavelet}' is not supported. Use 'haar' or 'db4'.");
        }

        return result;
    }

    #endregion

    #region Hilbert Transform & Analytic Signal

    /// <summary>
    /// Computes the Hilbert transform H[x] of a 1D real signal using FFT.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<double> Hilbert(NDArray<double> signal)
    {
        var (_, imag) = AnalyticSignal(signal);
        return imag;
    }

    /// <summary>
    /// Computes the Analytic Signal x_a(t) = x(t) + i * H[x(t)] using FFT.
    /// Returns (Real: original signal, Imag: Hilbert transform).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static (NDArray<double> Real, NDArray<double> Imag) AnalyticSignal(NDArray<double> signal)
    {
        if (signal.Rank != 1) throw new ArgumentException("AnalyticSignal requires a 1D tensor.");
        int n = signal.TotalLength;
        if (n == 0) return (new NDArray<double>(0), new NDArray<double>(0));

        var cInput = new Complex<double>[n];
        for (int i = 0; i < n; i++)
        {
            cInput[i] = new Complex<double>(signal[i], 0.0);
        }

        var fVal = FFT.FFT1D<double>(cInput);

        // Multipliers for analytic signal in frequency domain
        if (n % 2 == 0)
        {
            // Even length
            for (int i = 1; i < n / 2; i++) fVal[i] = fVal[i] * 2.0;
            for (int i = n / 2 + 1; i < n; i++) fVal[i] = Complex<double>.Zero;
        }
        else
        {
            // Odd length
            int half = (n + 1) / 2;
            for (int i = 1; i < half; i++) fVal[i] = fVal[i] * 2.0;
            for (int i = half; i < n; i++) fVal[i] = Complex<double>.Zero;
        }

        var analytic = FFT.IFFT1D<double>(fVal);

        var realPart = new NDArray<double>(n);
        var imagPart = new NDArray<double>(n);

        for (int i = 0; i < n; i++)
        {
            realPart[i] = analytic[i].Real;
            imagPart[i] = analytic[i].Imaginary;
        }

        return (realPart, imagPart);
    }

    #endregion
}

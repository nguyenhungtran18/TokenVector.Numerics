using System;
using System.Numerics;
using System.Threading.Tasks;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.LinAlg;

/// <summary>
/// Universal Fast Fourier Transform (FFT) engine supporting arbitrary signal lengths via Bluestein's Chirp-Z algorithm and Cooley-Tukey Radix-2.
/// </summary>
public static class FFT
{
    /// <summary>
    /// Computes 1D Fast Fourier Transform on an array of Complex numbers for ANY arbitrary length N.
    /// </summary>
    public static Complex<T>[] FFT1D<T>(ReadOnlySpan<Complex<T>> input) where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>
    {
        int n = input.Length;
        if (n <= 1) return input.ToArray();

        // If length is power of 2, use fast Cooley-Tukey Radix-2
        if ((n & (n - 1)) == 0)
        {
            return FFTRadix2(input);
        }

        // For arbitrary length, use Bluestein's Chirp-Z FFT algorithm
        return FFTBluestein(input);
    }

    /// <summary>
    /// Computes 1D Inverse Fast Fourier Transform for ANY arbitrary length N.
    /// </summary>
    public static Complex<T>[] IFFT1D<T>(ReadOnlySpan<Complex<T>> input) where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>
    {
        int n = input.Length;
        Complex<T>[] conj = new Complex<T>[n];
        for (int i = 0; i < n; i++) conj[i] = input[i].Conjugate();

        var transformed = FFT1D<T>(conj);
        T factor = T.One / T.CreateChecked(n);

        for (int i = 0; i < n; i++)
        {
            transformed[i] = transformed[i].Conjugate() * factor;
        }

        return transformed;
    }

    /// <summary>
    /// Computes 2D Fast Fourier Transform on a row-major 2D matrix of complex numbers for arbitrary dimensions.
    /// </summary>
    public static Complex<T>[,] FFT2D<T>(Complex<T>[,] input) where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>
    {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);

        var result = new Complex<T>[rows, cols];

        // FFT on each row
        Parallel.For(0, rows, r =>
        {
            Span<Complex<T>> rowSpan = stackalloc Complex<T>[cols];
            for (int c = 0; c < cols; c++) rowSpan[c] = input[r, c];
            var rowTransformed = FFT1D<T>(rowSpan);
            for (int c = 0; c < cols; c++) result[r, c] = rowTransformed[c];
        });

        // FFT on each column
        Parallel.For(0, cols, c =>
        {
            Span<Complex<T>> colSpan = stackalloc Complex<T>[rows];
            for (int r = 0; r < rows; r++) colSpan[r] = result[r, c];
            var colTransformed = FFT1D<T>(colSpan);
            for (int r = 0; r < rows; r++) result[r, c] = colTransformed[r];
        });

        return result;
    }

    /// <summary>
    /// Computes 2D Inverse Fast Fourier Transform for arbitrary dimensions.
    /// </summary>
    public static Complex<T>[,] IFFT2D<T>(Complex<T>[,] input) where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>
    {
        int rows = input.GetLength(0);
        int cols = input.GetLength(1);

        var conj = new Complex<T>[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                conj[r, c] = input[r, c].Conjugate();

        var transformed = FFT2D(conj);
        T factor = T.One / T.CreateChecked(rows * cols);

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                transformed[r, c] = transformed[r, c].Conjugate() * factor;

        return transformed;
    }

    /// <summary>
    /// Real-to-Complex 1D FFT from NDArray&lt;T&gt; to NDArray&lt;T&gt; with trailing dimension [N, 2].
    /// </summary>
    public static NDArray<T> RFFT1D<T>(NDArray<T> realSignal) where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>
    {
        if (realSignal.Rank != 1) throw new ArgumentException("RFFT1D requires a 1D tensor.");
        int n = realSignal.Shape[0];

        Span<Complex<T>> cInput = stackalloc Complex<T>[n];
        for (int i = 0; i < n; i++) cInput[i] = new Complex<T>(realSignal[i], T.Zero);

        var cOutput = FFT1D<T>(cInput);
        var result = new NDArray<T>(n, 2);

        for (int i = 0; i < n; i++)
        {
            result[i, 0] = cOutput[i].Real;
            result[i, 1] = cOutput[i].Imaginary;
        }

        return result;
    }

    /// <summary>
    /// Complex-to-Real 1D Inverse FFT from NDArray&lt;T&gt; of shape [N, 2] to real NDArray&lt;T&gt; of shape [N].
    /// </summary>
    public static NDArray<T> IRFFT1D<T>(NDArray<T> complexSignal) where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>
    {
        if (complexSignal.Rank != 2 || complexSignal.Shape[1] != 2)
        {
            throw new ArgumentException("IRFFT1D requires a 2D tensor of shape [N, 2].");
        }

        int n = complexSignal.Shape[0];
        Span<Complex<T>> cInput = stackalloc Complex<T>[n];
        for (int i = 0; i < n; i++) cInput[i] = new Complex<T>(complexSignal[i, 0], complexSignal[i, 1]);

        var cOutput = IFFT1D<T>(cInput);
        var result = new NDArray<T>(n);

        for (int i = 0; i < n; i++)
        {
            result[i] = cOutput[i].Real;
        }

        return result;
    }

    #region Internal FFT Algorithms

    private static Complex<T>[] FFTRadix2<T>(ReadOnlySpan<Complex<T>> input) where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>
    {
        int n = input.Length;
        Complex<T>[] data = input.ToArray();
        BitReverse(data);

        T twoPi = T.CreateChecked(2.0 * Math.PI);

        for (int len = 2; len <= n; len <<= 1)
        {
            T angle = -twoPi / T.CreateChecked(len);
            Complex<T> wlen = new(T.Cos(angle), T.Sin(angle));

            for (int i = 0; i < n; i += len)
            {
                Complex<T> w = Complex<T>.One;
                int half = len >> 1;

                for (int j = 0; j < half; j++)
                {
                    Complex<T> u = data[i + j];
                    Complex<T> v = data[i + j + half] * w;

                    data[i + j] = u + v;
                    data[i + j + half] = u - v;

                    w = w * wlen;
                }
            }
        }

        return data;
    }

    private static Complex<T>[] FFTBluestein<T>(ReadOnlySpan<Complex<T>> input) where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>
    {
        int n = input.Length;
        // Find smallest power of 2 >= 2*n - 1
        int m = 1;
        while (m < 2 * n - 1) m <<= 1;

        T pi = T.CreateChecked(Math.PI);
        T factor = pi / T.CreateChecked(n);

        // Precompute chirp factors
        Complex<T>[] chirp = new Complex<T>[n];
        for (int i = 0; i < n; i++)
        {
            long iSq = (long)i * i % (2 * n);
            T angle = -factor * T.CreateChecked(iSq);
            chirp[i] = new Complex<T>(T.Cos(angle), T.Sin(angle));
        }

        // a_k = input[k] * chirp[k] (zero padded to m)
        Complex<T>[] a = new Complex<T>[m];
        for (int i = 0; i < n; i++)
        {
            a[i] = input[i] * chirp[i];
        }

        // b_k = conj(chirp[k]) for 0 <= k < n, and b_{m-k} = conj(chirp[k]) for 1 <= k < n
        Complex<T>[] b = new Complex<T>[m];
        b[0] = chirp[0].Conjugate();
        for (int i = 1; i < n; i++)
        {
            var c = chirp[i].Conjugate();
            b[i] = c;
            b[m - i] = c;
        }

        // Fast convolution via Radix-2 FFT
        var fA = FFTRadix2<T>(a);
        var fB = FFTRadix2<T>(b);

        Complex<T>[] fC = new Complex<T>[m];
        for (int i = 0; i < m; i++)
        {
            fC[i] = fA[i] * fB[i];
        }

        // IFFT of fC
        Complex<T>[] conjFC = new Complex<T>[m];
        for (int i = 0; i < m; i++) conjFC[i] = fC[i].Conjugate();
        var ifftC = FFTRadix2<T>(conjFC);

        T mScale = T.One / T.CreateChecked(m);
        Complex<T>[] result = new Complex<T>[n];
        for (int i = 0; i < n; i++)
        {
            var convVal = ifftC[i].Conjugate() * mScale;
            result[i] = convVal * chirp[i];
        }

        return result;
    }

    private static void BitReverse<T>(Complex<T>[] data) where T : unmanaged, INumber<T>
    {
        int n = data.Length;
        int j = 0;
        for (int i = 0; i < n - 1; i++)
        {
            if (i < j)
            {
                (data[i], data[j]) = (data[j], data[i]);
            }
            int k = n >> 1;
            while (k <= j)
            {
                j -= k;
                k >>= 1;
            }
            j += k;
        }
    }

    #endregion
}

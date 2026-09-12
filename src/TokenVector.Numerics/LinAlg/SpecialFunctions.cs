// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;
using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.LinAlg;

/// <summary>
/// Provides advanced mathematical and scientific special functions (Erf, Erfc, Gamma, LogGamma, BesselI0, BesselJ0) matching scipy.special / PyTorch Special.
/// </summary>
public static class SpecialFunctions
{
    #region Error Functions (Erf, Erfc)

    /// <summary>
    /// Computes the Error Function erf(x) = (2/sqrt(pi)) * \int_0^x e^{-t^2} dt.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Erf<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        var contig = tensor.Contiguous();
        var result = new NDArray<T>(tensor.Shape);

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double x = double.CreateTruncating(contig[i]);
            result[i] = T.CreateTruncating(ErfScalar(x));
        });

        return result;
    }

    /// <summary>
    /// Computes the Complementary Error Function erfc(x) = 1 - erf(x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Erfc<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        var contig = tensor.Contiguous();
        var result = new NDArray<T>(tensor.Shape);

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double x = double.CreateTruncating(contig[i]);
            result[i] = T.CreateTruncating(1.0 - ErfScalar(x));
        });

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double ErfScalar(double x)
    {
        // Abramowitz and Stegun formula 7.1.26 (max error: 1.5e-7)
        double sign = x < 0 ? -1.0 : 1.0;
        x = Math.Abs(x);

        double p = 0.3275911;
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return sign * y;
    }

    #endregion

    #region Gamma & Log-Gamma Functions

    /// <summary>
    /// Computes the Gamma function \Gamma(x) via Lanczos approximation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Gamma<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        var contig = tensor.Contiguous();
        var result = new NDArray<T>(tensor.Shape);

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double x = double.CreateTruncating(contig[i]);
            result[i] = T.CreateTruncating(GammaScalar(x));
        });

        return result;
    }

    /// <summary>
    /// Computes the natural logarithm of the absolute value of the Gamma function \ln|\Gamma(x)|.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> LogGamma<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        var contig = tensor.Contiguous();
        var result = new NDArray<T>(tensor.Shape);

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double x = double.CreateTruncating(contig[i]);
            result[i] = T.CreateTruncating(LogGammaScalar(x));
        });

        return result;
    }

    private static readonly double[] LanczosP =
    [
        676.520368128714,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.138571095856205,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double GammaScalar(double z)
    {
        if (z < 0.5)
        {
            // Reflection formula: Gamma(z) * Gamma(1-z) = pi / sin(pi * z)
            return Math.PI / (Math.Sin(Math.PI * z) * GammaScalar(1.0 - z));
        }

        z -= 1.0;
        double x = 0.99999999999980993;
        for (int i = 0; i < LanczosP.Length; i++)
        {
            x += LanczosP[i] / (z + i + 1);
        }

        double t = z + 7.5;
        return Math.Sqrt(2.0 * Math.PI) * Math.Pow(t, z + 0.5) * Math.Exp(-t) * x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double LogGammaScalar(double z)
    {
        if (z < 0.5)
        {
            return Math.Log(Math.PI / Math.Abs(Math.Sin(Math.PI * z))) - LogGammaScalar(1.0 - z);
        }

        z -= 1.0;
        double x = 0.99999999999980993;
        for (int i = 0; i < LanczosP.Length; i++)
        {
            x += LanczosP[i] / (z + i + 1);
        }

        double t = z + 7.5;
        return 0.5 * Math.Log(2.0 * Math.PI) + (z + 0.5) * Math.Log(t) - t + Math.Log(x);
    }

    #endregion

    #region Bessel Functions (I0, J0)

    /// <summary>
    /// Computes the Modified Bessel function of order 0: I_0(x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> BesselI0<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        var contig = tensor.Contiguous();
        var result = new NDArray<T>(tensor.Shape);

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double x = double.CreateTruncating(contig[i]);
            result[i] = T.CreateTruncating(BesselI0Scalar(x));
        });

        return result;
    }

    /// <summary>
    /// Computes the Bessel function of the first kind of order 0: J_0(x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> BesselJ0<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        var contig = tensor.Contiguous();
        var result = new NDArray<T>(tensor.Shape);

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double x = double.CreateTruncating(contig[i]);
            result[i] = T.CreateTruncating(BesselJ0Scalar(x));
        });

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double BesselI0Scalar(double x)
    {
        double ax = Math.Abs(x);
        if (ax < 3.75)
        {
            double y = (x / 3.75) * (x / 3.75);
            return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
        }
        else
        {
            double y = 3.75 / ax;
            return (Math.Exp(ax) / Math.Sqrt(ax)) * (0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2 + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2))))))));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double BesselJ0Scalar(double x)
    {
        double ax = Math.Abs(x);
        if (ax < 8.0)
        {
            double y = x * x;
            double ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456))));
            double ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
            return ans1 / ans2;
        }
        else
        {
            double z = 8.0 / ax;
            double y = z * z;
            double xx = ax - 0.785398164;
            double f0 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
            double theta0 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * -0.934945152e-7)));
            return Math.Sqrt(0.636619772 / ax) * (f0 * Math.Cos(xx + theta0));
        }
    }

    #endregion

    #region Advanced Special Functions (Beta, Digamma, Sinc, Logit, Expit, Erfinv)

    public static double Erf(double x) => ErfScalar(x);
    public static double LogGamma(double x) => LogGammaScalar(x);

    public static double Beta(double a, double b) =>
        Math.Exp(LogGammaScalar(a) + LogGammaScalar(b) - LogGammaScalar(a + b));

    public static double LogBeta(double a, double b) =>
        LogGammaScalar(a) + LogGammaScalar(b) - LogGammaScalar(a + b);

    public static double Sinc(double x) =>
        (Math.Abs(x) < 1e-12) ? 1.0 : Math.Sin(Math.PI * x) / (Math.PI * x);

    public static double Logit(double p) =>
        Math.Log(p / (1.0 - p));

    public static double Expit(double x) =>
        1.0 / (1.0 + Math.Exp(-x));

    public static double Digamma(double x)
    {
        // Asymptotic series for Digamma (Psi)
        double result = 0.0;
        while (x < 6.0)
        {
            result -= 1.0 / x;
            x += 1.0;
        }
        double r = 1.0 / x;
        result += Math.Log(x) - 0.5 * r;
        double r2 = r * r;
        result -= r2 * (1.0 / 12.0 - r2 * (1.0 / 120.0 - r2 * (1.0 / 252.0)));
        return result;
    }

    public static double Erfinv(double y)
    {
        // Winitzki approximation for inverse erf
        double a = 0.147;
        double logTerm = Math.Log(1.0 - y * y);
        double term1 = 2.0 / (Math.PI * a) + logTerm / 2.0;
        double innerSqrt = term1 * term1 - logTerm / a;
        double sign = y < 0 ? -1.0 : 1.0;
        return sign * Math.Sqrt(Math.Max(0.0, Math.Sqrt(innerSqrt) - term1));
    }

    #endregion
}

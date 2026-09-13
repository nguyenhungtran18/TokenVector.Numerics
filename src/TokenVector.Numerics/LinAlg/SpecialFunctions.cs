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

    #region Bessel Functions (I0, J0, Y0, K0)

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

    /// <summary>
    /// Computes the Bessel function of the second kind of order 0: Y_0(x) (Neumann function).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> BesselY0<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        var contig = tensor.Contiguous();
        var result = new NDArray<T>(tensor.Shape);

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double x = double.CreateTruncating(contig[i]);
            result[i] = T.CreateTruncating(BesselY0Scalar(x));
        });

        return result;
    }

    /// <summary>
    /// Computes the Modified Bessel function of the second kind of order 0: K_0(x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> BesselK0<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        var contig = tensor.Contiguous();
        var result = new NDArray<T>(tensor.Shape);

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double x = double.CreateTruncating(contig[i]);
            result[i] = T.CreateTruncating(BesselK0Scalar(x));
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double BesselY0Scalar(double x)
    {
        if (x <= 0.0) return double.NaN;
        if (x < 8.0)
        {
            double y = x * x;
            double ans1 = -2957821389.0 + y * (7062834065.0 + y * (-512354625.6 + y * (9872750.342 + y * (-77899.10659 + y * 214.072168))));
            double ans2 = 40076544269.0 + y * (745249964.8 + y * (7189466.438 + y * (47447.26470 + y * (226.1030244 + y * 1.0))));
            return (ans1 / ans2) + (2.0 / Math.PI) * Math.Log(x) * BesselJ0Scalar(x);
        }
        else
        {
            double z = 8.0 / x;
            double y = z * z;
            double xx = x - 0.785398164;
            double f0 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
            double theta0 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * -0.934945152e-7)));
            return Math.Sqrt(0.636619772 / x) * (f0 * Math.Sin(xx + theta0));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double BesselK0Scalar(double x)
    {
        if (x <= 0.0) return double.NaN;
        if (x <= 2.0)
        {
            double y = x * x * 0.25;
            double poly = -0.57721566 + y * (0.42278420 + y * (0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740)))));
            return -Math.Log(x * 0.5) * BesselI0Scalar(x) + poly;
        }
        else
        {
            double y = 2.0 / x;
            double poly = 1.25331414 + y * (-0.07832358 + y * (0.02189568 + y * (-0.01062446 + y * (0.00587872 + y * (-0.00251540 + y * 0.00053208)))));
            return (Math.Exp(-x) / Math.Sqrt(x)) * poly;
        }
    }

    #endregion

    #region Lambert W Function

    /// <summary>
    /// Computes the Lambert W function W_k(x) where W(x) * exp(W(x)) = x via Halley's 3rd order iteration.
    /// Supports principal branch k=0 (for x >= -1/e) and branch k=-1 (for -1/e &lt;= x &lt; 0).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> LambertW<T>(NDArray<T> tensor, int branch = 0) where T : unmanaged, INumber<T>
    {
        var contig = tensor.Contiguous();
        var result = new NDArray<T>(tensor.Shape);

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double x = double.CreateTruncating(contig[i]);
            result[i] = T.CreateTruncating(LambertWScalar(x, branch));
        });

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static double LambertW(double x, int branch = 0) => LambertWScalar(x, branch);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double LambertWScalar(double x, int branch)
    {
        const double invE = 0.3678794411714423215955; // 1/e
        if (double.IsNaN(x) || x < -invE) return double.NaN;

        double w;
        if (branch == 0)
        {
            if (x == 0.0) return 0.0;
            if (Math.Abs(x + invE) < 1e-14) return -1.0;

            if (x < 1.0)
            {
                // Series near 0: w ~ x - x^2 + 1.5 x^3 - 8/3 x^4
                if (Math.Abs(x) < 0.1)
                {
                    w = x * (1.0 + x * (-1.0 + x * (1.5 - (8.0 / 3.0) * x)));
                }
                else
                {
                    // Branch point expansion near -1/e
                    double p = Math.Sqrt(2.0 * (Math.E * x + 1.0));
                    w = -1.0 + p - (1.0 / 3.0) * p * p + (11.0 / 72.0) * p * p * p;
                }
            }
            else
            {
                // Asymptotic expansion for large x: w ~ ln(x) - ln(ln(x))
                double l1 = Math.Log(x);
                double l2 = Math.Log(l1);
                w = l1 - l2 + l2 / l1;
            }
        }
        else if (branch == -1)
        {
            if (x >= 0.0 || x < -invE) return double.NaN;
            if (Math.Abs(x + invE) < 1e-14) return -1.0;

            // Asymptotic expansion near 0-
            double l1 = Math.Log(-x);
            double l2 = Math.Log(-l1);
            w = l1 - l2 + l2 / l1;
        }
        else
        {
            throw new ArgumentOutOfRangeException(nameof(branch), "Only branches 0 and -1 are supported.");
        }

        // Halley's 3rd order iteration: w_{n+1} = w_n - f / (f' - f * f'' / (2 * f'))
        for (int iter = 0; iter < 10; iter++)
        {
            double ew = Math.Exp(w);
            double wew = w * ew;
            double f = wew - x;

            if (Math.Abs(f) < 1e-15 * Math.Max(1.0, Math.Abs(wew))) break;

            double fPrime = ew * (w + 1.0);
            double fDoublePrime = ew * (w + 2.0);

            double denom = fPrime - (f * fDoublePrime) / (2.0 * fPrime);
            double delta = f / denom;
            w -= delta;

            if (Math.Abs(delta) < 1e-15 * Math.Max(1.0, Math.Abs(w))) break;
        }

        return w;
    }

    #endregion

    #region Airy Functions (Ai, Bi)

    /// <summary>
    /// Computes the Airy function of the first kind: Ai(x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> AiryAi<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        var contig = tensor.Contiguous();
        var result = new NDArray<T>(tensor.Shape);

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double x = double.CreateTruncating(contig[i]);
            result[i] = T.CreateTruncating(AiryAiScalar(x));
        });

        return result;
    }

    /// <summary>
    /// Computes the Airy function of the second kind: Bi(x).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> AiryBi<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        var contig = tensor.Contiguous();
        var result = new NDArray<T>(tensor.Shape);

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double x = double.CreateTruncating(contig[i]);
            result[i] = T.CreateTruncating(AiryBiScalar(x));
        });

        return result;
    }

    public static double AiryAi(double x) => AiryAiScalar(x);
    public static double AiryBi(double x) => AiryBiScalar(x);

    private const double Ai0 = 0.35502805388781724;  // 1 / (3^(2/3) * Gamma(2/3))
    private const double Bi0 = 0.61492662744600073;  // 1 / (3^(1/6) * Gamma(2/3)) = Ai0 * sqrt(3)
    private const double AiPrime0 = 0.25881940379280680; // 1 / (3^(1/3) * Gamma(1/3))

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double AiryAiScalar(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (x > 25.0) return 0.0;

        if (x >= 4.0)
        {
            // Asymptotic expansion for large positive x
            double z = (2.0 / 3.0) * Math.Pow(x, 1.5);
            double pre = Math.Exp(-z) / (2.0 * Math.Sqrt(Math.PI) * Math.Pow(x, 0.25));
            double sum = 1.0 - (5.0 / (72.0 * z)) + (385.0 / (10368.0 * z * z));
            return pre * sum;
        }
        else if (x <= -4.0)
        {
            // Asymptotic expansion for large negative x
            double ax = -x;
            double z = (2.0 / 3.0) * Math.Pow(ax, 1.5);
            double pre = 1.0 / (Math.Sqrt(Math.PI) * Math.Pow(ax, 0.25));
            return pre * Math.Sin(z + Math.PI * 0.25);
        }
        else
        {
            // Maclaurin series around 0
            ComputeAirySeries(x, out double f, out double g);
            return Ai0 * f - AiPrime0 * g;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double AiryBiScalar(double x)
    {
        if (double.IsNaN(x)) return double.NaN;

        if (x >= 4.0)
        {
            double z = (2.0 / 3.0) * Math.Pow(x, 1.5);
            double pre = Math.Exp(z) / (Math.Sqrt(Math.PI) * Math.Pow(x, 0.25));
            double sum = 1.0 + (5.0 / (72.0 * z)) + (385.0 / (10368.0 * z * z));
            return pre * sum;
        }
        else if (x <= -4.0)
        {
            double ax = -x;
            double z = (2.0 / 3.0) * Math.Pow(ax, 1.5);
            double pre = 1.0 / (Math.Sqrt(Math.PI) * Math.Pow(ax, 0.25));
            return pre * Math.Cos(z + Math.PI * 0.25);
        }
        else
        {
            ComputeAirySeries(x, out double f, out double g);
            return Math.Sqrt(3.0) * (Ai0 * f + AiPrime0 * g);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeAirySeries(double x, out double f, out double g)
    {
        double x3 = x * x * x;
        f = 1.0;
        g = x;

        double termF = 1.0;
        double termG = x;

        for (int k = 1; k <= 30; k++)
        {
            termF *= x3 / ((3.0 * k - 2.0) * (3.0 * k - 1.0) * (3.0 * k));
            f += termF;

            termG *= x3 / ((3.0 * k - 1.0) * (3.0 * k) * (3.0 * k + 1.0));
            g += termG;

            if (Math.Abs(termF) < 1e-16 && Math.Abs(termG) < 1e-16) break;
        }
    }

    #endregion

    #region Advanced Special Functions (Beta, Digamma, Sinc, Logit, Expit, Erfinv, Bessel)

    public static double Erf(double x) => ErfScalar(x);
    public static double LogGamma(double x) => LogGammaScalar(x);
    public static double BesselI0(double x) => BesselI0Scalar(x);
    public static double BesselJ0(double x) => BesselJ0Scalar(x);
    public static double BesselY0(double x) => BesselY0Scalar(x);
    public static double BesselK0(double x) => BesselK0Scalar(x);

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

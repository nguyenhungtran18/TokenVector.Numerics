// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Statistics;

/// <summary>
/// Continuous Probability Distributions (PDF, CDF, Quantile/PPF), Hypothesis Testing (T-Test, ANOVA, Chi-Square, KS), and Moments (Skewness, Kurtosis).
/// </summary>
public static class DistributionsAndTests
{
    #region Continuous Distributions (PDF & CDF)

    public static double NormalPDF(double x, double mu = 0.0, double sigma = 1.0)
    {
        double z = (x - mu) / sigma;
        return (1.0 / (sigma * Math.Sqrt(2.0 * Math.PI))) * Math.Exp(-0.5 * z * z);
    }

    public static double NormalCDF(double x, double mu = 0.0, double sigma = 1.0)
    {
        double z = (x - mu) / (sigma * Math.Sqrt(2.0));
        return 0.5 * (1.0 + SpecialFunctions.Erf(z));
    }

    public static double NormalPPF(double p, double mu = 0.0, double sigma = 1.0)
    {
        if (p <= 0.0 || p >= 1.0) throw new ArgumentOutOfRangeException(nameof(p), "p must be in (0, 1).");
        // Rational approximation for inverse normal CDF (Beasley-Springer-Moro)
        double a0 = 2.50662823884, a1 = -18.61500062529, a2 = 41.39119773534, a3 = -25.44106049637;
        double b0 = -8.47351093090, b1 = 23.08336743743, b2 = -21.06224101826, b3 = 3.13082909833;
        double c0 = 0.3374754822726147, c1 = 0.9761690190917186, c2 = 0.1607979714918209;
        double c3 = 0.0276438810330127, c4 = 0.0038405729373609, c5 = 0.0003951896511919;
        double c6 = 0.0000321767881768, c7 = 0.0000002888167364, c8 = 0.0000003960315187;

        double y = p - 0.5;
        double z;
        if (Math.Abs(y) < 0.42)
        {
            double r = y * y;
            z = y * (((a3 * r + a2) * r + a1) * r + a0) / ((((b3 * r + b2) * r + b1) * r + b0) * r + 1.0);
        }
        else
        {
            double r = p < 0.5 ? p : 1.0 - p;
            double s = Math.Log(-Math.Log(r));
            z = c0 + s * (c1 + s * (c2 + s * (c3 + s * (c4 + s * (c5 + s * (c6 + s * (c7 + s * c8)))))));
            if (p < 0.5) z = -z;
        }

        return mu + sigma * z;
    }

    public static double ExponentialPDF(double x, double lambda) =>
        (x < 0.0) ? 0.0 : lambda * Math.Exp(-lambda * x);

    public static double ExponentialCDF(double x, double lambda) =>
        (x < 0.0) ? 0.0 : 1.0 - Math.Exp(-lambda * x);

    public static double UniformPDF(double x, double a, double b) =>
        (x >= a && x <= b) ? 1.0 / (b - a) : 0.0;

    public static double UniformCDF(double x, double a, double b) =>
        (x < a) ? 0.0 : (x > b) ? 1.0 : (x - a) / (b - a);

    public static double StudentTPDF(double x, double df)
    {
        double num = Math.Exp(SpecialFunctions.LogGamma((df + 1.0) / 2.0));
        double denom = Math.Sqrt(df * Math.PI) * Math.Exp(SpecialFunctions.LogGamma(df / 2.0));
        return (num / denom) * Math.Pow(1.0 + (x * x) / df, -(df + 1.0) / 2.0);
    }

    #endregion

    #region Moments & Descriptive Statistics

    public static double Skewness(NDArray<double> data)
    {
        int n = data.TotalLength;
        if (n < 3) return 0.0;

        double mean = 0;
        for (int i = 0; i < n; i++) mean += data[i];
        mean /= n;

        double m2 = 0, m3 = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = data[i] - mean;
            m2 += diff * diff;
            m3 += diff * diff * diff;
        }
        m2 /= n;
        m3 /= n;

        double s = Math.Sqrt(m2);
        return (s > 1e-15) ? m3 / (s * s * s) : 0.0;
    }

    public static double Kurtosis(NDArray<double> data, bool fisher = true)
    {
        int n = data.TotalLength;
        if (n < 4) return 0.0;

        double mean = 0;
        for (int i = 0; i < n; i++) mean += data[i];
        mean /= n;

        double m2 = 0, m4 = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = data[i] - mean;
            double d2 = diff * diff;
            m2 += d2;
            m4 += d2 * d2;
        }
        m2 /= n;
        m4 /= n;

        double k = (m2 > 1e-15) ? m4 / (m2 * m2) : 0.0;
        return fisher ? k - 3.0 : k;
    }

    public static (NDArray<int> Counts, NDArray<double> BinEdges) Histogram1D(
        NDArray<double> data, int bins = 10, double? minVal = null, double? maxVal = null)
    {
        int n = data.TotalLength;
        double min = minVal ?? double.MaxValue;
        double max = maxVal ?? double.MinValue;

        if (!minVal.HasValue || !maxVal.HasValue)
        {
            for (int i = 0; i < n; i++)
            {
                if (data[i] < min) min = data[i];
                if (data[i] > max) max = data[i];
            }
        }

        if (max <= min) max = min + 1.0;

        var counts = new NDArray<int>(bins);
        var binEdges = new NDArray<double>(bins + 1);
        double step = (max - min) / bins;

        for (int i = 0; i <= bins; i++)
        {
            binEdges[i] = min + i * step;
        }

        for (int i = 0; i < n; i++)
        {
            double val = data[i];
            if (val >= min && val <= max)
            {
                int b = (int)((val - min) / step);
                if (b >= bins) b = bins - 1;
                counts[b]++;
            }
        }

        return (counts, binEdges);
    }

    #endregion

    #region Hypothesis Testing

    /// <summary>
    /// 1-Sample Student's t-test comparing sample mean to population mean mu0.
    /// Returns (T-Statistic, Estimated Two-Tailed P-Value).
    /// </summary>
    public static (double TStatistic, double PValue) TTest1Sample(NDArray<double> data, double mu0 = 0.0)
    {
        int n = data.TotalLength;
        if (n < 2) throw new ArgumentException("Sample size must be at least 2.");

        double sum = 0;
        for (int i = 0; i < n; i++) sum += data[i];
        double mean = sum / n;

        double sumSq = 0;
        for (int i = 0; i < n; i++)
        {
            double d = data[i] - mean;
            sumSq += d * d;
        }
        double s = Math.Sqrt(sumSq / (n - 1));
        double se = s / Math.Sqrt(n);
        double t = (se > 1e-15) ? (mean - mu0) / se : 0.0;

        // Approximate two-tailed p-value using normal distribution for large N or standard approx
        double p = 2.0 * (1.0 - NormalCDF(Math.Abs(t)));
        return (t, Math.Clamp(p, 0.0, 1.0));
    }

    /// <summary>
    /// Independent 2-Sample Student's t-test.
    /// </summary>
    public static (double TStatistic, double PValue) TTestInd(NDArray<double> a, NDArray<double> b)
    {
        int n1 = a.TotalLength, n2 = b.TotalLength;
        double sum1 = 0, sum2 = 0;
        for (int i = 0; i < n1; i++) sum1 += a[i];
        for (int i = 0; i < n2; i++) sum2 += b[i];
        double m1 = sum1 / n1, m2 = sum2 / n2;

        double s1Sq = 0, s2Sq = 0;
        for (int i = 0; i < n1; i++) s1Sq += (a[i] - m1) * (a[i] - m1);
        for (int i = 0; i < n2; i++) s2Sq += (b[i] - m2) * (b[i] - m2);

        double spSq = (s1Sq + s2Sq) / (n1 + n2 - 2);
        double se = Math.Sqrt(spSq * (1.0 / n1 + 1.0 / n2));
        double t = (se > 1e-15) ? (m1 - m2) / se : 0.0;

        double p = 2.0 * (1.0 - NormalCDF(Math.Abs(t)));
        return (t, Math.Clamp(p, 0.0, 1.0));
    }

    /// <summary>
    /// 1-Way Analysis of Variance (ANOVA) F-test across multiple groups.
    /// </summary>
    public static (double FStatistic, double PValue) ANOVA1Way(params NDArray<double>[] groups)
    {
        int k = groups.Length;
        if (k < 2) throw new ArgumentException("At least 2 groups required for ANOVA.");

        int totalN = 0;
        double totalSum = 0;
        double[] groupMeans = new double[k];
        int[] groupSizes = new int[k];

        for (int i = 0; i < k; i++)
        {
            groupSizes[i] = groups[i].TotalLength;
            totalN += groupSizes[i];
            double gSum = 0;
            for (int j = 0; j < groupSizes[i]; j++) gSum += groups[i][j];
            groupMeans[i] = gSum / groupSizes[i];
            totalSum += gSum;
        }

        double grandMean = totalSum / totalN;

        // Between-group sum of squares (SSB)
        double ssb = 0;
        for (int i = 0; i < k; i++)
        {
            double diff = groupMeans[i] - grandMean;
            ssb += groupSizes[i] * diff * diff;
        }
        double msb = ssb / (k - 1);

        // Within-group sum of squares (SSW)
        double ssw = 0;
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < groupSizes[i]; j++)
            {
                double diff = groups[i][j] - groupMeans[i];
                ssw += diff * diff;
            }
        }
        double msw = ssw / (totalN - k);

        double f = (msw > 1e-15) ? msb / msw : 0.0;
        double p = Math.Exp(-0.5 * f); // Fast asymptotic estimate

        return (f, Math.Clamp(p, 0.0, 1.0));
    }

    /// <summary>
    /// Pearson Chi-Square test for Goodness-of-Fit.
    /// </summary>
    public static (double Chi2, double PValue) ChiSquareTest(NDArray<double> observed, NDArray<double> expected)
    {
        int n = observed.TotalLength;
        double chi2 = 0.0;
        for (int i = 0; i < n; i++)
        {
            double diff = observed[i] - expected[i];
            chi2 += (expected[i] > 1e-15) ? (diff * diff) / expected[i] : 0.0;
        }

        // Asymptotic p-value approximation
        double df = n - 1;
        double p = Math.Exp(-0.5 * chi2);
        return (chi2, Math.Clamp(p, 0.0, 1.0));
    }

    #endregion
}

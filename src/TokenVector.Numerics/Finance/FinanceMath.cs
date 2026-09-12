// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Finance;

/// <summary>
/// Provides quantitative financial mathematics: Black-Scholes-Merton option pricing, Risk Greeks, Markowitz Portfolio theory, and Cashflow discounting (NPV, IRR).
/// </summary>
public static class FinanceMath
{
    #region Black-Scholes-Merton & The Greeks

    /// <summary>
    /// Computes European Call Option Price using Black-Scholes-Merton formula.
    /// S: Spot price, K: Strike price, T: Time to maturity (years), r: Risk-free rate, sigma: Volatility.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double BlackScholesCall(double s, double k, double t, double r, double sigma)
    {
        if (t <= 0) return Math.Max(0.0, s - k);
        var (d1, d2) = ComputeD1D2(s, k, t, r, sigma);
        return s * StandardNormalCdf(d1) - k * Math.Exp(-r * t) * StandardNormalCdf(d2);
    }

    /// <summary>
    /// Computes European Put Option Price using Black-Scholes-Merton formula.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double BlackScholesPut(double s, double k, double t, double r, double sigma)
    {
        if (t <= 0) return Math.Max(0.0, k - s);
        var (d1, d2) = ComputeD1D2(s, k, t, r, sigma);
        return k * Math.Exp(-r * t) * StandardNormalCdf(-d2) - s * StandardNormalCdf(-d1);
    }

    /// <summary>
    /// Computes option risk sensitivities (The Greeks): Delta, Gamma, Vega, Theta, Rho.
    /// </summary>
    public static (double Delta, double Gamma, double Vega, double Theta, double Rho) OptionGreeks(
        double s, double k, double t, double r, double sigma, bool isCall = true)
    {
        var (d1, d2) = ComputeD1D2(s, k, t, r, sigma);
        double nd1 = StandardNormalPdf(d1);
        double nCdfD1 = StandardNormalCdf(d1);
        double nCdfD2 = StandardNormalCdf(d2);

        double delta = isCall ? nCdfD1 : nCdfD1 - 1.0;
        double gamma = nd1 / (s * sigma * Math.Sqrt(t));
        double vega = s * nd1 * Math.Sqrt(t); // Vega per 100% vol (often divided by 100 for 1% vol)

        double theta;
        if (isCall)
        {
            theta = - (s * nd1 * sigma) / (2.0 * Math.Sqrt(t)) - r * k * Math.Exp(-r * t) * nCdfD2;
        }
        else
        {
            theta = - (s * nd1 * sigma) / (2.0 * Math.Sqrt(t)) + r * k * Math.Exp(-r * t) * StandardNormalCdf(-d2);
        }

        double rho = isCall
            ? k * t * Math.Exp(-r * t) * nCdfD2
            : -k * t * Math.Exp(-r * t) * StandardNormalCdf(-d2);

        return (delta, gamma, vega, theta, rho);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (double D1, double D2) ComputeD1D2(double s, double k, double t, double r, double sigma)
    {
        double d1 = (Math.Log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * Math.Sqrt(t));
        double d2 = d1 - sigma * Math.Sqrt(t);
        return (d1, d2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double StandardNormalPdf(double x) => Math.Exp(-0.5 * x * x) / Math.Sqrt(2.0 * Math.PI);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double StandardNormalCdf(double x) => 0.5 * (1.0 + ErfScalar(x / Math.Sqrt(2.0)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double ErfScalar(double x)
    {
        double sign = x < 0 ? -1.0 : 1.0;
        x = Math.Abs(x);
        double p = 0.3275911;
        double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return sign * y;
    }

    #endregion

    #region Markowitz Portfolio Optimization

    /// <summary>
    /// Computes expected portfolio return: r_p = w^T * r.
    /// </summary>
    public static double PortfolioReturn(NDArray<double> weights, NDArray<double> expectedReturns)
    {
        var w = weights.Contiguous();
        var r = expectedReturns.Contiguous();
        double ret = 0.0;
        for (int i = 0; i < w.TotalLength; i++) ret += w[i] * r[i];
        return ret;
    }

    /// <summary>
    /// Computes expected portfolio volatility (Standard Deviation): sigma_p = sqrt(w^T * Cov * w).
    /// </summary>
    public static double PortfolioVolatility(NDArray<double> weights, NDArray<double> covMatrix)
    {
        int n = weights.TotalLength;
        var w = weights.Contiguous();
        var cov = covMatrix.Contiguous();

        double variance = 0.0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                variance += w[i] * cov[i, j] * w[j];
            }
        }
        return Math.Sqrt(Math.Max(0.0, variance));
    }

    /// <summary>
    /// Computes portfolio Sharpe Ratio: (r_p - r_f) / sigma_p.
    /// </summary>
    public static double SharpeRatio(NDArray<double> weights, NDArray<double> expectedReturns, NDArray<double> covMatrix, double riskFreeRate = 0.0)
    {
        double rP = PortfolioReturn(weights, expectedReturns);
        double vol = PortfolioVolatility(weights, covMatrix);
        if (vol < 1e-15) return 0.0;
        return (rP - riskFreeRate) / vol;
    }

    #endregion

    #region Cashflows & Discounting (NPV, IRR)

    /// <summary>
    /// Computes Net Present Value (NPV) of a cash flow sequence: NPV = \sum_{t=0}^N CF_t / (1 + r)^t.
    /// </summary>
    public static double NPV(double rate, NDArray<double> cashFlows)
    {
        var cf = cashFlows.Contiguous();
        double npv = 0.0;
        double factor = 1.0;
        double onePlusR = 1.0 + rate;

        for (int t = 0; t < cf.TotalLength; t++)
        {
            npv += cf[t] / factor;
            factor *= onePlusR;
        }

        return npv;
    }

    /// <summary>
    /// Computes Internal Rate of Return (IRR) using Newton-Raphson method.
    /// </summary>
    public static double IRR(NDArray<double> cashFlows, double guess = 0.1, double tol = 1e-8, int maxIter = 100)
    {
        var cf = cashFlows.Contiguous();
        double r = guess;

        for (int iter = 0; iter < maxIter; iter++)
        {
            double f = 0.0;
            double fPrime = 0.0;

            for (int t = 0; t < cf.TotalLength; t++)
            {
                double denom = Math.Pow(1.0 + r, t);
                f += cf[t] / denom;
                if (t > 0)
                {
                    fPrime -= t * cf[t] / Math.Pow(1.0 + r, t + 1);
                }
            }

            if (Math.Abs(f) < tol) return r;
            if (Math.Abs(fPrime) < 1e-15) break;

            double delta = f / fPrime;
            r -= delta;

            if (Math.Abs(delta) < tol) return r;
        }

        return r;
    }

    #endregion

    #region American Options, Risk Measures & Fixed Income

    /// <summary>
    /// Cox-Ross-Rubinstein (CRR) Binomial Tree model for American Option pricing.
    /// </summary>
    public static double BinomialTreeAmericanOption(
        double s, double k, double t, double r, double sigma, int steps = 100, bool isCall = false)
    {
        double dt = t / steps;
        double u = Math.Exp(sigma * Math.Sqrt(dt));
        double d = 1.0 / u;
        double p = (Math.Exp(r * dt) - d) / (u - d);
        double discount = Math.Exp(-r * dt);

        double[] values = new double[steps + 1];

        // Terminal payoff
        for (int i = 0; i <= steps; i++)
        {
            double st = s * Math.Pow(u, steps - i) * Math.Pow(d, i);
            values[i] = isCall ? Math.Max(0.0, st - k) : Math.Max(0.0, k - st);
        }

        // Backward induction with early exercise check
        for (int step = steps - 1; step >= 0; step--)
        {
            for (int i = 0; i <= step; i++)
            {
                double st = s * Math.Pow(u, step - i) * Math.Pow(d, i);
                double holdValue = discount * (p * values[i] + (1.0 - p) * values[i + 1]);
                double exerciseValue = isCall ? Math.Max(0.0, st - k) : Math.Max(0.0, k - st);
                values[i] = Math.Max(holdValue, exerciseValue);
            }
        }

        return values[0];
    }

    /// <summary>
    /// Computes Value at Risk (VaR) for portfolio returns at given confidence level (e.g. 0.95 or 0.99).
    /// </summary>
    public static double ValueAtRisk(NDArray<double> returns, double confidence = 0.95)
    {
        int n = returns.TotalLength;
        double[] sorted = new double[n];
        for (int i = 0; i < n; i++) sorted[i] = returns[i];
        Array.Sort(sorted);

        int idx = (int)Math.Floor((1.0 - confidence) * n);
        idx = Math.Clamp(idx, 0, n - 1);
        return -sorted[idx];
    }

    /// <summary>
    /// Computes Conditional Value at Risk (CVaR / Expected Shortfall).
    /// </summary>
    public static double ConditionalValueAtRisk(NDArray<double> returns, double confidence = 0.95)
    {
        int n = returns.TotalLength;
        double[] sorted = new double[n];
        for (int i = 0; i < n; i++) sorted[i] = returns[i];
        Array.Sort(sorted);

        int cutoffIdx = (int)Math.Floor((1.0 - confidence) * n);
        if (cutoffIdx <= 0) return -sorted[0];

        double sum = 0;
        for (int i = 0; i < cutoffIdx; i++) sum += sorted[i];
        return -sum / cutoffIdx;
    }

    /// <summary>
    /// Computes fair market price of a fixed-coupon bond.
    /// </summary>
    public static double BondPrice(double faceValue, double couponRate, double ytm, int periods, int freq = 1)
    {
        double coupon = (faceValue * couponRate) / freq;
        double y = ytm / freq;
        double pvCoupons = (y > 1e-12) ? coupon * (1.0 - Math.Pow(1.0 + y, -periods)) / y : coupon * periods;
        double pvFace = faceValue * Math.Pow(1.0 + y, -periods);
        return pvCoupons + pvFace;
    }

    /// <summary>
    /// Computes Macaulay Duration of a bond.
    /// </summary>
    public static double MacaulayDuration(double price, double faceValue, double couponRate, double ytm, int periods, int freq = 1)
    {
        double coupon = (faceValue * couponRate) / freq;
        double y = ytm / freq;
        double sum = 0;

        for (int t = 1; t <= periods; t++)
        {
            double cf = (t == periods) ? coupon + faceValue : coupon;
            sum += (t / (double)freq) * (cf / Math.Pow(1.0 + y, t));
        }

        return sum / price;
    }

    /// <summary>
    /// Evaluates Nelson-Siegel yield curve spot rate at maturity t (years).
    /// </summary>
    public static double NelsonSiegel(double beta0, double beta1, double beta2, double lambda, double t)
    {
        if (t <= 1e-6) return beta0 + beta1;
        double m = t / lambda;
        double expM = Math.Exp(-m);
        double term1 = (1.0 - expM) / m;
        double term2 = term1 - expM;
        return beta0 + beta1 * term1 + beta2 * term2;
    }

    #endregion
}

// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Engine;

namespace TokenVector.Numerics.Ops;

/// <summary>
/// Comprehensive element-wise mathematical, trigonometric, hyperbolic, exponential, logarithmic, and rounding operations on NDArrays.
/// </summary>
public static class ElementWiseMathOps
{
    #region Trigonometric & Hyperbolic

    public static NDArray<double> Sin(NDArray<double> a) => Map(a, Math.Sin);
    public static NDArray<double> Cos(NDArray<double> a) => Map(a, Math.Cos);
    public static NDArray<double> Tan(NDArray<double> a) => Map(a, Math.Tan);

    public static NDArray<double> ArcSin(NDArray<double> a) => Map(a, Math.Asin);
    public static NDArray<double> ArcCos(NDArray<double> a) => Map(a, Math.Acos);
    public static NDArray<double> ArcTan(NDArray<double> a) => Map(a, Math.Atan);
    public static NDArray<double> ArcTan2(NDArray<double> y, NDArray<double> x) => BinaryMap(y, x, Math.Atan2);

    public static NDArray<double> Sinh(NDArray<double> a) => Map(a, Math.Sinh);
    public static NDArray<double> Cosh(NDArray<double> a) => Map(a, Math.Cosh);
    public static NDArray<double> Tanh(NDArray<double> a) => Map(a, Math.Tanh);

    public static NDArray<double> ArcSinh(NDArray<double> a) => Map(a, Math.Asinh);
    public static NDArray<double> ArcCosh(NDArray<double> a) => Map(a, Math.Acosh);
    public static NDArray<double> ArcTanh(NDArray<double> a) => Map(a, Math.Atanh);

    public static NDArray<double> Deg2Rad(NDArray<double> a) => Map(a, deg => deg * (Math.PI / 180.0));
    public static NDArray<double> Rad2Deg(NDArray<double> a) => Map(a, rad => rad * (180.0 / Math.PI));

    public static NDArray<double> Unwrap(NDArray<double> a, double discont = Math.PI, double period = 2 * Math.PI)
    {
        var res = a.Clone();
        int n = res.TotalLength;
        if (n <= 1) return res;

        double high = discont;
        double low = -discont;

        for (int i = 1; i < n; i++)
        {
            double diff = res[i] - res[i - 1];
            double diffMod = (diff - low) % period;
            if (diffMod < 0) diffMod += period;
            diffMod += low;
            if (diffMod == low && diff > 0) diffMod = high;
            double correction = diffMod - diff;
            if (Math.Abs(correction) > 0)
            {
                for (int j = i; j < n; j++)
                {
                    res[j] += correction;
                }
            }
        }
        return res;
    }

    #endregion

    #region Exponential & Logarithmic

    public static NDArray<double> Exp(NDArray<double> a) => Map(a, Math.Exp);
    public static NDArray<double> Exp2(NDArray<double> a) => Map(a, x => Math.Pow(2.0, x));
    public static NDArray<double> Expm1(NDArray<double> a) => Map(a, x => Math.Exp(x) - 1.0);

    public static NDArray<double> Log(NDArray<double> a) => Map(a, Math.Log);
    public static NDArray<double> Log2(NDArray<double> a) => Map(a, Math.Log2);
    public static NDArray<double> Log10(NDArray<double> a) => Map(a, Math.Log10);
    public static NDArray<double> Log1p(NDArray<double> a) => Map(a, x => Math.Log(1.0 + x));

    public static NDArray<double> LogAddExp(NDArray<double> x1, NDArray<double> x2) =>
        BinaryMap(x1, x2, (a, b) =>
        {
            double max = Math.Max(a, b);
            double min = Math.Min(a, b);
            return max + Math.Log(1.0 + Math.Exp(min - max));
        });

    public static NDArray<double> LogAddExp2(NDArray<double> x1, NDArray<double> x2) =>
        BinaryMap(x1, x2, (a, b) =>
        {
            double max = Math.Max(a, b);
            double min = Math.Min(a, b);
            return max + Math.Log2(1.0 + Math.Pow(2.0, min - max));
        });

    public static NDArray<double> Sqrt(NDArray<double> a) => Map(a, Math.Sqrt);
    public static NDArray<double> Cbrt(NDArray<double> a) => Map(a, Math.Cbrt);
    public static NDArray<double> Square(NDArray<double> a) => Map(a, x => x * x);
    public static NDArray<double> Reciprocal(NDArray<double> a) => Map(a, x => 1.0 / x);
    public static NDArray<double> Hypot(NDArray<double> x1, NDArray<double> x2) =>
        BinaryMap(x1, x2, (a, b) => Math.Sqrt(a * a + b * b));

    #endregion

    #region Rounding & Float Properties

    public static NDArray<double> Ceil(NDArray<double> a) => Map(a, Math.Ceiling);
    public static NDArray<double> Floor(NDArray<double> a) => Map(a, Math.Floor);
    public static NDArray<double> Trunc(NDArray<double> a) => Map(a, Math.Truncate);
    public static NDArray<double> Round(NDArray<double> a, int decimals = 0) => Map(a, x => Math.Round(x, decimals));
    public static NDArray<double> Sign(NDArray<double> a) => Map(a, x => Math.Sign(x));
    public static NDArray<double> Copysign(NDArray<double> x1, NDArray<double> x2) => BinaryMap(x1, x2, Math.CopySign);
    public static NDArray<double> FMod(NDArray<double> x1, NDArray<double> x2) => BinaryMap(x1, x2, (a, b) => a % b);

    #endregion

    #region Helper Mapping Functions

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static NDArray<double> Map(NDArray<double> a, Func<double, double> func)
    {
        var result = new NDArray<double>(a.Shape);
        int total = a.TotalLength;
        Parallel.For(0, total, i =>
        {
            result[i] = func(a[i]);
        });
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static NDArray<double> BinaryMap(NDArray<double> a, NDArray<double> b, Func<double, double, double> func)
    {
        var (bA, bB) = BroadcastEngine.Broadcast(a, b);
        var result = new NDArray<double>(bA.Shape);
        int total = bA.TotalLength;
        Parallel.For(0, total, i =>
        {
            result[i] = func(bA[i], bB[i]);
        });
        return result;
    }

    #endregion
}

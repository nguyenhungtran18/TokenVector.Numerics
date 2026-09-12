// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;
using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.LinAlg;

/// <summary>
/// Provides polynomial fitting, evaluation, and root-finding routines (np.polyfit, np.polyval, np.roots).
/// </summary>
public static class Polynomial
{
    /// <summary>
    /// Fits a polynomial of degree `deg` to points (x, y) using least-squares Vandermonde formulation.
    /// Returns coefficients in descending order: [c_deg, c_{deg-1}, ..., c_0].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<double> PolyFit(NDArray<double> x, NDArray<double> y, int deg)
    {
        if (x.Rank != 1 || y.Rank != 1)
            throw new ArgumentException("x and y must be 1D vectors.");
        if (x.Shape[0] != y.Shape[0])
            throw new ArgumentException("x and y must have the same length.");
        if (deg < 0)
            throw new ArgumentOutOfRangeException(nameof(deg), "Polynomial degree must be >= 0.");

        int n = x.Shape[0];
        int numCoeffs = deg + 1;
        if (n < numCoeffs)
            throw new InvalidOperationException($"Number of points ({n}) must be >= degree + 1 ({numCoeffs}).");

        var xContig = x.Contiguous();
        var v = new NDArray<double>(n, numCoeffs);

        for (int i = 0; i < n; i++)
        {
            double xi = xContig[i];
            double p = 1.0;
            // Fill from right to left (power 0 up to deg)
            for (int j = numCoeffs - 1; j >= 0; j--)
            {
                v[i, j] = p;
                p *= xi;
            }
        }

        // Solve least-squares: V * c = y
        return LinAlgExtended.LstSq(v, y);
    }

    /// <summary>
    /// Evaluates a polynomial with coefficients `coeffs` (in descending degree order) at points `x` using Horner's method.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> PolyVal<T>(NDArray<T> coeffs, NDArray<T> x) where T : unmanaged, INumber<T>
    {
        if (coeffs.Rank != 1)
            throw new ArgumentException("Polynomial coefficients must be a 1D vector.");
        if (coeffs.Shape[0] == 0)
            throw new ArgumentException("Coefficients vector cannot be empty.");

        int numCoeffs = coeffs.Shape[0];
        var cContig = coeffs.Contiguous();
        var xContig = x.Contiguous();
        var result = new NDArray<T>(x.Shape);

        Parallel.For(0, x.TotalLength, i =>
        {
            T xi = xContig[i];
            T acc = cContig[0];
            for (int c = 1; c < numCoeffs; c++)
            {
                acc = acc * xi + cContig[c];
            }
            result[i] = acc;
        });

        return result;
    }

    /// <summary>
    /// Computes the roots of a polynomial with given coefficients in descending degree order.
    /// For degree 1: linear solution.
    /// For degree 2: quadratic formula with real/complex roots.
    /// For general degree: eigenvalues of the Companion Matrix.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Complex<double>[] Roots(NDArray<double> coeffs)
    {
        if (coeffs.Rank != 1)
            throw new ArgumentException("Coefficients must be a 1D vector.");

        var cContig = coeffs.Contiguous();
        // Skip leading zeros
        int start = 0;
        while (start < cContig.TotalLength && Math.Abs(cContig[start]) < 1e-15)
        {
            start++;
        }

        int remaining = cContig.TotalLength - start;
        if (remaining <= 1)
        {
            return Array.Empty<Complex<double>>();
        }

        int deg = remaining - 1;
        double leading = cContig[start];

        if (deg == 1)
        {
            double r = -cContig[start + 1] / leading;
            return [new Complex<double>(r, 0.0)];
        }

        if (deg == 2)
        {
            double a = leading;
            double b = cContig[start + 1];
            double c = cContig[start + 2];
            double disc = b * b - 4.0 * a * c;

            if (disc >= 0)
            {
                double sqrtD = Math.Sqrt(disc);
                return
                [
                    new Complex<double>((-b + sqrtD) / (2.0 * a), 0.0),
                    new Complex<double>((-b - sqrtD) / (2.0 * a), 0.0)
                ];
            }
            else
            {
                double sqrtD = Math.Sqrt(-disc);
                double realPart = -b / (2.0 * a);
                double imagPart = sqrtD / (2.0 * a);
                return
                [
                    new Complex<double>(realPart, imagPart),
                    new Complex<double>(realPart, -imagPart)
                ];
            }
        }

        // General degree: Companion matrix
        var comp = new NDArray<double>(deg, deg);

        for (int j = 0; j < deg; j++)
        {
            comp[0, j] = -cContig[start + 1 + j] / leading;
        }
        for (int i = 1; i < deg; i++)
        {
            comp[i, i - 1] = 1.0;
        }

        return ComputeEigenvaluesGeneral(comp);
    }

    /// <summary>
    /// Computes the roots of a polynomial and returns an NDArray of shape [N, 2] (Col 0: Real, Col 1: Imaginary).
    /// </summary>
    public static NDArray<double> RootsTensor(NDArray<double> coeffs)
    {
        var roots = Roots(coeffs);
        var res = new NDArray<double>(roots.Length, 2);
        for (int i = 0; i < roots.Length; i++)
        {
            res[i, 0] = roots[i].Real;
            res[i, 1] = roots[i].Imaginary;
        }
        return res;
    }

    private static Complex<double>[] ComputeEigenvaluesGeneral(NDArray<double> mat)
    {
        int n = mat.Shape[0];
        double[,] h = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                h[i, j] = mat[i, j];

        int maxIter = 100 * n;
        int m = n;

        var roots = new List<Complex<double>>();

        while (m > 2 && maxIter > 0)
        {
            maxIter--;
            if (Math.Abs(h[m - 1, m - 2]) <= 1e-12 * (Math.Abs(h[m - 2, m - 2]) + Math.Abs(h[m - 1, m - 1])))
            {
                roots.Add(new Complex<double>(h[m - 1, m - 1], 0.0));
                m--;
                continue;
            }

            double d = (h[m - 2, m - 2] - h[m - 1, m - 1]) / 2.0;
            double sign = d >= 0 ? 1.0 : -1.0;
            double mu = h[m - 1, m - 1] - (h[m - 1, m - 2] * h[m - 2, m - 1]) / (d + sign * Math.Sqrt(d * d + h[m - 1, m - 2] * h[m - 2, m - 1]));

            for (int i = 0; i < m - 1; i++)
            {
                double x = h[i, i] - (i == 0 ? mu : 0);
                double y = h[i + 1, i];
                double r = Math.Sqrt(x * x + y * y);
                if (r < 1e-15) continue;
                double c = x / r;
                double s = -y / r;

                for (int j = 0; j < m; j++)
                {
                    double t1 = c * h[i, j] - s * h[i + 1, j];
                    double t2 = s * h[i, j] + c * h[i + 1, j];
                    h[i, j] = t1;
                    h[i + 1, j] = t2;
                }

                for (int j = 0; j < m; j++)
                {
                    double t1 = c * h[j, i] - s * h[j, i + 1];
                    double t2 = s * h[j, i] + c * h[j, i + 1];
                    h[j, i] = t1;
                    h[j, i + 1] = t2;
                }
            }
        }

        if (m == 2)
        {
            double a = 1.0;
            double b = -(h[0, 0] + h[1, 1]);
            double c = h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0];
            double disc = b * b - 4.0 * a * c;
            if (disc >= 0)
            {
                double sqrtD = Math.Sqrt(disc);
                roots.Add(new Complex<double>((-b + sqrtD) / 2.0, 0.0));
                roots.Add(new Complex<double>((-b - sqrtD) / 2.0, 0.0));
            }
            else
            {
                double sqrtD = Math.Sqrt(-disc);
                roots.Add(new Complex<double>(-b / 2.0, sqrtD / 2.0));
                roots.Add(new Complex<double>(-b / 2.0, -sqrtD / 2.0));
            }
        }
        else if (m == 1)
        {
            roots.Add(new Complex<double>(h[0, 0], 0.0));
        }

        return roots.ToArray();
    }
}

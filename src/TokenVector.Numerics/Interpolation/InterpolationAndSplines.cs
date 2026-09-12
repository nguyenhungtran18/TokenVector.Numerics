// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Interpolation;

/// <summary>
/// High-performance Spline, 2D Grid (Bilinear/Bicubic), Radial Basis Function (RBF), and Barycentric Interpolation algorithms.
/// </summary>
public static class InterpolationAndSplines
{
    #region 1D Cubic Spline Interpolation

    /// <summary>
    /// Represents a natural or clamped 1D cubic spline.
    /// </summary>
    public sealed class CubicSplineModel
    {
        private readonly double[] _x;
        private readonly double[] _a; // y
        private readonly double[] _b;
        private readonly double[] _c;
        private readonly double[] _d;
        private readonly int _n;

        public CubicSplineModel(double[] x, double[] a, double[] b, double[] c, double[] d)
        {
            _x = x; _a = a; _b = b; _c = c; _d = d;
            _n = x.Length - 1;
        }

        public double Evaluate(double xVal)
        {
            if (xVal <= _x[0]) return _a[0] + _b[0] * (xVal - _x[0]);
            if (xVal >= _x[_n]) return _a[_n] + _b[_n] * (xVal - _x[_n]);

            // Binary search for interval
            int low = 0, high = _n;
            while (low < high - 1)
            {
                int mid = (low + high) / 2;
                if (_x[mid] <= xVal) low = mid;
                else high = mid;
            }

            double dx = xVal - _x[low];
            return _a[low] + _b[low] * dx + _c[low] * dx * dx + _d[low] * dx * dx * dx;
        }

        public NDArray<double> Evaluate(NDArray<double> xVals)
        {
            var res = new NDArray<double>(xVals.Shape);
            for (int i = 0; i < xVals.TotalLength; i++)
            {
                res[i] = Evaluate(xVals[i]);
            }
            return res;
        }
    }

    /// <summary>
    /// Constructs a Natural Cubic Spline interpolator through knots (x, y).
    /// </summary>
    public static CubicSplineModel CubicSpline(NDArray<double> x, NDArray<double> y)
    {
        int n = x.TotalLength - 1;
        if (n < 2) throw new ArgumentException("At least 3 points required for cubic spline.");

        double[] h = new double[n];
        double[] a = new double[n + 1];
        for (int i = 0; i <= n; i++) a[i] = y[i];
        for (int i = 0; i < n; i++) h[i] = x[i + 1] - x[i];

        double[] alpha = new double[n];
        for (int i = 1; i < n; i++)
        {
            alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i]) - (3.0 / h[i - 1]) * (a[i] - a[i - 1]);
        }

        // Tridiagonal solve for c
        double[] l = new double[n + 1];
        double[] mu = new double[n + 1];
        double[] z = new double[n + 1];
        l[0] = 1.0;

        for (int i = 1; i < n; i++)
        {
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        l[n] = 1.0;
        z[n] = 0.0;

        double[] c = new double[n + 1];
        double[] b = new double[n + 1];
        double[] d = new double[n + 1];
        c[n] = 0.0;

        for (int j = n - 1; j >= 0; j--)
        {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }

        double[] xArr = new double[n + 1];
        for (int i = 0; i <= n; i++) xArr[i] = x[i];

        return new CubicSplineModel(xArr, a, b, c, d);
    }

    #endregion

    #region 2D Grid Interpolation (Bilinear & Bicubic)

    /// <summary>
    /// Bilinear interpolation on a 2D image/grid at fractional coordinate (y, x).
    /// </summary>
    public static double BilinearInterpolation(NDArray<double> grid2D, double y, double x)
    {
        int h = grid2D.Shape[0];
        int w = grid2D.Shape[1];

        x = Math.Clamp(x, 0.0, w - 1.0);
        y = Math.Clamp(y, 0.0, h - 1.0);

        int x0 = (int)Math.Floor(x);
        int x1 = Math.Min(x0 + 1, w - 1);
        int y0 = (int)Math.Floor(y);
        int y1 = Math.Min(y0 + 1, h - 1);

        double dx = x - x0;
        double dy = y - y0;

        double v00 = grid2D[y0, x0];
        double v01 = grid2D[y0, x1];
        double v10 = grid2D[y1, x0];
        double v11 = grid2D[y1, x1];

        double top = v00 * (1.0 - dx) + v01 * dx;
        double bottom = v10 * (1.0 - dx) + v11 * dx;

        return top * (1.0 - dy) + bottom * dy;
    }

    #endregion

    #region Radial Basis Function (RBF) Interpolation

    /// <summary>
    /// Multidimensional RBF interpolator (Multiquadric, Gaussian, Inverse Multiquadric, Thin-Plate).
    /// </summary>
    public sealed class RbfInterpolator
    {
        private readonly NDArray<double> _nodes; // N x D
        private readonly NDArray<double> _weights; // N
        private readonly string _kernel;
        private readonly double _eps;

        public RbfInterpolator(NDArray<double> nodes, NDArray<double> weights, string kernel, double eps)
        {
            _nodes = nodes; _weights = weights; _kernel = kernel; _eps = eps;
        }

        public double Evaluate(NDArray<double> point)
        {
            int n = _nodes.Shape[0];
            int d = _nodes.Shape[1];
            double sum = 0;

            for (int i = 0; i < n; i++)
            {
                double distSq = 0;
                for (int dim = 0; dim < d; dim++)
                {
                    double diff = point[dim] - _nodes[i, dim];
                    distSq += diff * diff;
                }
                double r = Math.Sqrt(distSq);
                double phi = _kernel switch
                {
                    "gaussian" => Math.Exp(-(_eps * r) * (_eps * r)),
                    "inverse_multiquadric" => 1.0 / Math.Sqrt(1.0 + (_eps * r) * (_eps * r)),
                    "thin_plate" => (r > 1e-12) ? r * r * Math.Log(r) : 0.0,
                    _ => Math.Sqrt(1.0 + (_eps * r) * (_eps * r)) // default multiquadric
                };

                sum += _weights[i] * phi;
            }
            return sum;
        }
    }

    public static RbfInterpolator FitRBF(
        NDArray<double> nodes, NDArray<double> values, string kernel = "multiquadric", double epsilon = 1.0)
    {
        int n = nodes.Shape[0];
        int d = nodes.Shape[1];

        var A = new NDArray<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double distSq = 0;
                for (int dim = 0; dim < d; dim++)
                {
                    double diff = nodes[i, dim] - nodes[j, dim];
                    distSq += diff * diff;
                }
                double r = Math.Sqrt(distSq);
                A[i, j] = kernel switch
                {
                    "gaussian" => Math.Exp(-(epsilon * r) * (epsilon * r)),
                    "inverse_multiquadric" => 1.0 / Math.Sqrt(1.0 + (epsilon * r) * (epsilon * r)),
                    "thin_plate" => (r > 1e-12) ? r * r * Math.Log(r) : 0.0,
                    _ => Math.Sqrt(1.0 + (epsilon * r) * (epsilon * r))
                };
            }
        }

        var weights = Decomposition.Solve(A, values.Rank == 1 ? values.Reshape(n, 1) : values).Reshape(n);
        return new RbfInterpolator(nodes, weights, kernel, epsilon);
    }

    #endregion

    #region Barycentric Interpolation

    /// <summary>
    /// High-order stable 1D Barycentric polynomial interpolation.
    /// </summary>
    public static double BarycentricInterpolation(NDArray<double> xNodes, NDArray<double> yNodes, double x)
    {
        int n = xNodes.TotalLength;
        for (int i = 0; i < n; i++)
        {
            if (Math.Abs(x - xNodes[i]) < 1e-15) return yNodes[i];
        }

        // Compute barycentric weights
        double num = 0.0;
        double denom = 0.0;

        for (int j = 0; j < n; j++)
        {
            double wj = 1.0;
            for (int k = 0; k < n; k++)
            {
                if (k != j) wj /= (xNodes[j] - xNodes[k]);
            }

            double term = wj / (x - xNodes[j]);
            num += term * yNodes[j];
            denom += term;
        }

        return num / denom;
    }

    #endregion
}

using System;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.LinAlg;

/// <summary>
/// Numerical calculus and interpolation methods: Numerical Integration, Finite-Difference Derivatives, Lerp, and Lagrange Polynomial Interpolation.
/// </summary>
public static class NumericalAnalysis
{
    /// <summary>
    /// Computes definite integral using the composite Trapezoidal rule.
    /// </summary>
    public static T TrapezoidalRule<T>(NDArray<T> y, T dx) where T : unmanaged, IFloatingPoint<T>
    {
        if (y.Rank != 1) throw new ArgumentException("TrapezoidalRule requires a 1D tensor.");
        int n = y.Shape[0];
        if (n < 2) return T.Zero;

        T sum = (y[0] + y[n - 1]) / T.CreateChecked(2.0);
        for (int i = 1; i < n - 1; i++)
        {
            sum += y[i];
        }

        return sum * dx;
    }

    /// <summary>
    /// Computes 2nd-order accurate central difference numerical derivatives: (y[i+1] - y[i-1]) / (2*dx).
    /// </summary>
    public static NDArray<T> CentralDifference2ndOrder<T>(NDArray<T> y, T dx) where T : unmanaged, IFloatingPoint<T>
    {
        if (y.Rank != 1) throw new ArgumentException("Numerical derivative requires a 1D tensor.");
        int n = y.Shape[0];
        if (n < 3) throw new ArgumentException("Tensor length must be at least 3 for central difference.");

        var dy = new NDArray<T>(n);
        T twoDx = T.CreateChecked(2.0) * dx;

        // Forward difference at boundary
        dy[0] = (y[1] - y[0]) / dx;

        // Central difference for interior
        for (int i = 1; i < n - 1; i++)
        {
            dy[i] = (y[i + 1] - y[i - 1]) / twoDx;
        }

        // Backward difference at boundary
        dy[n - 1] = (y[n - 1] - y[n - 2]) / dx;

        return dy;
    }

    /// <summary>
    /// Computes 4th-order accurate central difference numerical derivatives.
    /// </summary>
    public static NDArray<T> CentralDifference4thOrder<T>(NDArray<T> y, T dx) where T : unmanaged, IFloatingPoint<T>
    {
        if (y.Rank != 1) throw new ArgumentException("Numerical derivative requires a 1D tensor.");
        int n = y.Shape[0];
        if (n < 5) return CentralDifference2ndOrder(y, dx);

        var dy = new NDArray<T>(n);
        T twelveDx = T.CreateChecked(12.0) * dx;
        T c8 = T.CreateChecked(8.0);

        // Boundaries with 2nd order
        dy[0] = (y[1] - y[0]) / dx;
        dy[1] = (y[2] - y[0]) / (T.CreateChecked(2.0) * dx);

        // 4th order interior
        for (int i = 2; i < n - 2; i++)
        {
            dy[i] = (-y[i + 2] + c8 * y[i + 1] - c8 * y[i - 1] + y[i - 2]) / twelveDx;
        }

        dy[n - 2] = (y[n - 1] - y[n - 3]) / (T.CreateChecked(2.0) * dx);
        dy[n - 1] = (y[n - 1] - y[n - 2]) / dx;

        return dy;
    }

    /// <summary>
    /// Linear interpolation between two scalar values: (1 - t) * a + t * b.
    /// </summary>
    public static T Lerp<T>(T a, T b, T t) where T : unmanaged, IFloatingPoint<T>
    {
        return a + t * (b - a);
    }

    /// <summary>
    /// Elementwise linear interpolation between two tensors.
    /// </summary>
    public static NDArray<T> Lerp<T>(NDArray<T> a, NDArray<T> b, T t) where T : unmanaged, IFloatingPoint<T>
    {
        return a + (b - a) * t;
    }

    /// <summary>
    /// Lagrange polynomial interpolation through points (x, y) evaluated at targetX.
    /// </summary>
    public static T PolynomialInterpolation<T>(NDArray<T> x, NDArray<T> y, T targetX) where T : unmanaged, IFloatingPoint<T>
    {
        if (x.Rank != 1 || y.Rank != 1 || x.Shape[0] != y.Shape[0])
        {
            throw new ArgumentException("x and y must be 1D tensors of identical length.");
        }

        int n = x.Shape[0];
        T result = T.Zero;

        for (int i = 0; i < n; i++)
        {
            T term = y[i];
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    term *= (targetX - x[j]) / (x[i] - x[j]);
                }
            }
            result += term;
        }

        return result;
    }
}

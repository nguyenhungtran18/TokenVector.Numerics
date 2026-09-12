using System;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Spatial;

/// <summary>
/// Non-Euclidean Hyperbolic Geometry algorithms for Poincaré ball and Lorentz / Hyperboloid models.
/// Widely used in modern Graph Neural Networks, hierarchical representation learning, and general relativity.
/// </summary>
public static class HyperbolicGeometry
{
    private const double DefaultEps = 1e-7;

    /// <summary>
    /// Computes the geodesic distance between two points in the Poincaré ball model with curvature c.
    /// d_c(u, v) = (2 / sqrt(c)) * artanh( sqrt(c) * || -u (+) v || )
    /// </summary>
    public static double PoincareDistance(NDArray<double> u, NDArray<double> v, double c = 1.0, double eps = DefaultEps)
    {
        if (u.TotalLength != v.TotalLength)
            throw new ArgumentException("Vectors u and v must have the same dimension.");

        int dim = u.TotalLength;
        double normUSq = 0, normVSq = 0, diffSq = 0;

        for (int i = 0; i < dim; i++)
        {
            double ui = u[i];
            double vi = v[i];
            double d = ui - vi;

            normUSq += ui * ui;
            normVSq += vi * vi;
            diffSq += d * d;
        }

        // Clamp norms within the ball of radius 1/sqrt(c) - eps
        double maxNormSq = (1.0 / c) - eps;
        normUSq = Math.Min(normUSq, maxNormSq);
        normVSq = Math.Min(normVSq, maxNormSq);

        double denom = (1.0 - c * normUSq) * (1.0 - c * normVSq);
        if (denom < 1e-15) denom = 1e-15;

        double delta = 2.0 * c * diffSq / denom;
        double arg = 1.0 + delta;

        // arcosh(arg) = ln(arg + sqrt(arg^2 - 1))
        double arcosh = Math.Log(arg + Math.Sqrt(Math.Max(0.0, arg * arg - 1.0)));
        return arcosh / Math.Sqrt(c);
    }

    /// <summary>
    /// Computes Möbius addition u (+) v on the Poincaré ball with curvature c.
    /// u (+) v = ((1 + 2c&lt;u,v&gt; + c||v||^2)u + (1 - c||u||^2)v) / (1 + 2c&lt;u,v&gt; + c^2||u||^2||v||^2)
    /// </summary>
    public static NDArray<double> MobiusAddition(NDArray<double> u, NDArray<double> v, double c = 1.0)
    {
        if (u.TotalLength != v.TotalLength)
            throw new ArgumentException("Vectors u and v must have the same dimension.");

        int dim = u.TotalLength;
        double uDotV = 0, uSq = 0, vSq = 0;

        for (int i = 0; i < dim; i++)
        {
            double ui = u[i];
            double vi = v[i];
            uDotV += ui * vi;
            uSq += ui * ui;
            vSq += vi * vi;
        }

        double alpha = 1.0 + 2.0 * c * uDotV + c * vSq;
        double beta = 1.0 - c * uSq;
        double denom = 1.0 + 2.0 * c * uDotV + c * c * uSq * vSq;
        if (Math.Abs(denom) < 1e-15) denom = 1e-15;

        NDArray<double> result = new NDArray<double>(u.Shape);
        for (int i = 0; i < dim; i++)
        {
            result[i] = (alpha * u[i] + beta * v[i]) / denom;
        }

        return result;
    }

    /// <summary>
    /// Exponential map at origin projecting tangent space vector v onto the Poincaré ball.
    /// exp_0^c(v) = tanh(sqrt(c) ||v||) * (v / (sqrt(c) ||v||))
    /// </summary>
    public static NDArray<double> ExpMap0(NDArray<double> v, double c = 1.0)
    {
        double norm = 0;
        for (int i = 0; i < v.TotalLength; i++)
            norm += v[i] * v[i];
        norm = Math.Sqrt(norm);

        if (norm < 1e-12)
            return v.Clone();

        double sqrtC = Math.Sqrt(c);
        double scale = Math.Tanh(sqrtC * norm) / (sqrtC * norm);

        NDArray<double> res = new NDArray<double>(v.Shape);
        for (int i = 0; i < v.TotalLength; i++)
            res[i] = v[i] * scale;

        return res;
    }

    /// <summary>
    /// Logarithmic map at origin mapping a point y in the Poincaré ball back to the tangent space.
    /// log_0^c(y) = artanh(sqrt(c) ||y||) * (y / (sqrt(c) ||y||))
    /// </summary>
    public static NDArray<double> LogMap0(NDArray<double> y, double c = 1.0)
    {
        double norm = 0;
        for (int i = 0; i < y.TotalLength; i++)
            norm += y[i] * y[i];
        norm = Math.Sqrt(norm);

        if (norm < 1e-12)
            return y.Clone();

        double sqrtC = Math.Sqrt(c);
        double x = Math.Min(sqrtC * norm, 1.0 - 1e-12);
        // artanh(x) = 0.5 * ln((1+x)/(1-x))
        double artanh = 0.5 * Math.Log((1.0 + x) / (1.0 - x));
        double scale = artanh / (sqrtC * norm);

        NDArray<double> res = new NDArray<double>(y.Shape);
        for (int i = 0; i < y.TotalLength; i++)
            res[i] = y[i] * scale;

        return res;
    }
}

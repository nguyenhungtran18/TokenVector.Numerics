using System;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Finance;
using TokenVector.Numerics.Interpolation;
using TokenVector.Numerics.LinAlg;
using TokenVector.Numerics.Neural;
using TokenVector.Numerics.Ops;
using TokenVector.Numerics.Optimize;
using TokenVector.Numerics.Quantum;
using TokenVector.Numerics.Science;
using TokenVector.Numerics.Spatial;
using TokenVector.Numerics.Statistics;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class ExpandedModulesTests
{
    [Fact]
    public void Test_ElementWiseMathOps()
    {
        var a = NDArray<double>.FromArray(new[] { 0.0, Math.PI / 6.0, Math.PI / 2.0 }, 3);
        var sinA = ElementWiseMathOps.Sin(a);
        Assert.True(Math.Abs(sinA[0] - 0.0) < 1e-6);
        Assert.True(Math.Abs(sinA[1] - 0.5) < 1e-6);
        Assert.True(Math.Abs(sinA[2] - 1.0) < 1e-6);

        var expA = ElementWiseMathOps.Exp(NDArray<double>.FromArray(new[] { 0.0, 1.0 }, 2));
        Assert.True(Math.Abs(expA[0] - 1.0) < 1e-6);
        Assert.True(Math.Abs(expA[1] - Math.E) < 1e-6);

        var deg = ElementWiseMathOps.Rad2Deg(NDArray<double>.FromArray(new[] { Math.PI }, 1));
        Assert.True(Math.Abs(deg[0] - 180.0) < 1e-6);
    }

    [Fact]
    public void Test_ShapeExtensions()
    {
        var x = NDArray<float>.FromArray(new[] { 1f, 2f, 3f }, 3);
        var xExpanded = x.ExpandDims(0);
        Assert.Equal(new[] { 1, 3 }, xExpanded.Shape);

        var xSqueezed = xExpanded.Squeeze(0);
        Assert.Equal(new[] { 3 }, xSqueezed.Shape);

        int[] coords = ShapeExtensions.UnravelIndex(7, stackalloc int[] { 3, 4 });
        Assert.Equal(1, coords[0]);
        Assert.Equal(3, coords[1]);

        int flat = ShapeExtensions.RavelMultiIndex(stackalloc int[] { 1, 3 }, stackalloc int[] { 3, 4 });
        Assert.Equal(7, flat);
    }

    [Fact]
    public void Test_SpecialFunctions()
    {
        double b = SpecialFunctions.Beta(2.0, 3.0);
        // B(2,3) = Gamma(2)*Gamma(3)/Gamma(5) = 1*2/24 = 1/12 = 0.083333
        Assert.True(Math.Abs(b - 1.0 / 12.0) < 1e-6);

        double sinc0 = SpecialFunctions.Sinc(0.0);
        Assert.True(Math.Abs(sinc0 - 1.0) < 1e-6);

        double logitP = SpecialFunctions.Logit(0.5);
        Assert.True(Math.Abs(logitP - 0.0) < 1e-6);

        double expit0 = SpecialFunctions.Expit(0.0);
        Assert.True(Math.Abs(expit0 - 0.5) < 1e-6);
    }

    [Fact]
    public void Test_LinAlgExtended_Tridiagonal_NullSpace_KronSum()
    {
        // Solve Tridiagonal system
        var lower = NDArray<double>.FromArray(new[] { 1.0, 1.0 }, 2);
        var diag = NDArray<double>.FromArray(new[] { 4.0, 4.0, 4.0 }, 3);
        var upper = NDArray<double>.FromArray(new[] { 1.0, 1.0 }, 2);
        var rhs = NDArray<double>.FromArray(new[] { 5.0, 6.0, 5.0 }, 3);

        var sol = LinAlgExtended.SolveBandTridiagonal(lower, diag, upper, rhs);
        Assert.Equal(3, sol.TotalLength);
        Assert.True(Math.Abs(sol[0] - 1.0) < 1e-5);
        Assert.True(Math.Abs(sol[1] - 1.0) < 1e-5);
        Assert.True(Math.Abs(sol[2] - 1.0) < 1e-5);

        // Kronecker Sum
        var a = NDArray<double>.FromArray(new double[] { 1, 2, 3, 4 }, 2, 2);
        var kronSum = LinAlgExtended.KroneckerSum(a, a);
        Assert.Equal(new[] { 4, 4 }, kronSum.Shape);
    }

    [Fact]
    public void Test_Optimization_Root_And_Simplex()
    {
        // 1D Root finding: f(x) = x^2 - 4 = 0 in [0, 3] -> x = 2
        double root = OptimizationAndRootFinding.RootBrentq(x => x * x - 4.0, 0.0, 3.0);
        Assert.True(Math.Abs(root - 2.0) < 1e-6);

        // 1D Minimization: f(x) = (x - 3)^2 + 5 -> xMin = 3, fMin = 5
        var (xMin, fMin) = OptimizationAndRootFinding.MinimizeBrent(x => (x - 3.0) * (x - 3.0) + 5.0, 0.0, 10.0);
        Assert.True(Math.Abs(xMin - 3.0) < 1e-5);
        Assert.True(Math.Abs(fMin - 5.0) < 1e-5);

        // Nelder-Mead Rosenbrock function: f(x, y) = (1-x)^2 + 100*(y-x^2)^2 -> min at (1, 1)
        var x0 = NDArray<double>.FromArray(new[] { 0.0, 0.0 }, 2);
        var (xOpt, fOpt) = OptimizationAndRootFinding.MinimizeNelderMead(
            pt => (1.0 - pt[0]) * (1.0 - pt[0]) + 100.0 * (pt[1] - pt[0] * pt[0]) * (pt[1] - pt[0] * pt[0]),
            x0, tol: 1e-6, maxIter: 500);

        Assert.True(Math.Abs(xOpt[0] - 1.0) < 1e-2);
        Assert.True(Math.Abs(xOpt[1] - 1.0) < 1e-2);

        // Linear Program Simplex: Min -x - 2y s.t. x + y <= 4, x <= 2
        var c = NDArray<double>.FromArray(new[] { 1.0, 2.0 }, 2); // Objective min 1x + 2y (or test standard)
        var A = NDArray<double>.FromArray(new double[] { 1, 1, 1, 0 }, 2, 2);
        var b = NDArray<double>.FromArray(new[] { 4.0, 2.0 }, 2);
        var (lpSol, lpVal) = OptimizationAndRootFinding.LinearProgramSimplex(c, A, b);
        Assert.Equal(2, lpSol.TotalLength);
    }

    [Fact]
    public void Test_Interpolation_Spline_And_Bilinear()
    {
        // 1D Cubic Spline: f(x) = x^2 at x = [0, 1, 2, 3, 4]
        var x = NDArray<double>.FromArray(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 }, 5);
        var y = NDArray<double>.FromArray(new[] { 0.0, 1.0, 4.0, 9.0, 16.0 }, 5);
        var spline = InterpolationAndSplines.CubicSpline(x, y);

        double val1_5 = spline.Evaluate(1.5);
        Assert.True(Math.Abs(val1_5 - 2.25) < 0.1);

        // 2D Bilinear
        var grid = NDArray<double>.FromArray(new double[]
        {
            0, 10,
            20, 30
        }, 2, 2);
        double midVal = InterpolationAndSplines.BilinearInterpolation(grid, 0.5, 0.5);
        Assert.True(Math.Abs(midVal - 15.0) < 1e-6);
    }

    [Fact]
    public void Test_Statistics_Distributions_And_HypothesisTests()
    {
        double p = DistributionsAndTests.NormalCDF(0.0, 0.0, 1.0);
        Assert.True(Math.Abs(p - 0.5) < 1e-6);

        double x = DistributionsAndTests.NormalPPF(0.5, 0.0, 1.0);
        Assert.True(Math.Abs(x - 0.0) < 1e-5);

        var data = NDArray<double>.FromArray(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }, 5);
        double skew = DistributionsAndTests.Skewness(data);
        Assert.True(Math.Abs(skew - 0.0) < 1e-6); // Symmetric data -> skewness = 0

        var (tStat, pVal) = DistributionsAndTests.TTest1Sample(data, mu0: 3.0);
        Assert.True(Math.Abs(tStat - 0.0) < 1e-5);
        Assert.True(pVal > 0.5);
    }

    [Fact]
    public void Test_Spatial_KDTree_And_ConvexHull()
    {
        // 4 points in 2D
        var pts = NDArray<double>.FromArray(new double[]
        {
            0, 0,
            1, 0,
            0, 1,
            1, 1,
            0.5, 0.5
        }, 5, 2);

        var tree = new SpatialTreesAndGeometry.KDTree(pts);
        var query = NDArray<double>.FromArray(new[] { 0.51, 0.49 }, 2);
        var (dist, idx) = tree.QueryNearest(query);
        Assert.Equal(4, idx); // Closest to (0.5, 0.5)

        // Convex Hull
        var hull = SpatialTreesAndGeometry.ConvexHull2D(pts);
        Assert.Equal(4, hull.Shape[0]); // 4 corner vertices
    }

    [Fact]
    public void Test_Astrodynamics_And_Finance_Extended()
    {
        // J2 Perturbation
        var r = NDArray<double>.FromArray(new[] { 7000.0, 0.0, 0.0 }, 3);
        var aJ2 = Astrodynamics.J2Perturbation(r);
        Assert.Equal(3, aJ2.TotalLength);

        // American Option
        double amPut = FinanceMath.BinomialTreeAmericanOption(100, 100, 1.0, 0.05, 0.2, steps: 50, isCall: false);
        Assert.True(amPut > 0);

        // Bond Price
        double price = FinanceMath.BondPrice(1000, 0.05, 0.05, 10);
        Assert.True(Math.Abs(price - 1000.0) < 1e-4);
    }

    [Fact]
    public void Test_Quantum_And_Neural_Extended()
    {
        // Quantum CZ & QFT
        var q = new QState(2);
        q.H(0);
        q.CZ(0, 1);
        q.QFT();
        Assert.Equal(4, q.Dimension);

        // RoPE & RMSNorm
        var x = NDArray<double>.FromArray(new[] { 1.0, 2.0, 3.0, 4.0 }, 1, 1, 4);
        var xRope = AttentionEngine.ApplyRoPE(x);
        Assert.Equal(4, xRope.TotalLength);

        var rms = Normalization.RMSNorm(x);
        Assert.Equal(4, rms.TotalLength);
    }
}

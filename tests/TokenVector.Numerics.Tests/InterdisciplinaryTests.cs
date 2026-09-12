// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using TokenVector.Numerics.Core;
using TokenVector.Numerics.Finance;
using TokenVector.Numerics.LinAlg;
using TokenVector.Numerics.Physics;
using TokenVector.Numerics.Science;
using TokenVector.Numerics.Spatial;
using TokenVector.Numerics.Statistics;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class InterdisciplinaryTests
{
    [Fact]
    public void Test_Astrodynamics_Kepler_Hohmann_WGS84()
    {
        // 1. Solve Kepler: M = E - e*sin(E) for e = 0.1, M = 1.0
        double e = 0.1;
        double m = 1.0;
        double E = Astrodynamics.SolveKepler(m, e);
        double reconstructedM = E - e * Math.Sin(E);
        Assert.Equal(m, reconstructedM, precision: 8);

        // 2. Hohmann Transfer between LEO (r1 = 6378 + 300 km) and GEO (r2 = 42164 km)
        double rLEO = 6678.137;
        double rGEO = 42164.0;
        var (dv1, dv2, totalDv, tof) = Astrodynamics.HohmannTransfer(rLEO, rGEO);
        Assert.True(dv1 > 2.0 && dv1 < 3.0); // ~2.42 km/s
        Assert.True(dv2 > 1.0 && dv2 < 2.0); // ~1.46 km/s
        Assert.True(totalDv > 3.5 && totalDv < 4.5); // ~3.89 km/s
        Assert.True(tof > 18000.0); // ~5.3 hours = ~19000 s

        // 3. Geodetic WGS84 roundtrip: Equator lat = 0, lon = 105.0, alt = 0.5 km
        var ecef = Astrodynamics.GeodeticToEcef(0.0, 105.0, 0.5);
        var (lat, lon, alt) = Astrodynamics.EcefToGeodetic(ecef[0], ecef[1], ecef[2]);
        Assert.Equal(0.0, lat, precision: 5);
        Assert.Equal(105.0, lon, precision: 5);
        Assert.Equal(0.5, alt, precision: 4);
    }

    [Fact]
    public void Test_FinanceMath_BlackScholes_Markowitz_NPV_IRR()
    {
        // 1. Black-Scholes Call & Put: S = 100, K = 100, T = 1 year, r = 5%, sigma = 20%
        double call = FinanceMath.BlackScholesCall(100.0, 100.0, 1.0, 0.05, 0.20);
        double put = FinanceMath.BlackScholesPut(100.0, 100.0, 1.0, 0.05, 0.20);

        // Put-Call Parity: C - P = S - K * exp(-r*T)
        double diff = call - put;
        double expectedDiff = 100.0 - 100.0 * Math.Exp(-0.05 * 1.0);
        Assert.Equal(expectedDiff, diff, precision: 4);

        // Greeks
        var (delta, gamma, vega, theta, rho) = FinanceMath.OptionGreeks(100.0, 100.0, 1.0, 0.05, 0.20, isCall: true);
        Assert.True(delta > 0.5 && delta < 0.7);
        Assert.True(gamma > 0.0);
        Assert.True(vega > 0.0);

        // 2. Markowitz Portfolio
        var weights = NDArray<double>.FromArray([0.6, 0.4], 2);
        var returns = NDArray<double>.FromArray([0.10, 0.15], 2);
        var cov = NDArray<double>.FromArray([0.04, 0.01, 0.01, 0.09], 2, 2);

        double pRet = FinanceMath.PortfolioReturn(weights, returns);
        Assert.Equal(0.12, pRet, precision: 4);

        double pVol = FinanceMath.PortfolioVolatility(weights, cov);
        Assert.True(pVol > 0.10 && pVol < 0.25);

        double sharpe = FinanceMath.SharpeRatio(weights, returns, cov, riskFreeRate: 0.02);
        Assert.True(sharpe > 0.0);

        // 3. NPV and IRR
        var cashFlows = NDArray<double>.FromArray([-100.0, 30.0, 40.0, 50.0, 20.0], 5);
        double npv = FinanceMath.NPV(0.10, cashFlows);
        double irr = FinanceMath.IRR(cashFlows);
        Assert.True(irr > 0.10 && irr < 0.20); // ~14.5%
        Assert.Equal(0.0, FinanceMath.NPV(irr, cashFlows), precision: 4);
    }

    [Fact]
    public void Test_TimeSeriesAndKalman()
    {
        // 1. Kalman Filter 1D
        var kf = new TimeSeriesAndKalman.KalmanFilter1D(initialState: 0.0, initialVariance: 1.0, processNoise: 0.01, measurementNoise: 0.1);
        kf.Predict();
        double state1 = kf.Update(10.0);
        Assert.True(state1 > 0.0 && state1 < 10.0);

        // 2. Kalman Filter ND: 2D State [Position, Velocity]
        var x0 = NDArray<double>.FromArray([0.0, 1.0], 2, 1);
        var p0 = NDArray<double>.Eye(2);
        var f = NDArray<double>.FromArray([1.0, 1.0, 0.0, 1.0], 2, 2); // dt = 1
        var h = NDArray<double>.FromArray([1.0, 0.0], 1, 2); // Observe position only
        var q = NDArray<double>.Eye(2) * 0.01;
        var r = NDArray<double>.FromArray([0.1], 1, 1);

        var kfNd = new TimeSeriesAndKalman.KalmanFilterND(x0, p0, f, h, q, r);
        kfNd.Predict();
        var z = NDArray<double>.FromArray([1.05], 1, 1);
        var estimated = kfNd.Update(z);
        Assert.Equal(2, estimated.Shape[0]);

        // 3. Holt Linear Trend
        var series = NDArray<double>.FromArray([10.0, 12.0, 14.0, 16.0, 18.0], 5);
        var (fitted, forecast) = TimeSeriesAndKalman.HoltLinearTrend(series, alpha: 0.8, beta: 0.2, forecastSteps: 2);
        Assert.Equal(2, forecast.TotalLength);
        Assert.True(forecast[0] > 18.0);
        Assert.True(forecast[1] > forecast[0]);

        // 4. ACF & PACF
        var acf = TimeSeriesAndKalman.Autocorrelation(series, maxLag: 2);
        Assert.Equal(1.0, acf[0], precision: 5);
        var pacf = TimeSeriesAndKalman.PartialAutocorrelation(series, maxLag: 2);
        Assert.Equal(1.0, pacf[0], precision: 5);
    }

    [Fact]
    public void Test_PhysicsODEAndFields()
    {
        // 1. RK4 on simple harmonic oscillator: y = [pos, vel], y' = [vel, -pos]
        // y(0) = [1, 0] => analytical solution is cos(t)
        var y0 = NDArray<double>.FromArray([1.0, 0.0], 2);
        Func<double, NDArray<double>, NDArray<double>> f = (t, y) =>
        {
            return NDArray<double>.FromArray([y[1], -y[0]], 2);
        };

        var (timestamps, trajectory) = PhysicsODEAndFields.SolveRK4(f, 0.0, Math.PI, y0, numSteps: 100);
        var finalState = trajectory[^1];
        // At t = pi, pos = cos(pi) = -1, vel = -sin(pi) = 0
        Assert.Equal(-1.0, finalState[0], precision: 3);
        Assert.Equal(0.0, finalState[1], precision: 3);

        // 2. 3D Vector Calculus
        // Scalar field f(x, y, z) = x^2 + y^2 + z^2 on 5x5x5 grid
        var grid = new NDArray<double>(5, 5, 5);
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++)
                for (int z = 0; z < 5; z++)
                    grid[x, y, z] = x * x + y * y + z * z;

        var laplacian = PhysicsODEAndFields.Laplacian3D(grid, 1.0, 1.0, 1.0);
        // Laplacian of x^2 + y^2 + z^2 is 2 + 2 + 2 = 6
        Assert.Equal(6.0, laplacian[2, 2, 2], precision: 2);
    }

    [Fact]
    public void Test_Geometry3DAndPointClouds()
    {
        // 1. Point to plane distance: plane z = 0 (point (0,0,0), normal (0,0,1)), point (3, 4, 5)
        var p = NDArray<double>.FromArray([3.0, 4.0, 5.0], 3);
        var planePt = NDArray<double>.FromArray([0.0, 0.0, 0.0], 3);
        var planeNorm = NDArray<double>.FromArray([0.0, 0.0, 1.0], 3);
        double dist = Geometry3DAndPointClouds.PointToPlaneDistance(p, planePt, planeNorm);
        Assert.Equal(5.0, dist, precision: 5);

        // 2. Ray-Triangle intersection
        var rayOrig = NDArray<double>.FromArray([0.0, 0.0, -5.0], 3);
        var rayDir = NDArray<double>.FromArray([0.0, 0.0, 1.0], 3);
        var v0 = NDArray<double>.FromArray([-1.0, -1.0, 0.0], 3);
        var v1 = NDArray<double>.FromArray([1.0, -1.0, 0.0], 3);
        var v2 = NDArray<double>.FromArray([0.0, 1.0, 0.0], 3);

        var (hasHit, t, u, v) = Geometry3DAndPointClouds.RayTriangleIntersect(rayOrig, rayDir, v0, v1, v2);
        Assert.True(hasHit);
        Assert.Equal(5.0, t, precision: 4);

        // 3. Kabsch point cloud alignment
        var src = NDArray<double>.FromArray([
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0
        ], 3, 3);

        // Target is translated by (10, 20, 30)
        var tgt = NDArray<double>.FromArray([
            10.0, 20.0, 30.0,
            11.0, 20.0, 30.0,
            10.0, 21.0, 30.0
        ], 3, 3);

        var (rot, trans) = Geometry3DAndPointClouds.AlignPointCloudsKabsch(src, tgt);
        Assert.Equal(10.0, trans[0], precision: 4);
        Assert.Equal(20.0, trans[1], precision: 4);
        Assert.Equal(30.0, trans[2], precision: 4);
    }

    [Fact]
    public void Test_MatrixFunctions_Expm_Sqrtm_Sylvester()
    {
        // 1. Expm of zero matrix = Identity
        var zeros = NDArray<double>.Zeros(3, 3);
        var expmZeros = MatrixFunctions.Expm(zeros);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(i == j ? 1.0 : 0.0, expmZeros[i, j], precision: 5);

        // Expm of diagonal matrix diag([1, 2]) = diag([e^1, e^2])
        var diagMat = NDArray<double>.FromArray([1.0, 0.0, 0.0, 2.0], 2, 2);
        var expmDiag = MatrixFunctions.Expm(diagMat);
        Assert.Equal(Math.Exp(1.0), expmDiag[0, 0], precision: 4);
        Assert.Equal(Math.Exp(2.0), expmDiag[1, 1], precision: 4);

        // 2. Sqrtm: S * S = A
        var a = NDArray<double>.FromArray([4.0, 1.0, 0.0, 9.0], 2, 2);
        var s = MatrixFunctions.Sqrtm(a);
        var sSq = MatrixMultiplication.MatMul(s, s);
        Assert.Equal(4.0, sSq[0, 0], precision: 4);
        Assert.Equal(1.0, sSq[0, 1], precision: 4);
        Assert.Equal(9.0, sSq[1, 1], precision: 4);

        // 3. Sylvester Equation: AX + XB = C
        var A = NDArray<double>.FromArray([2.0, 0.0, 0.0, 3.0], 2, 2);
        var B = NDArray<double>.FromArray([1.0, 0.0, 0.0, 4.0], 2, 2);
        var C = NDArray<double>.FromArray([6.0, 0.0, 0.0, 14.0], 2, 2);

        var X = MatrixFunctions.SolveSylvester(A, B, C);
        // (A_11 + B_11) * X_11 = (2 + 1)*X_11 = 6 => X_11 = 2
        // (A_22 + B_22) * X_22 = (3 + 4)*X_22 = 14 => X_22 = 2
        Assert.Equal(2.0, X[0, 0], precision: 4);
        Assert.Equal(2.0, X[1, 1], precision: 4);
    }
}

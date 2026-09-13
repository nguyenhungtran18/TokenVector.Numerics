// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;
using TokenVector.Numerics.Optimize;
using TokenVector.Numerics.Physics;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class Version101Tests
{
    [Fact]
    public void TestLambertW_Branches()
    {
        // Branch 0
        Assert.Equal(0.0, SpecialFunctions.LambertW(0.0, 0), precision: 10);
        Assert.Equal(1.0, SpecialFunctions.LambertW(Math.E, 0), precision: 10);
        Assert.Equal(2.0, SpecialFunctions.LambertW(2.0 * Math.Exp(2.0), 0), precision: 10);
        Assert.Equal(-1.0, SpecialFunctions.LambertW(-1.0 / Math.E, 0), precision: 10);

        // Branch -1
        Assert.Equal(-1.0, SpecialFunctions.LambertW(-1.0 / Math.E, -1), precision: 10);
        double wMinus1 = SpecialFunctions.LambertW(-0.1, -1);
        Assert.Equal(-3.577152063957297, wMinus1, precision: 6);

        // Tensor version
        var tensor = NDArray<double>.FromArray([0.0, Math.E, 2.0 * Math.Exp(2.0)], 3);
        var res = SpecialFunctions.LambertW(tensor, 0);
        Assert.Equal(0.0, res[0], precision: 10);
        Assert.Equal(1.0, res[1], precision: 10);
        Assert.Equal(2.0, res[2], precision: 10);
    }

    [Fact]
    public void TestBesselAndAiryFunctions()
    {
        // Bessel Y0 and K0
        double y0 = SpecialFunctions.BesselY0(1.0);
        Assert.Equal(0.0882, y0, precision: 3);

        double k0 = SpecialFunctions.BesselK0(1.0);
        Assert.Equal(0.4210, k0, precision: 3);

        // Airy Ai and Bi
        double ai0 = SpecialFunctions.AiryAi(0.0);
        Assert.Equal(0.35502805, ai0, precision: 5);

        double bi0 = SpecialFunctions.AiryBi(0.0);
        Assert.Equal(0.61492663, bi0, precision: 5);

        // Tensor forms
        var tensor = NDArray<double>.FromArray([1.0, 2.0], 2);
        var y0Tensor = SpecialFunctions.BesselY0(tensor);
        var k0Tensor = SpecialFunctions.BesselK0(tensor);
        var aiTensor = SpecialFunctions.AiryAi(tensor);
        var biTensor = SpecialFunctions.AiryBi(tensor);

        Assert.Equal(2, y0Tensor.TotalLength);
        Assert.Equal(2, k0Tensor.TotalLength);
        Assert.Equal(2, aiTensor.TotalLength);
        Assert.Equal(2, biTensor.TotalLength);
    }

    [Fact]
    public void TestSchurDecomposition()
    {
        var a = NDArray<double>.FromArray([
            3.0, -2.0,
            1.0,  0.0
        ], 2, 2);

        var (q, t) = Decomposition.Schur(a);

        // Check Q is orthogonal: Q^T * Q = I
        var qT = q.MatrixTranspose();
        var qqT = MatrixMultiplication.MatMul(qT, q);
        Assert.Equal(1.0, qqT[0, 0], precision: 4);
        Assert.Equal(0.0, qqT[0, 1], precision: 4);
        Assert.Equal(0.0, qqT[1, 0], precision: 4);
        Assert.Equal(1.0, qqT[1, 1], precision: 4);

        // Check A = Q * T * Q^T
        var qt = MatrixMultiplication.MatMul(q, t);
        var reconstructed = MatrixMultiplication.MatMul(qt, qT);

        Assert.Equal(a[0, 0], reconstructed[0, 0], precision: 4);
        Assert.Equal(a[0, 1], reconstructed[0, 1], precision: 4);
        Assert.Equal(a[1, 0], reconstructed[1, 0], precision: 4);
        Assert.Equal(a[1, 1], reconstructed[1, 1], precision: 4);
    }

    [Fact]
    public void TestSolveSylvester()
    {
        var a = NDArray<double>.FromArray([
            2.0, 1.0,
            0.0, 3.0
        ], 2, 2);

        var b = NDArray<double>.FromArray([
            1.0, 0.0,
            2.0, 4.0
        ], 2, 2);

        var c = NDArray<double>.FromArray([
            5.0, 6.0,
            7.0, 8.0
        ], 2, 2);

        // Solve A * X + X * B = C
        var x = Decomposition.SolveSylvester(a, b, c);

        var ax = MatrixMultiplication.MatMul(a, x);
        var xb = MatrixMultiplication.MatMul(x, b);
        var result = ax + xb;

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(c[i, j], result[i, j], precision: 4);
            }
        }
    }

    [Fact]
    public void TestWaveletTransform_Haar_Reconstruction()
    {
        var signal = NDArray<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 8);
        var (cA, cD) = SignalProcessing.DWT(signal, "haar");

        Assert.Equal(4, cA.TotalLength);
        Assert.Equal(4, cD.TotalLength);

        var reconstructed = SignalProcessing.IDWT(cA, cD, "haar");
        Assert.Equal(8, reconstructed.TotalLength);

        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(signal[i], reconstructed[i], precision: 6);
        }
    }

    [Fact]
    public void TestHilbertTransform()
    {
        int n = 32;
        var signal = new NDArray<double>(n);
        for (int i = 0; i < n; i++)
        {
            double t = 2.0 * Math.PI * i / n;
            signal[i] = Math.Cos(t);
        }

        var (realPart, imagPart) = SignalProcessing.AnalyticSignal(signal);

        // Real part matches original signal
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(signal[i], realPart[i], precision: 4);
        }

        // Imaginary part should be sin(t) for cos(t)
        for (int i = 0; i < n; i++)
        {
            double t = 2.0 * Math.PI * i / n;
            Assert.Equal(Math.Sin(t), imagPart[i], precision: 4);
        }
    }

    [Fact]
    public void TestQPSolve_EqualityConstrained()
    {
        // min 0.5 * (x1^2 + x2^2) s.t. x1 + x2 = 1 => x* = [0.5, 0.5]
        var P = NDArray<double>.Eye(2);
        var q = NDArray<double>.Zeros(2);
        var A = NDArray<double>.FromArray([1.0, 1.0], 1, 2);
        var b = NDArray<double>.FromArray([1.0], 1);

        var (xOpt, minVal) = OptimizationAndRootFinding.QPSolve(P, q, A: A, b: b);

        Assert.Equal(0.5, xOpt[0], precision: 3);
        Assert.Equal(0.5, xOpt[1], precision: 3);
        Assert.Equal(0.25, minVal, precision: 3);
    }

    [Fact]
    public void TestQPSolve_BoxBounded()
    {
        // min 0.5 * (x1^2 + x2^2) - 2*x1 - 2*x2 with 0 <= x1, x2 <= 1 => x* = [1.0, 1.0]
        var P = NDArray<double>.Eye(2);
        var q = NDArray<double>.FromArray([-2.0, -2.0], 2);
        var lb = NDArray<double>.FromArray([0.0, 0.0], 2);
        var ub = NDArray<double>.FromArray([1.0, 1.0], 2);

        var (xOpt, minVal) = OptimizationAndRootFinding.QPSolve(P, q, lb: lb, ub: ub);

        Assert.Equal(1.0, xOpt[0], precision: 3);
        Assert.Equal(1.0, xOpt[1], precision: 3);
        Assert.Equal(-3.0, minVal, precision: 3);
    }

    [Fact]
    public void TestSymplecticVerlet_EnergyConservation()
    {
        // Harmonic oscillator: F(q) = -k * q, mass m = 1, k = 1
        // E = 0.5 * p^2 + 0.5 * q^2 = constant
        Func<NDArray<double>, NDArray<double>> force = q => q * (-1.0);

        var q0 = NDArray<double>.FromArray([1.0], 1);
        var p0 = NDArray<double>.FromArray([0.0], 1);

        double t0 = 0.0;
        double tEnd = 20.0 * Math.PI; // 10 complete oscillation cycles
        int steps = 1000;

        var (ts, qs, ps) = PhysicsODEAndFields.SolveSymplecticVerlet(force, t0, tEnd, q0, p0, mass: 1.0, numSteps: steps);

        Assert.Equal(steps + 1, ts.Length);

        double initialEnergy = 0.5 * p0[0] * p0[0] + 0.5 * q0[0] * q0[0]; // 0.5

        for (int i = 0; i <= steps; i++)
        {
            double qVal = qs[i][0];
            double pVal = ps[i][0];
            double energy = 0.5 * pVal * pVal + 0.5 * qVal * qVal;

            // Symplectic integrator preserves energy without dampening or blowing up
            Assert.True(Math.Abs(energy - initialEnergy) < 1e-2, $"Energy drifted at step {i}: {energy}");
        }
    }
}

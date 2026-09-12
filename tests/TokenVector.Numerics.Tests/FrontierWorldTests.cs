using System;
using TokenVector.Numerics.Biology;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Crypto;
using TokenVector.Numerics.Graphs;
using TokenVector.Numerics.Neural;
using TokenVector.Numerics.Physics;
using TokenVector.Numerics.Quantum;
using TokenVector.Numerics.Robotics;
using TokenVector.Numerics.Spatial;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class FrontierWorldTests
{
    [Fact]
    public void Test_Quantum_QState_BellState_And_Entropy()
    {
        // 2-qubit state initialized to |00>
        var q = new QState(2);
        Assert.Equal(4, q.Dimension);

        // Create Bell state (|00> + |11>) / sqrt(2)
        q.H(0);
        q.CNOT(0, 1);

        var probs = q.GetProbabilities();
        Assert.True(Math.Abs(probs[0] - 0.5) < 1e-6); // |00>
        Assert.True(Math.Abs(probs[1] - 0.0) < 1e-6); // |01>
        Assert.True(Math.Abs(probs[2] - 0.0) < 1e-6); // |10>
        Assert.True(Math.Abs(probs[3] - 0.5) < 1e-6); // |11>

        // Test Von Neumann Entropy for pure state = 0
        double entropy = q.VonNeumannEntropy();
        Assert.True(Math.Abs(entropy) < 1e-6);

        // Apply Pauli X, Y, Z, S, T gates
        q.X(0);
        q.Z(1);
        q.S(0);
        q.T(1);
        Assert.Equal(4, q.Dimension);
    }

    [Fact]
    public void Test_Crypto_NTT_PolyMul_And_LLL()
    {
        // Test NTT on polynomials modulo q = 3329 (Kyber prime), n = 4, primitive root = 17
        long q = 3329;
        long root = 17;
        var a = NDArray<long>.FromArray(new[] { 1L, 2L, 3L, 4L }, 4);
        var b = NDArray<long>.FromArray(new[] { 5L, 6L, 7L, 8L }, 4);

        var aHat = LatticeMath.ForwardNTT(a, q, root);
        var aRecov = LatticeMath.InverseNTT(aHat, q, root);

        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(a[i], aRecov[i]);
        }

        // Test Polynomial multiplication via NTT
        var prod = LatticeMath.PolyMulNTT(a, b, q, root);
        Assert.Equal(4, prod.TotalLength);

        // Test LLL Lattice Basis Reduction
        var basis = NDArray<double>.FromArray(new double[]
        {
            1, 1, 1,
            -1, 0, 2,
            3, 5, 6
        }, 3, 3);

        var reduced = LatticeMath.LLLReduction(basis, delta: 0.75);
        Assert.Equal(3, reduced.Shape[0]);
        Assert.Equal(3, reduced.Shape[1]);
        Assert.Equal(9, reduced.TotalLength);
    }

    [Fact]
    public void Test_Biology_Dihedral_RMSD_TMScore()
    {
        // 4 points forming a 90 degree dihedral angle
        var p1 = NDArray<double>.FromArray(new[] { 1.0, 0.0, 0.0 }, 3);
        var p2 = NDArray<double>.FromArray(new[] { 0.0, 0.0, 0.0 }, 3);
        var p3 = NDArray<double>.FromArray(new[] { 0.0, 1.0, 0.0 }, 3);
        var p4 = NDArray<double>.FromArray(new[] { 0.0, 1.0, 1.0 }, 3);

        double angle = ProteinGeometry.ComputeDihedralAngle(p1, p2, p3, p4);
        Assert.True(Math.Abs(Math.Abs(angle) - Math.PI / 2.0) < 1e-4);

        // Test RMSD and TM-Score for identical protein coordinates
        var proteinA = NDArray<double>.FromArray(new double[]
        {
            0, 0, 0,
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        }, 4, 3);
        var proteinB = proteinA.Clone();

        double rmsd = ProteinGeometry.KabschRMSD(proteinA, proteinB);
        Assert.True(rmsd < 1e-6);

        double tmScore = ProteinGeometry.TMScore(proteinA, proteinB);
        Assert.True(Math.Abs(tmScore - 1.0) < 1e-4);
    }

    [Fact]
    public void Test_Graphs_Laplacian_And_ChebyshevConv()
    {
        // 3-node path graph: 0 - 1 - 2
        var adj = NDArray<double>.FromArray(new double[]
        {
            0, 1, 0,
            1, 0, 1,
            0, 1, 0
        }, 3, 3);

        var L = SpectralGraph.Laplacian(adj, normalized: true);
        Assert.Equal(3, L.Shape[0]);
        Assert.Equal(3, L.Shape[1]);
        Assert.True(Math.Abs(L[0, 0] - 1.0) < 1e-5);
        Assert.True(Math.Abs(L[1, 1] - 1.0) < 1e-5);

        // Feature matrix X: 3 nodes x 2 features
        var X = NDArray<double>.FromArray(new double[]
        {
            1.0, 0.5,
            0.0, 1.0,
            2.0, 0.0
        }, 3, 2);

        // Chebyshev coefficients [theta_0, theta_1, theta_2]
        var coeffs = NDArray<double>.FromArray(new[] { 1.0, 0.5, 0.1 }, 3);

        var gcnOut = SpectralGraph.ChebyshevGraphConv(L, X, coeffs);
        Assert.Equal(3, gcnOut.Shape[0]);
        Assert.Equal(2, gcnOut.Shape[1]);
    }

    [Fact]
    public void Test_Physics_LatticeBoltzmann2D()
    {
        int ny = 10;
        int nx = 20;
        double tau = 0.6;
        var f = LatticeBoltzmann2D.InitializeEquilibrium(ny, nx, rho0: 1.0, u0: 0.05, v0: 0.0);

        var obstacle = new BoolNDArray(ny, nx);
        obstacle[5, 10] = true;

        var (rho, u, v, fNext) = LatticeBoltzmann2D.Step(f, tau, obstacle);

        Assert.Equal(10, rho.Shape[0]);
        Assert.Equal(20, rho.Shape[1]);
        Assert.True(rho[0, 0] > 0.0);
        Assert.Equal(10, fNext.Shape[0]);
        Assert.Equal(20, fNext.Shape[1]);
        Assert.Equal(9, fNext.Shape[2]);
    }

    [Fact]
    public void Test_Neural_Sinkhorn_And_DDIM()
    {
        // 2D Sinkhorn Optimal Transport between two 3-point distributions
        var a = NDArray<double>.FromArray(new[] { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 }, 3);
        var b = NDArray<double>.FromArray(new[] { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 }, 3);
        var costMatrix = NDArray<double>.FromArray(new double[]
        {
            0.0, 1.0, 4.0,
            1.0, 0.0, 1.0,
            4.0, 1.0, 0.0
        }, 3, 3);

        var (distance, transportPlan) = OptimalTransportAndDiffusion.Sinkhorn(a, b, costMatrix, reg: 0.1);
        Assert.True(distance >= 0.0);
        Assert.Equal(3, transportPlan.Shape[0]);
        Assert.Equal(3, transportPlan.Shape[1]);

        // Generative DDIM step
        var xt = NDArray<double>.FromArray(new[] { 0.5, -0.2, 0.8 }, 3);
        var epsModel = NDArray<double>.FromArray(new[] { 0.1, -0.05, 0.2 }, 3);
        var xPrev = OptimalTransportAndDiffusion.DDIMStep(xt, epsModel, alpha_t: 0.8, alpha_prev: 0.9, eta: 0.0);

        Assert.Equal(3, xPrev.TotalLength);
    }

    [Fact]
    public void Test_Robotics_LQR_And_JacobianDLS()
    {
        // 2D discrete double integrator system
        var A = NDArray<double>.FromArray(new double[]
        {
            1.0, 1.0,
            0.0, 1.0
        }, 2, 2);
        var B = NDArray<double>.FromArray(new double[]
        {
            0.0,
            1.0
        }, 2, 1);
        var Q = NDArray<double>.FromArray(new double[]
        {
            1.0, 0.0,
            0.0, 1.0
        }, 2, 2);
        var R = NDArray<double>.FromArray(new double[]
        {
            1.0
        }, 1, 1);

        var (K, P) = OptimalControlAndRobotics.SolveDiscreteLQR(A, B, Q, R);
        Assert.Equal(1, K.Shape[0]);
        Assert.Equal(2, K.Shape[1]);
        Assert.Equal(2, P.Shape[0]);
        Assert.Equal(2, P.Shape[1]);

        // Inverse Kinematics DLS
        var J = NDArray<double>.FromArray(new double[]
        {
            -1.0, 0.5,
            1.0, 0.5
        }, 2, 2);
        var dx = NDArray<double>.FromArray(new[] { 0.1, -0.05 }, 2);
        var dTheta = OptimalControlAndRobotics.JacobianDLS(J, dx, lambda: 0.05);

        Assert.Equal(2, dTheta.TotalLength);

        // Unicycle Kinematics
        var (xNext, yNext, thNext) = OptimalControlAndRobotics.UnicycleKinematics(0, 0, 0, 1.0, 0.1, 0.1);
        Assert.True(xNext > 0);
    }

    [Fact]
    public void Test_Spatial_HyperbolicGeometry()
    {
        var u = NDArray<double>.FromArray(new[] { 0.1, 0.2 }, 2);
        var v = NDArray<double>.FromArray(new[] { -0.1, 0.3 }, 2);

        double dist = HyperbolicGeometry.PoincareDistance(u, v);
        Assert.True(dist > 0);

        // Distance from point to itself is 0
        double selfDist = HyperbolicGeometry.PoincareDistance(u, u);
        Assert.True(selfDist < 1e-6);

        // Möbius addition
        var sum = HyperbolicGeometry.MobiusAddition(u, v);
        Assert.Equal(2, sum.TotalLength);

        // Exp and Log Maps
        var tanV = NDArray<double>.FromArray(new[] { 0.2, -0.1 }, 2);
        var poincarePoint = HyperbolicGeometry.ExpMap0(tanV);
        var recovTan = HyperbolicGeometry.LogMap0(poincarePoint);

        Assert.True(Math.Abs(tanV[0] - recovTan[0]) < 1e-5);
        Assert.True(Math.Abs(tanV[1] - recovTan[1]) < 1e-5);
    }
}

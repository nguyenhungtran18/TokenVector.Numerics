// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.LinAlg;

/// <summary>
/// Provides advanced matrix analytic functions: Matrix Exponential (Expm via Padé [6/6]), Matrix Square Root (Sqrtm via Denman-Beavers), and Sylvester Equation Solver (AX + XB = C).
/// </summary>
public static class MatrixFunctions
{
    /// <summary>
    /// Computes the Matrix Exponential e^A using (6, 6) Padé Approximation with Scaling and Squaring (scipy.linalg.expm).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<double> Expm(NDArray<double> a)
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1])
            throw new ArgumentException("Expm requires a square matrix.");

        int n = a.Shape[0];
        double normA = MatrixOps.Norm(a, "inf");

        // Scaling factor: find s such that ||A / 2^s|| <= 0.5
        int s = 0;
        if (normA > 0.5)
        {
            s = (int)Math.Ceiling(Math.Log2(normA / 0.5));
            if (s < 0) s = 0;
        }

        double scale = Math.Pow(2.0, s);
        var aScaled = a * (1.0 / scale);

        // Padé (6, 6) coefficients: c_k = (12 - k)! * 6! / (12! * (6 - k)! * k!)
        // c = [1, 1/2, 5/44, 1/66, 1/792, 1/15840, 1/665280]
        double[] c =
        [
            1.0,
            0.5,
            5.0 / 44.0,
            1.0 / 66.0,
            1.0 / 792.0,
            1.0 / 15840.0,
            1.0 / 665280.0
        ];

        var iMat = NDArray<double>.Eye(n);
        var a2 = MatrixMultiplication.MatMul(aScaled, aScaled);
        var a4 = MatrixMultiplication.MatMul(a2, a2);
        var a6 = MatrixMultiplication.MatMul(a4, a2);

        // U = A * (c1*I + c3*A^2 + c5*A^4)
        // V = c0*I + c2*A^2 + c4*A^4 + c6*A^6
        var uInner = iMat * c[1] + a2 * c[3] + a4 * c[5];
        var u = MatrixMultiplication.MatMul(aScaled, uInner);

        var v = iMat * c[0] + a2 * c[2] + a4 * c[4] + a6 * c[6];

        // R = (V - U)^-1 * (V + U)
        var num = v + u;
        var den = v - u;
        var denInv = Decomposition.Inverse(den);
        var r = MatrixMultiplication.MatMul(denInv, num);

        // Squaring step: R = R^(2^s)
        for (int step = 0; step < s; step++)
        {
            r = MatrixMultiplication.MatMul(r, r);
        }

        return r;
    }

    /// <summary>
    /// Computes the principal Matrix Square Root S = sqrt(A) such that S * S = A using Denman-Beavers iteration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<double> Sqrtm(NDArray<double> a, double tol = 1e-10, int maxIter = 50)
    {
        if (a.Rank != 2 || a.Shape[0] != a.Shape[1])
            throw new ArgumentException("Sqrtm requires a square matrix.");

        int n = a.Shape[0];
        var y = a.Clone();
        var z = NDArray<double>.Eye(n);

        for (int iter = 0; iter < maxIter; iter++)
        {
            var zInv = Decomposition.Inverse(z);
            var yInv = Decomposition.Inverse(y);

            var nextY = (y + zInv) * 0.5;
            var nextZ = (z + yInv) * 0.5;

            var diff = nextY - y;
            double err = MatrixOps.Norm(diff, "fro");

            y = nextY;
            z = nextZ;

            if (err < tol) break;
        }

        return y;
    }

    /// <summary>
    /// Solves the Sylvester matrix equation AX + XB = C for X using Kronecker formulation: (I (x) A + B^T (x) I) vec(X) = vec(C).
    /// </summary>
    public static NDArray<double> SolveSylvester(NDArray<double> a, NDArray<double> b, NDArray<double> c)
    {
        int m = a.Shape[0];
        int n = b.Shape[0];

        var iM = NDArray<double>.Eye(m);
        var iN = NDArray<double>.Eye(n);

        // K = (I_n (x) A) + (B^T (x) I_m)
        var term1 = MatrixOps.Kron(iN, a);
        var term2 = MatrixOps.Kron(b.MatrixTranspose(), iM);
        var kMat = term1 + term2;

        // vec(C) is column-major flatten of C
        var vecC = new NDArray<double>(m * n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                vecC[j * m + i] = c[i, j];
            }
        }

        // Solve K * vec(X) = vec(C)
        var vecX = Decomposition.Solve(kMat, vecC);

        // Reconstruct X [m, n]
        var x = new NDArray<double>(m, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                x[i, j] = vecX[j * m + i];
            }
        }

        return x;
    }
}

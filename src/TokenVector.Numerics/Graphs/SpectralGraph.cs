// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Graphs;

/// <summary>
/// Provides spectral graph theory, graph Laplacians, and spectral Chebyshev graph convolutions for Graph Neural Networks (GNN).
/// </summary>
public static class SpectralGraph
{
    /// <summary>
    /// Computes the unnormalized or normalized Graph Laplacian L = D - A or L_norm = I - D^{-1/2} * A * D^{-1/2}.
    /// </summary>
    public static NDArray<double> Laplacian(NDArray<double> adjacencyMatrix, bool normalized = true)
    {
        int n = adjacencyMatrix.Shape[0];
        var degrees = new double[n];

        for (int i = 0; i < n; i++)
        {
            double d = 0.0;
            for (int j = 0; j < n; j++) d += adjacencyMatrix[i, j];
            degrees[i] = d;
        }

        if (!normalized)
        {
            var l = new NDArray<double>(n, n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    l[i, j] = (i == j) ? (degrees[i] - adjacencyMatrix[i, j]) : -adjacencyMatrix[i, j];
                }
            }
            return l;
        }
        else
        {
            var lNorm = NDArray<double>.Eye(n);
            for (int i = 0; i < n; i++)
            {
                double invSqrtDi = (degrees[i] > 1e-15) ? (1.0 / Math.Sqrt(degrees[i])) : 0.0;
                for (int j = 0; j < n; j++)
                {
                    double invSqrtDj = (degrees[j] > 1e-15) ? (1.0 / Math.Sqrt(degrees[j])) : 0.0;
                    if (adjacencyMatrix[i, j] != 0.0)
                    {
                        lNorm[i, j] -= invSqrtDi * adjacencyMatrix[i, j] * invSqrtDj;
                    }
                }
            }
            return lNorm;
        }
    }

    /// <summary>
    /// Computes fast spectral graph convolution using Chebyshev polynomial expansion: Y = \sum_{k=0}^{K-1} theta_k * T_k(\tilde{L}) * X.
    /// Rescaled Laplacian: \tilde{L} = (2 / lambda_max) * L - I.
    /// </summary>
    public static NDArray<double> ChebyshevGraphConv(
        NDArray<double> laplacian, NDArray<double> nodeFeatures, NDArray<double> chebyshevCoeffs, double lambdaMax = 2.0)
    {
        int n = laplacian.Shape[0];
        int kOrder = chebyshevCoeffs.TotalLength;

        // Rescaled Laplacian L_tilde = (2 / lambdaMax) * L - I
        var lTilde = laplacian * (2.0 / lambdaMax) - NDArray<double>.Eye(n);

        // T_0(L) * X = X
        var t0 = nodeFeatures.Clone();
        var y = t0 * chebyshevCoeffs[0];

        if (kOrder > 1)
        {
            // T_1(L) * X = L_tilde * X
            var t1 = MatrixMultiplication.MatMul(lTilde, nodeFeatures);
            y = y + t1 * chebyshevCoeffs[1];

            // T_k(L) * X = 2 * L_tilde * T_{k-1} * X - T_{k-2} * X
            for (int k = 2; k < kOrder; k++)
            {
                var lT1 = MatrixMultiplication.MatMul(lTilde, t1);
                var tk = lT1 * 2.0 - t0;

                y = y + tk * chebyshevCoeffs[k];

                t0 = t1;
                t1 = tk;
            }
        }

        return y;
    }
}

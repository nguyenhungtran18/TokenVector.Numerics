// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Neural;

/// <summary>
/// Provides entropic Optimal Transport (Sinkhorn Algorithm / Wasserstein Distance) and Generative Diffusion Sampling (DDIM / SDE score-based steps).
/// </summary>
public static class OptimalTransportAndDiffusion
{
    /// <summary>
    /// Computes the entropic regularized Optimal Transport distance (Sinkhorn Distance / Wasserstein-1/2) between probability distributions a and b given cost matrix M.
    /// min_{P} &lt;P, M&gt; + reg * KL(P || a (x) b)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static (double Distance, NDArray<double> TransportPlan) Sinkhorn(
        NDArray<double> a, NDArray<double> b, NDArray<double> costMatrix,
        double reg = 0.1, int maxIter = 100, double tol = 1e-6)
    {
        int n = a.TotalLength;
        int m = b.TotalLength;

        // Gibbs kernel K = exp(-M / reg)
        var kMat = new NDArray<double>(n, m);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                kMat[i, j] = Math.Exp(-costMatrix[i, j] / reg);
            }
        }

        // Scaling vectors u and v
        var u = NDArray<double>.Ones(n);
        var v = NDArray<double>.Ones(m);

        for (int iter = 0; iter < maxIter; iter++)
        {
            // u = a / (K * v)
            var kv = MatrixMultiplication.MatMul(kMat, v.Reshape(m, 1)).Reshape(n);
            var nextU = new NDArray<double>(n);
            for (int i = 0; i < n; i++)
            {
                nextU[i] = a[i] / (kv[i] + 1e-15);
            }

            // v = b / (K^T * u)
            var kTu = MatrixMultiplication.MatMul(kMat.MatrixTranspose(), nextU.Reshape(n, 1)).Reshape(m);
            var nextV = new NDArray<double>(m);
            for (int j = 0; j < m; j++)
            {
                nextV[j] = b[j] / (kTu[j] + 1e-15);
            }

            double err = MatrixOps.Norm(nextU - u, "1");
            u = nextU;
            v = nextV;

            if (err < tol) break;
        }

        // Optimal Transport Plan P = diag(u) * K * diag(v)
        var pPlan = new NDArray<double>(n, m);
        double totalCost = 0.0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                double pVal = u[i] * kMat[i, j] * v[j];
                pPlan[i, j] = pVal;
                totalCost += pVal * costMatrix[i, j];
            }
        }

        return (totalCost, pPlan);
    }

    /// <summary>
    /// Computes one deterministic sampling step for Generative Diffusion Models using Denoising Diffusion Implicit Models (DDIM).
    /// x_t: current noisy sample, predictedNoise: model output epsilon_theta(x_t, t),
    /// alpha_t: cumulative alpha product \bar{alpha}_t, alpha_prev: \bar{alpha}_{t-1}.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<double> DDIMStep(
        NDArray<double> x_t, NDArray<double> predictedNoise,
        double alpha_t, double alpha_prev, double eta = 0.0)
    {
        double sqrtAlphaT = Math.Sqrt(alpha_t);
        double sqrtOneMinusAlphaT = Math.Sqrt(1.0 - alpha_t);

        double sqrtAlphaPrev = Math.Sqrt(alpha_prev);
        double sqrtOneMinusAlphaPrev = Math.Sqrt(1.0 - alpha_prev);

        // 1. Estimate original x_0: pred_x0 = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
        var predX0 = (x_t - predictedNoise * sqrtOneMinusAlphaT) * (1.0 / sqrtAlphaT);

        // 2. Direction pointing to x_t: dir_xt = sqrt(1 - alpha_prev - sigma_t^2) * eps
        double sigmaT = eta * Math.Sqrt((1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev));
        double dirCoeff = Math.Sqrt(Math.Max(0.0, 1.0 - alpha_prev - sigmaT * sigmaT));
        var dirXt = predictedNoise * dirCoeff;

        // 3. x_{t-1} = sqrt(alpha_prev) * pred_x0 + dir_xt
        return predX0 * sqrtAlphaPrev + dirXt;
    }
}

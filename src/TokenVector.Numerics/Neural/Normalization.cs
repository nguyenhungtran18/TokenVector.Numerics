using System;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Neural;

/// <summary>
/// Neural network normalization layers: LayerNorm and BatchNorm with forward and backward gradients.
/// </summary>
public static class Normalization
{
    /// <summary>
    /// Computes Layer Normalization over the last normalized shape dimensions.
    /// y = gamma * ((x - mean) / sqrt(var + eps)) + beta.
    /// </summary>
    public static (NDArray<T> Output, NDArray<T> Mean, NDArray<T> InvStd) LayerNorm<T>(
        NDArray<T> x,
        NDArray<T>? gamma = null,
        NDArray<T>? beta = null,
        T? eps = null) where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        T epsilon = eps ?? T.CreateChecked(1e-5);
        int lastAxis = x.Rank - 1;

        // 1. Mean along last axis
        var mean = x.Mean(lastAxis, keepdims: true);

        // 2. Variance along last axis
        var diff = x - mean;
        var sqDiff = diff * diff;
        var variance = sqDiff.Mean(lastAxis, keepdims: true);

        // 3. InvStd = 1 / sqrt(var + eps)
        var invStd = new NDArray<T>(variance.Shape);
        var vSpan = variance.IsContiguous ? variance.AsReadOnlySpan() : variance.Contiguous().AsReadOnlySpan();
        var iSpan = invStd.AsSpan();
        for (int i = 0; i < vSpan.Length; i++)
        {
            iSpan[i] = T.One / T.Sqrt(vSpan[i] + epsilon);
        }

        // 4. Normalized x_hat = (x - mean) * invStd
        var xHat = diff * invStd;

        // 5. Affine transform: gamma * x_hat + beta
        var output = xHat;
        if (gamma != null) output = output * gamma;
        if (beta != null) output = output + beta;

        return (output, mean, invStd);
    }

    /// <summary>
    /// Computes Root Mean Square Normalization (RMSNorm): y = (x / sqrt(mean(x^2) + eps)) * gamma.
    /// Widely used in modern LLMs (LLaMA, Mistral, Gemma, DeepSeek).
    /// </summary>
    public static NDArray<T> RMSNorm<T>(NDArray<T> x, NDArray<T>? gamma = null, T? eps = null)
        where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        T epsilon = eps ?? T.CreateChecked(1e-5);
        int lastAxis = x.Rank - 1;

        var xSq = x * x;
        var meanSq = xSq.Mean(lastAxis, keepdims: true);

        var invRms = new NDArray<T>(meanSq.Shape);
        var mSpan = meanSq.IsContiguous ? meanSq.AsReadOnlySpan() : meanSq.Contiguous().AsReadOnlySpan();
        var iSpan = invRms.AsSpan();
        for (int i = 0; i < mSpan.Length; i++)
        {
            iSpan[i] = T.One / T.Sqrt(mSpan[i] + epsilon);
        }

        var normalized = x * invRms;
        if (gamma != null) normalized = normalized * gamma;
        return normalized;
    }
}

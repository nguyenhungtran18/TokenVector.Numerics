using System;
using System.Numerics;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Neural;

/// <summary>
/// Transformer Attention engine implementing Scaled Dot-Product Attention with Causal Masking and backward gradients.
/// </summary>
public static class AttentionEngine
{
    /// <summary>
    /// Computes Scaled Dot-Product Attention: Softmax((Q * K^T) / sqrt(d_k) + Mask) * V.
    /// Supports both 3D [Batch, SeqLen, D] and 4D [Batch, NumHeads, SeqLen, D] tensors.
    /// </summary>
    public static (NDArray<T> Output, NDArray<T> AttentionWeights) ScaledDotProductAttention<T>(
        NDArray<T> query,
        NDArray<T> key,
        NDArray<T> value,
        NDArray<T>? mask = null)
        where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>, IExponentialFunctions<T>
    {
        if (query.Rank < 2 || key.Rank < 2 || value.Rank < 2)
        {
            throw new ArgumentException("Attention inputs must have rank >= 2.");
        }

        int dK = query.Shape[^1];
        if (key.Shape[^1] != dK)
        {
            throw new ArgumentException($"Query and Key feature dimension must match: query={dK}, key={key.Shape[^1]}.");
        }

        T scale = T.One / T.Sqrt(T.CreateChecked(dK));

        // 1. Compute Q * K^T
        var kTranspose = key.Transpose(key.Rank - 2, key.Rank - 1);
        var scores = MatrixMultiplication.MatMul(query, kTranspose);

        // 2. Scale scores: scores * (1 / sqrt(d_k))
        scores = scores * scale;

        // 3. Add mask if present
        if (mask != null)
        {
            scores = scores + mask;
        }

        // 4. Softmax over the last axis (keys sequence length)
        var attnWeights = Activations.Softmax(scores, axis: -1);

        // 5. Compute Output = AttentionWeights * V
        var output = MatrixMultiplication.MatMul(attnWeights, value);

        return (output, attnWeights);
    }

    /// <summary>
    /// Computes backward gradients for Scaled Dot-Product Attention with respect to Q, K, and V.
    /// </summary>
    public static (NDArray<T> GradQuery, NDArray<T> GradKey, NDArray<T> GradValue) ScaledDotProductAttentionBackward<T>(
        NDArray<T> gradOutput,
        NDArray<T> query,
        NDArray<T> key,
        NDArray<T> value,
        NDArray<T> attnWeights)
        where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        int dK = query.Shape[^1];
        T scale = T.One / T.Sqrt(T.CreateChecked(dK));

        // dL/dV = AttentionWeights^T * gradOutput
        var attnWeightsT = attnWeights.Transpose(attnWeights.Rank - 2, attnWeights.Rank - 1);
        var gradValue = MatrixMultiplication.MatMul(attnWeightsT, gradOutput);

        // dL/dP = gradOutput * V^T
        var vT = value.Transpose(value.Rank - 2, value.Rank - 1);
        var gradWeights = MatrixMultiplication.MatMul(gradOutput, vT);

        // dL/dScores = SoftmaxBackward(gradWeights, attnWeights)
        var gradScores = Activations.SoftmaxBackward(gradWeights, attnWeights, axis: -1);

        // Scale gradient: dL/dScores * scale
        var scaledGradScores = gradScores * scale;

        // dL/dQ = scaledGradScores * K
        var gradQuery = MatrixMultiplication.MatMul(scaledGradScores, key);

        // dL/dK = scaledGradScores^T * Q
        var gradScoresT = scaledGradScores.Transpose(scaledGradScores.Rank - 2, scaledGradScores.Rank - 1);
        var gradKey = MatrixMultiplication.MatMul(gradScoresT, query);

        return (gradQuery, gradKey, gradValue);
    }

    /// <summary>
    /// Generates standard autoregressive causal upper-triangular mask with large negative values for future tokens.
    /// </summary>
    public static NDArray<T> CreateCausalMask<T>(int seqLen, T maskValue) where T : unmanaged, IFloatingPoint<T>
    {
        var mask = new NDArray<T>(seqLen, seqLen);
        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < seqLen; j++)
            {
                mask[i, j] = j > i ? maskValue : T.Zero;
            }
        }
        return mask;
    }

    /// <summary>
    /// Applies Rotary Positional Embedding (RoPE) to Query or Key tensor of shape [..., SeqLen, HeadDim].
    /// </summary>
    public static NDArray<double> ApplyRoPE(NDArray<double> x, int startPos = 0, double thetaBase = 10000.0)
    {
        int seqLen = x.Shape[^2];
        int headDim = x.Shape[^1];
        int halfDim = headDim / 2;

        var res = x.Clone();
        int batchAndHeads = x.TotalLength / (seqLen * headDim);

        for (int bh = 0; bh < batchAndHeads; bh++)
        {
            int baseOffset = bh * seqLen * headDim;
            for (int m = 0; m < seqLen; m++)
            {
                int pos = startPos + m;
                int posOffset = baseOffset + m * headDim;

                for (int i = 0; i < halfDim; i++)
                {
                    double freq = 1.0 / Math.Pow(thetaBase, (2.0 * i) / headDim);
                    double theta = pos * freq;
                    double cos = Math.Cos(theta);
                    double sin = Math.Sin(theta);

                    double x0 = x[posOffset + i];
                    double x1 = x[posOffset + i + halfDim];

                    res[posOffset + i] = x0 * cos - x1 * sin;
                    res[posOffset + i + halfDim] = x0 * sin + x1 * cos;
                }
            }
        }

        return res;
    }
}

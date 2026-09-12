using System;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Neural;

/// <summary>
/// Deep learning loss functions and analytical gradient estimators (MSE, CrossEntropy with Fused Softmax).
/// </summary>
public static class Losses
{
    /// <summary>
    /// Computes Mean Squared Error loss: (1/N) * sum((predictions - targets)^2).
    /// </summary>
    public static T MSELoss<T>(NDArray<T> predictions, NDArray<T> targets) where T : unmanaged, IFloatingPoint<T>
    {
        var diff = predictions - targets;
        var sqDiff = diff * diff;
        return sqDiff.Mean();
    }

    /// <summary>
    /// Computes gradient of MSE loss with respect to predictions: (2/N) * (predictions - targets).
    /// </summary>
    public static NDArray<T> MSELossBackward<T>(NDArray<T> predictions, NDArray<T> targets) where T : unmanaged, IFloatingPoint<T>
    {
        var diff = predictions - targets;
        T scale = T.CreateChecked(2.0) / T.CreateChecked(predictions.TotalLength);
        return diff * scale;
    }

    /// <summary>
    /// Computes Cross-Entropy loss with fused Softmax for numerical stability.
    /// Logits: [BatchSize, NumClasses], TargetClasses: [BatchSize].
    /// </summary>
    public static T CrossEntropyLoss<T>(NDArray<T> logits, NDArray<int> targetClasses)
        where T : unmanaged, IFloatingPoint<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        if (logits.Rank != 2 || targetClasses.Rank != 1)
        {
            throw new ArgumentException("CrossEntropyLoss requires 2D logits [BatchSize, Classes] and 1D targetClasses [BatchSize].");
        }

        int batchSize = logits.Shape[0];
        int numClasses = logits.Shape[1];
        if (targetClasses.Shape[0] != batchSize)
        {
            throw new ArgumentException("Batch size mismatch between logits and targetClasses.");
        }

        var probs = Activations.Softmax(logits, axis: -1);
        T totalLoss = T.Zero;
        T eps = T.CreateChecked(1e-12);

        for (int b = 0; b < batchSize; b++)
        {
            int targetIdx = targetClasses[b];
            if ((uint)targetIdx >= (uint)numClasses)
            {
                throw new IndexOutOfRangeException($"Target class {targetIdx} out of range [0, {numClasses}).");
            }

            T p = probs[b, targetIdx];
            if (p < eps) p = eps;
            totalLoss -= T.Log(p);
        }

        return totalLoss / T.CreateChecked(batchSize);
    }

    /// <summary>
    /// Computes gradient of Cross-Entropy loss with respect to logits: (1/BatchSize) * (Softmax(logits) - OneHot(targets)).
    /// </summary>
    public static NDArray<T> CrossEntropyLossBackward<T>(NDArray<T> logits, NDArray<int> targetClasses)
        where T : unmanaged, IFloatingPoint<T>, IExponentialFunctions<T>
    {
        int batchSize = logits.Shape[0];
        int numClasses = logits.Shape[1];

        var grad = Activations.Softmax(logits, axis: -1);
        T scale = T.One / T.CreateChecked(batchSize);

        for (int b = 0; b < batchSize; b++)
        {
            int targetIdx = targetClasses[b];
            grad[b, targetIdx] -= T.One;
        }

        return grad * scale;
    }
}

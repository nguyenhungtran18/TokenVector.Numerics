using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Engine;

/// <summary>
/// High-performance multi-dimensional broadcasting engine implementing NumPy right-aligned rules and Stride-Zero Tricking.
/// </summary>
public static class BroadcastEngine
{
    /// <summary>
    /// Computes the broadcasted output shape from two input shapes.
    /// </summary>
    public static int[] BroadcastShapes(ReadOnlySpan<int> shapeA, ReadOnlySpan<int> shapeB)
    {
        int rankA = shapeA.Length;
        int rankB = shapeB.Length;
        int maxRank = Math.Max(rankA, rankB);
        int[] outShape = new int[maxRank];

        for (int i = 0; i < maxRank; i++)
        {
            int dimA = (i < rankA) ? shapeA[rankA - 1 - i] : 1;
            int dimB = (i < rankB) ? shapeB[rankB - 1 - i] : 1;

            if (dimA == dimB || dimA == 1 || dimB == 1)
            {
                outShape[maxRank - 1 - i] = Math.Max(dimA, dimB);
            }
            else
            {
                throw new InvalidOperationException(
                    $"Cannot broadcast shapes with incompatible dimensions: {string.Join("x", shapeA.ToArray())} and {string.Join("x", shapeB.ToArray())}");
            }
        }

        return outShape;
    }

    /// <summary>
    /// Computes broadcast strides using Stride-0 trick for singleton dimensions.
    /// </summary>
    public static int[] ComputeBroadcastStrides(ReadOnlySpan<int> sourceShape, ReadOnlySpan<int> sourceStrides, ReadOnlySpan<int> targetShape)
    {
        int srcRank = sourceShape.Length;
        int targetRank = targetShape.Length;
        int[] broadcastStrides = new int[targetRank];

        int rankDiff = targetRank - srcRank;
        for (int i = 0; i < targetRank; i++)
        {
            if (i < rankDiff)
            {
                // Leading dimensions added by broadcasting have stride 0
                broadcastStrides[i] = 0;
            }
            else
            {
                int srcIdx = i - rankDiff;
                int srcDim = sourceShape[srcIdx];
                int targetDim = targetShape[i];

                if (srcDim == targetDim)
                {
                    broadcastStrides[i] = sourceStrides[srcIdx];
                }
                else if (srcDim == 1)
                {
                    // Stride-0 trick: virtual dimension expansion with zero RAM copy
                    broadcastStrides[i] = 0;
                }
                else
                {
                    throw new InvalidOperationException($"Incompatible dimension at axis {i}: source {srcDim} cannot broadcast to target {targetDim}.");
                }
            }
        }

        return broadcastStrides;
    }

    /// <summary>
    /// Broadcasts an NDArray to a target shape using Stride-0 trick zero-copy view.
    /// </summary>
    public static NDArray<T> BroadcastTo<T>(NDArray<T> source, params int[] targetShape) where T : unmanaged, INumber<T>
    {
        int[] strides = ComputeBroadcastStrides(source.Shape, source.Strides, targetShape);
        return new NDArray<T>(source.Buffer, targetShape, strides, source.Offset);
    }

    /// <summary>
    /// Broadcasts two NDArrays to a common output shape.
    /// </summary>
    public static (NDArray<T> Left, NDArray<T> Right) Broadcast<T>(NDArray<T> a, NDArray<T> b) where T : unmanaged, INumber<T>
    {
        int[] outShape = BroadcastShapes(a.Shape, b.Shape);
        return (BroadcastTo(a, outShape), BroadcastTo(b, outShape));
    }
}

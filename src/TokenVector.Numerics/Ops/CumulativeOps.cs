// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;
using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Ops;

/// <summary>
/// Provides cumulative operations and discrete differences along arbitrary axes (np.cumsum, np.cumprod, np.diff).
/// </summary>
public static class CumulativeOps
{
    /// <summary>
    /// Computes the cumulative sum of elements along a given axis or flattened array.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> CumSum<T>(NDArray<T> tensor, int? axis = null) where T : unmanaged, INumber<T>
    {
        var contig = tensor.Contiguous();
        if (axis == null)
        {
            var result = new NDArray<T>(contig.TotalLength);
            T running = T.Zero;
            for (int i = 0; i < contig.TotalLength; i++)
            {
                running += contig[i];
                result[i] = running;
            }
            return result;
        }

        int normAxis = NormalizeAxis(axis.Value, contig.Rank);
        var res = new NDArray<T>(contig.Shape);
        int axisDim = contig.Shape[normAxis];
        int outerSize = 1;
        for (int i = 0; i < normAxis; i++) outerSize *= contig.Shape[i];
        int innerSize = 1;
        for (int i = normAxis + 1; i < contig.Rank; i++) innerSize *= contig.Shape[i];

        Parallel.For(0, outerSize, o =>
        {
            for (int inn = 0; inn < innerSize; inn++)
            {
                T acc = T.Zero;
                for (int a = 0; a < axisDim; a++)
                {
                    int flatIdx = (o * axisDim + a) * innerSize + inn;
                    acc += contig[flatIdx];
                    res[flatIdx] = acc;
                }
            }
        });

        return res;
    }

    /// <summary>
    /// Computes the cumulative product of elements along a given axis or flattened array.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> CumProd<T>(NDArray<T> tensor, int? axis = null) where T : unmanaged, INumber<T>
    {
        var contig = tensor.Contiguous();
        if (axis == null)
        {
            var result = new NDArray<T>(contig.TotalLength);
            T running = T.One;
            for (int i = 0; i < contig.TotalLength; i++)
            {
                running *= contig[i];
                result[i] = running;
            }
            return result;
        }

        int normAxis = NormalizeAxis(axis.Value, contig.Rank);
        var res = new NDArray<T>(contig.Shape);
        int axisDim = contig.Shape[normAxis];
        int outerSize = 1;
        for (int i = 0; i < normAxis; i++) outerSize *= contig.Shape[i];
        int innerSize = 1;
        for (int i = normAxis + 1; i < contig.Rank; i++) innerSize *= contig.Shape[i];

        Parallel.For(0, outerSize, o =>
        {
            for (int inn = 0; inn < innerSize; inn++)
            {
                T acc = T.One;
                for (int a = 0; a < axisDim; a++)
                {
                    int flatIdx = (o * axisDim + a) * innerSize + inn;
                    acc *= contig[flatIdx];
                    res[flatIdx] = acc;
                }
            }
        });

        return res;
    }

    /// <summary>
    /// Computes the n-th discrete difference along the given axis (np.diff).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Diff<T>(NDArray<T> tensor, int n = 1, int axis = -1) where T : unmanaged, INumber<T>
    {
        if (n < 1) throw new ArgumentOutOfRangeException(nameof(n), "Order of difference must be >= 1.");
        var current = tensor.Contiguous();
        int normAxis = NormalizeAxis(axis, current.Rank);
        
        for (int iter = 0; iter < n; iter++)
        {
            int axisDim = current.Shape[normAxis];
            if (axisDim <= 1)
            {
                var emptyShape = (int[])current.Shape.Clone();
                emptyShape[normAxis] = 0;
                return new NDArray<T>(emptyShape);
            }

            var newShape = (int[])current.Shape.Clone();
            newShape[normAxis] = axisDim - 1;
            var next = new NDArray<T>(newShape);

            int outerSize = 1;
            for (int i = 0; i < normAxis; i++) outerSize *= current.Shape[i];
            int innerSize = 1;
            for (int i = normAxis + 1; i < current.Rank; i++) innerSize *= current.Shape[i];

            int nextAxisDim = axisDim - 1;
            var curCopy = current;
            var nextCopy = next;

            Parallel.For(0, outerSize, o =>
            {
                for (int inn = 0; inn < innerSize; inn++)
                {
                    for (int a = 0; a < nextAxisDim; a++)
                    {
                        int curFlatIdx1 = (o * axisDim + a + 1) * innerSize + inn;
                        int curFlatIdx0 = (o * axisDim + a) * innerSize + inn;
                        int nextFlatIdx = (o * nextAxisDim + a) * innerSize + inn;

                        T val = curCopy[curFlatIdx1] - curCopy[curFlatIdx0];
                        nextCopy[nextFlatIdx] = val;
                    }
                }
            });

            current = next;
        }

        return current;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int NormalizeAxis(int axis, int rank)
    {
        int normalized = axis < 0 ? axis + rank : axis;
        if (normalized < 0 || normalized >= rank)
        {
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is out of bounds for rank {rank}.");
        }
        return normalized;
    }
}

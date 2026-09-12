// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Numerics;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Engine;

namespace TokenVector.Numerics.Core;

/// <summary>
/// Shape transformations and dimension expansion/squeezing operations.
/// </summary>
public static class ShapeExtensions
{
    public static NDArray<T> ExpandDims<T>(this NDArray<T> arr, int axis) where T : unmanaged, INumber<T>
    {
        int rank = arr.Rank;
        if (axis < 0) axis += rank + 1;
        if (axis < 0 || axis > rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        int[] newShape = new int[rank + 1];
        int src = 0;
        for (int i = 0; i <= rank; i++)
        {
            if (i == axis) newShape[i] = 1;
            else newShape[i] = arr.Shape[src++];
        }

        return arr.Reshape(newShape);
    }

    public static NDArray<T> Squeeze<T>(this NDArray<T> arr, int? axis = null) where T : unmanaged, INumber<T>
    {
        if (axis.HasValue)
        {
            int ax = axis.Value;
            if (ax < 0) ax += arr.Rank;
            if (ax < 0 || ax >= arr.Rank)
                throw new ArgumentOutOfRangeException(nameof(axis));
            if (arr.Shape[ax] != 1)
                throw new InvalidOperationException($"Cannot squeeze dimension {ax} of size {arr.Shape[ax]}.");

            int[] newShape = new int[arr.Rank - 1];
            int dst = 0;
            for (int i = 0; i < arr.Rank; i++)
            {
                if (i != ax) newShape[dst++] = arr.Shape[i];
            }
            return arr.Reshape(newShape.Length == 0 ? new[] { 1 } : newShape);
        }
        else
        {
            var nonOnes = new System.Collections.Generic.List<int>();
            for (int i = 0; i < arr.Rank; i++)
            {
                if (arr.Shape[i] != 1) nonOnes.Add(arr.Shape[i]);
            }
            int[] newShape = nonOnes.Count == 0 ? new[] { 1 } : nonOnes.ToArray();
            return arr.Reshape(newShape);
        }
    }

    public static NDArray<T> BroadcastTo<T>(this NDArray<T> arr, params int[] targetShape) where T : unmanaged, INumber<T>
    {
        var dummy = new NDArray<T>(targetShape);
        var (bArr, _) = BroadcastEngine.Broadcast(arr, dummy);
        return bArr;
    }

    public static NDArray<T> AtLeast1D<T>(this NDArray<T> arr) where T : unmanaged, INumber<T>
    {
        if (arr.Rank >= 1) return arr;
        return arr.Reshape(1);
    }

    public static NDArray<T> AtLeast2D<T>(this NDArray<T> arr) where T : unmanaged, INumber<T>
    {
        if (arr.Rank >= 2) return arr;
        if (arr.Rank == 1) return arr.Reshape(1, arr.Shape[0]);
        return arr.Reshape(1, 1);
    }

    public static NDArray<T> AtLeast3D<T>(this NDArray<T> arr) where T : unmanaged, INumber<T>
    {
        if (arr.Rank >= 3) return arr;
        if (arr.Rank == 2) return arr.Reshape(1, arr.Shape[0], arr.Shape[1]);
        if (arr.Rank == 1) return arr.Reshape(1, arr.Shape[0], 1);
        return arr.Reshape(1, 1, 1);
    }

    public static int[] UnravelIndex(int flatIndex, ReadOnlySpan<int> shape)
    {
        int rank = shape.Length;
        int[] coords = new int[rank];
        int rem = flatIndex;

        for (int d = rank - 1; d >= 0; d--)
        {
            coords[d] = rem % shape[d];
            rem /= shape[d];
        }
        return coords;
    }

    public static int RavelMultiIndex(ReadOnlySpan<int> coords, ReadOnlySpan<int> shape)
    {
        int flat = 0;
        int stride = 1;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            flat += coords[d] * stride;
            stride *= shape[d];
        }
        return flat;
    }
}

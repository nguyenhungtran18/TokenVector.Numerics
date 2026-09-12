using System;
using System.Collections.Generic;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Ops;

/// <summary>
/// Tensor sorting, searching, partition, and element clipping operations (Sort, ArgSort, Unique, NonZero, Clip).
/// </summary>
public static class SortingOps
{
    /// <summary>
    /// Returns a sorted copy of the tensor along the specified axis (equivalent to np.sort).
    /// </summary>
    public static NDArray<T> Sort<T>(this NDArray<T> tensor, int axis = -1, bool descending = false) where T : unmanaged, INumber<T>
    {
        if (axis < 0) axis += tensor.Rank;
        int axisSize = tensor.Shape[axis];

        var result = new NDArray<T>(tensor.Shape);
        int totalSlices = tensor.TotalLength / axisSize;

        int[] sliceShape = new int[tensor.Rank - 1];
        int sIdx = 0;
        for (int d = 0; d < tensor.Rank; d++)
        {
            if (d != axis) sliceShape[sIdx++] = tensor.Shape[d];
        }

        Span<int> sliceCoords = stackalloc int[sliceShape.Length];

        for (int s = 0; s < totalSlices; s++)
        {
            ShapeHelper.GetMultiIndex(s, sliceShape, sliceCoords);

            T[] line = new T[axisSize];
            for (int a = 0; a < axisSize; a++)
            {
                int[] fullCoords = ReconstructCoords(sliceCoords, axis, a);
                line[a] = tensor[fullCoords];
            }

            Array.Sort(line);
            if (descending) Array.Reverse(line);

            for (int a = 0; a < axisSize; a++)
            {
                int[] fullCoords = ReconstructCoords(sliceCoords, axis, a);
                result[fullCoords] = line[a];
            }
        }

        return result;
    }

    /// <summary>
    /// Returns the indices that would sort the tensor along the specified axis (equivalent to np.argsort).
    /// </summary>
    public static NDArray<int> ArgSort<T>(this NDArray<T> tensor, int axis = -1, bool descending = false) where T : unmanaged, INumber<T>
    {
        if (axis < 0) axis += tensor.Rank;
        int axisSize = tensor.Shape[axis];

        var result = new NDArray<int>(tensor.Shape);
        int totalSlices = tensor.TotalLength / axisSize;

        int[] sliceShape = new int[tensor.Rank - 1];
        int sIdx = 0;
        for (int d = 0; d < tensor.Rank; d++)
        {
            if (d != axis) sliceShape[sIdx++] = tensor.Shape[d];
        }

        Span<int> argSliceCoords = stackalloc int[sliceShape.Length];

        for (int s = 0; s < totalSlices; s++)
        {
            ShapeHelper.GetMultiIndex(s, sliceShape, argSliceCoords);

            int[] indices = new int[axisSize];
            T[] values = new T[axisSize];

            for (int a = 0; a < axisSize; a++)
            {
                indices[a] = a;
                int[] fullCoords = ReconstructCoords(argSliceCoords, axis, a);
                values[a] = tensor[fullCoords];
            }

            Array.Sort(indices, (i1, i2) => descending ? values[i2].CompareTo(values[i1]) : values[i1].CompareTo(values[i2]));

            for (int a = 0; a < axisSize; a++)
            {
                int[] fullCoords = ReconstructCoords(argSliceCoords, axis, a);
                result[fullCoords] = indices[a];
            }
        }

        return result;
    }

    /// <summary>
    /// Finds the unique sorted elements of a tensor (equivalent to np.unique).
    /// </summary>
    public static NDArray<T> Unique<T>(this NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        HashSet<T> uniqueSet = new();
        foreach (var val in tensor)
        {
            uniqueSet.Add(val);
        }

        T[] sorted = new T[uniqueSet.Count];
        uniqueSet.CopyTo(sorted);
        Array.Sort(sorted);

        return NDArray<T>.FromArray(sorted, sorted.Length);
    }

    /// <summary>
    /// Returns the indices of the elements that are non-zero (equivalent to np.nonzero).
    /// Output shape: [Rank, NumNonZeros].
    /// </summary>
    public static NDArray<int> NonZero<T>(this NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        List<int[]> matchedCoords = new();
        int total = tensor.TotalLength;
        Span<int> coords = stackalloc int[tensor.Rank];

        for (int i = 0; i < total; i++)
        {
            ShapeHelper.GetMultiIndex(i, tensor.Shape, coords);
            var cArr = coords.ToArray();

            if (tensor[cArr] != T.Zero)
            {
                matchedCoords.Add(cArr);
            }
        }

        int count = matchedCoords.Count;
        var result = new NDArray<int>(tensor.Rank, count);

        for (int i = 0; i < count; i++)
        {
            for (int r = 0; r < tensor.Rank; r++)
            {
                result[r, i] = matchedCoords[i][r];
            }
        }

        return result;
    }

    /// <summary>
    /// Clamps tensor values within a range [min, max] (equivalent to np.clip).
    /// </summary>
    public static NDArray<T> Clip<T>(this NDArray<T> tensor, T min, T max) where T : unmanaged, INumber<T>
    {
        if (min > max) throw new ArgumentException("min cannot be greater than max.");
        var result = new NDArray<T>(tensor.Shape);
        var src = tensor.IsContiguous ? tensor.AsReadOnlySpan() : tensor.Contiguous().AsReadOnlySpan();
        var dst = result.AsSpan();

        for (int i = 0; i < src.Length; i++)
        {
            T val = src[i];
            if (val < min) dst[i] = min;
            else if (val > max) dst[i] = max;
            else dst[i] = val;
        }

        return result;
    }

    private static int[] ReconstructCoords(ReadOnlySpan<int> reducedCoords, int axis, int axisVal)
    {
        int rank = reducedCoords.Length + 1;
        int[] coords = new int[rank];
        int rIdx = 0;
        for (int i = 0; i < rank; i++)
        {
            if (i == axis) coords[i] = axisVal;
            else coords[i] = reducedCoords[rIdx++];
        }
        return coords;
    }
}

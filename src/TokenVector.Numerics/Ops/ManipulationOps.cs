using System;
using System.Collections.Generic;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Ops;

/// <summary>
/// Tensor structural manipulation operations: Concatenate, Stack, Split, Tile, Repeat, Pad, Roll, and Flip.
/// </summary>
public static class ManipulationOps
{
    /// <summary>
    /// Joins a sequence of tensors along an existing axis (equivalent to np.concatenate).
    /// </summary>
    public static NDArray<T> Concatenate<T>(IReadOnlyList<NDArray<T>> tensors, int axis = 0) where T : unmanaged, INumber<T>
    {
        if (tensors == null || tensors.Count == 0) throw new ArgumentException("Tensors list cannot be empty.");
        var first = tensors[0];
        int rank = first.Rank;

        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));

        int totalAxisDim = 0;
        foreach (var t in tensors)
        {
            if (t.Rank != rank) throw new ArgumentException("All tensors must have the same rank to concatenate.");
            for (int d = 0; d < rank; d++)
            {
                if (d != axis && t.Shape[d] != first.Shape[d])
                {
                    throw new ArgumentException($"Shape mismatch at dimension {d}: {t.Shape[d]} != {first.Shape[d]}.");
                }
            }
            totalAxisDim += t.Shape[axis];
        }

        int[] outShape = (int[])first.Shape.Clone();
        outShape[axis] = totalAxisDim;

        var result = new NDArray<T>(outShape);
        int currentOffsetOnAxis = 0;
        Span<int> srcCoords = stackalloc int[rank];

        foreach (var t in tensors)
        {
            int tAxisDim = t.Shape[axis];
            int tTotal = t.TotalLength;

            for (int i = 0; i < tTotal; i++)
            {
                ShapeHelper.GetMultiIndex(i, t.Shape, srcCoords);
                var dstCoords = (int[])srcCoords.ToArray().Clone();
                dstCoords[axis] += currentOffsetOnAxis;

                result[dstCoords] = t[srcCoords.ToArray()];
            }

            currentOffsetOnAxis += tAxisDim;
        }

        return result;
    }

    /// <summary>
    /// Joins a sequence of tensors along a new axis (equivalent to np.stack).
    /// </summary>
    public static NDArray<T> Stack<T>(IReadOnlyList<NDArray<T>> tensors, int axis = 0) where T : unmanaged, INumber<T>
    {
        if (tensors == null || tensors.Count == 0) throw new ArgumentException("Tensors list cannot be empty.");
        var first = tensors[0];
        int rank = first.Rank;

        if (axis < 0) axis += rank + 1;
        if (axis < 0 || axis > rank) throw new ArgumentOutOfRangeException(nameof(axis));

        int[] outShape = new int[rank + 1];
        int sIdx = 0;
        for (int i = 0; i <= rank; i++)
        {
            if (i == axis) outShape[i] = tensors.Count;
            else outShape[i] = first.Shape[sIdx++];
        }

        var result = new NDArray<T>(outShape);
        Span<int> stackSrcCoords = stackalloc int[rank];

        for (int tIdx = 0; tIdx < tensors.Count; tIdx++)
        {
            var t = tensors[tIdx];
            if (!t.Shape.AsSpan().SequenceEqual(first.Shape.AsSpan()))
            {
                throw new ArgumentException("All tensors must have identical shape for stacking.");
            }

            int tTotal = t.TotalLength;

            for (int i = 0; i < tTotal; i++)
            {
                ShapeHelper.GetMultiIndex(i, t.Shape, stackSrcCoords);
                int[] dstCoords = new int[rank + 1];
                int curSIdx = 0;
                for (int d = 0; d <= rank; d++)
                {
                    if (d == axis) dstCoords[d] = tIdx;
                    else dstCoords[d] = stackSrcCoords[curSIdx++];
                }

                result[dstCoords] = t[stackSrcCoords.ToArray()];
            }
        }

        return result;
    }

    /// <summary>
    /// Stacks tensors vertically (row wise) -> equivalent to np.vstack.
    /// </summary>
    public static NDArray<T> VStack<T>(params NDArray<T>[] tensors) where T : unmanaged, INumber<T> => Concatenate(tensors, axis: 0);

    /// <summary>
    /// Stacks tensors horizontally (column wise) -> equivalent to np.hstack.
    /// </summary>
    public static NDArray<T> HStack<T>(params NDArray<T>[] tensors) where T : unmanaged, INumber<T> => Concatenate(tensors, axis: 1);

    /// <summary>
    /// Splits tensor into multiple sub-tensors along an axis (equivalent to np.split).
    /// </summary>
    public static NDArray<T>[] Split<T>(NDArray<T> tensor, int numSections, int axis = 0) where T : unmanaged, INumber<T>
    {
        if (axis < 0) axis += tensor.Rank;
        int dimSize = tensor.Shape[axis];
        if (dimSize % numSections != 0)
        {
            throw new ArgumentException($"Dimension size {dimSize} along axis {axis} is not evenly divisible by {numSections}.");
        }

        int sectionSize = dimSize / numSections;
        var results = new NDArray<T>[numSections];

        for (int i = 0; i < numSections; i++)
        {
            var ranges = new (int Start, int Stop, int Step)[tensor.Rank];
            for (int d = 0; d < tensor.Rank; d++)
            {
                if (d == axis) ranges[d] = (i * sectionSize, (i + 1) * sectionSize, 1);
                else ranges[d] = (0, tensor.Shape[d], 1);
            }
            results[i] = tensor.Slice(ranges).Contiguous();
        }

        return results;
    }

    /// <summary>
    /// Constructs a tensor by repeating tensor dimensions the number of times given by reps (equivalent to np.tile).
    /// </summary>
    public static NDArray<T> Tile<T>(NDArray<T> tensor, params int[] reps) where T : unmanaged, INumber<T>
    {
        int rank = Math.Max(tensor.Rank, reps.Length);
        int[] inShape = new int[rank];
        int[] outReps = new int[rank];

        int tensorDiff = rank - tensor.Rank;
        int repsDiff = rank - reps.Length;

        for (int i = 0; i < rank; i++)
        {
            inShape[i] = (i >= tensorDiff) ? tensor.Shape[i - tensorDiff] : 1;
            outReps[i] = (i >= repsDiff) ? reps[i - repsDiff] : 1;
        }

        int[] outShape = new int[rank];
        for (int i = 0; i < rank; i++) outShape[i] = inShape[i] * outReps[i];

        var result = new NDArray<T>(outShape);
        int total = result.TotalLength;

        Span<int> dstCoords = stackalloc int[rank];
        for (int i = 0; i < total; i++)
        {
            ShapeHelper.GetMultiIndex(i, outShape, dstCoords);
            int[] srcCoords = new int[rank];
            for (int d = 0; d < rank; d++)
            {
                srcCoords[d] = dstCoords[d] % inShape[d];
            }

            int[] origCoords = (tensorDiff > 0) ? srcCoords[tensorDiff..] : srcCoords;
            result[dstCoords.ToArray()] = tensor[origCoords];
        }

        return result;
    }

    /// <summary>
    /// Repeats elements of a tensor (equivalent to np.repeat).
    /// </summary>
    public static NDArray<T> Repeat<T>(NDArray<T> tensor, int repeats, int? axis = null) where T : unmanaged, INumber<T>
    {
        if (repeats < 0) throw new ArgumentOutOfRangeException(nameof(repeats));

        if (axis == null)
        {
            // Flatten and repeat
            var flat = tensor.Contiguous();
            var result = new NDArray<T>(flat.TotalLength * repeats);
            int idx = 0;
            foreach (var val in flat)
            {
                for (int r = 0; r < repeats; r++)
                {
                    result[idx++] = val;
                }
            }
            return result;
        }

        int ax = axis.Value;
        if (ax < 0) ax += tensor.Rank;
        int[] outShape = (int[])tensor.Shape.Clone();
        outShape[ax] *= repeats;

        var res = new NDArray<T>(outShape);
        int total = res.TotalLength;

        Span<int> dstCoords = stackalloc int[tensor.Rank];
        for (int i = 0; i < total; i++)
        {
            ShapeHelper.GetMultiIndex(i, outShape, dstCoords);
            var srcCoords = (int[])dstCoords.ToArray().Clone();
            srcCoords[ax] /= repeats;

            res[dstCoords.ToArray()] = tensor[srcCoords];
        }

        return res;
    }

    /// <summary>
    /// Pads a tensor with constant values (equivalent to np.pad).
    /// </summary>
    public static NDArray<T> Pad<T>(NDArray<T> tensor, (int Before, int After)[] padWidths, T constantValue = default) where T : unmanaged, INumber<T>
    {
        if (padWidths.Length != tensor.Rank) throw new ArgumentException("padWidths length must match tensor rank.");

        int[] outShape = new int[tensor.Rank];
        for (int d = 0; d < tensor.Rank; d++)
        {
            outShape[d] = tensor.Shape[d] + padWidths[d].Before + padWidths[d].After;
        }

        var result = new NDArray<T>(outShape);
        result.AsSpan().Fill(constantValue);

        int inTotal = tensor.TotalLength;
        Span<int> srcCoords = stackalloc int[tensor.Rank];

        for (int i = 0; i < inTotal; i++)
        {
            ShapeHelper.GetMultiIndex(i, tensor.Shape, srcCoords);
            int[] dstCoords = new int[tensor.Rank];
            for (int d = 0; d < tensor.Rank; d++)
            {
                dstCoords[d] = srcCoords[d] + padWidths[d].Before;
            }

            result[dstCoords] = tensor[srcCoords.ToArray()];
        }

        return result;
    }

    /// <summary>
    /// Rolls tensor elements along a given axis (equivalent to np.roll).
    /// </summary>
    public static NDArray<T> Roll<T>(NDArray<T> tensor, int shift, int? axis = null) where T : unmanaged, INumber<T>
    {
        if (axis == null)
        {
            // Flat roll
            var flat = tensor.Contiguous();
            var result = new NDArray<T>(tensor.Shape);
            int n = flat.TotalLength;
            int actualShift = ((shift % n) + n) % n;

            for (int i = 0; i < n; i++)
            {
                int srcIdx = (i - actualShift + n) % n;
                result[i] = flat[srcIdx];
            }
            return result;
        }

        int ax = axis.Value;
        if (ax < 0) ax += tensor.Rank;
        int dimSize = tensor.Shape[ax];
        int effShift = ((shift % dimSize) + dimSize) % dimSize;

        var res = new NDArray<T>(tensor.Shape);
        int total = res.TotalLength;

        Span<int> dstCoords = stackalloc int[tensor.Rank];
        for (int i = 0; i < total; i++)
        {
            ShapeHelper.GetMultiIndex(i, tensor.Shape, dstCoords);
            var srcCoords = (int[])dstCoords.ToArray().Clone();
            srcCoords[ax] = (dstCoords[ax] - effShift + dimSize) % dimSize;

            res[dstCoords.ToArray()] = tensor[srcCoords];
        }

        return res;
    }

    /// <summary>
    /// Reverses the order of elements along the given axis (equivalent to np.flip).
    /// </summary>
    public static NDArray<T> Flip<T>(NDArray<T> tensor, int axis = 0) where T : unmanaged, INumber<T>
    {
        if (axis < 0) axis += tensor.Rank;
        int dimSize = tensor.Shape[axis];

        var result = new NDArray<T>(tensor.Shape);
        int total = result.TotalLength;

        Span<int> dstCoords = stackalloc int[tensor.Rank];
        for (int i = 0; i < total; i++)
        {
            ShapeHelper.GetMultiIndex(i, tensor.Shape, dstCoords);
            var srcCoords = (int[])dstCoords.ToArray().Clone();
            srcCoords[axis] = dimSize - 1 - dstCoords[axis];

            result[dstCoords.ToArray()] = tensor[srcCoords];
        }

        return result;
    }
}

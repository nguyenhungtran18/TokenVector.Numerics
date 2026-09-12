using System;
using System.Collections.Generic;
using System.Numerics;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Engine;

namespace TokenVector.Numerics.Ops;

/// <summary>
/// Advanced Fancy Indexing, Boolean Masking, Where, Take (Gather) and Put (Scatter) operations.
/// </summary>
public static class IndexingOps
{
    /// <summary>
    /// Filters tensor elements where the boolean mask is true, returning a flat 1D NDArray (equivalent to arr[mask]).
    /// </summary>
    public static NDArray<T> Filter<T>(this NDArray<T> tensor, BoolNDArray mask) where T : unmanaged, INumber<T>
    {
        if (tensor.TotalLength != mask.TotalLength)
        {
            throw new ArgumentException("Tensor and mask must have identical number of elements.");
        }

        List<T> matched = new();
        int i = 0;
        foreach (var val in tensor)
        {
            if (mask.Buffer[mask.Offset + i])
            {
                matched.Add(val);
            }
            i++;
        }

        return NDArray<T>.FromArray(matched.ToArray(), matched.Count);
    }

    /// <summary>
    /// In-place assignment to tensor elements where the boolean mask is true: tensor[mask] = value.
    /// </summary>
    public static void SetWhere<T>(this NDArray<T> tensor, BoolNDArray mask, T value) where T : unmanaged, INumber<T>
    {
        if (tensor.TotalLength != mask.TotalLength)
        {
            throw new ArgumentException("Tensor and mask must have identical number of elements.");
        }

        if (tensor.IsContiguous)
        {
            var span = tensor.AsSpan();
            for (int i = 0; i < tensor.TotalLength; i++)
            {
                if (mask.Buffer[mask.Offset + i])
                {
                    span[i] = value;
                }
            }
        }
        else
        {
            Span<int> coords = stackalloc int[tensor.Rank];
            for (int i = 0; i < tensor.TotalLength; i++)
            {
                if (mask.Buffer[mask.Offset + i])
                {
                    ShapeHelper.GetMultiIndex(i, tensor.Shape, coords);
                    tensor[coords.ToArray()] = value;
                }
            }
        }
    }

    /// <summary>
    /// Selects elements from x when condition is true, and from y when condition is false (equivalent to np.where).
    /// </summary>
    public static NDArray<T> Where<T>(BoolNDArray condition, NDArray<T> x, NDArray<T> y) where T : unmanaged, INumber<T>
    {
        int[] outShape = BroadcastEngine.BroadcastShapes(x.Shape, y.Shape);
        outShape = BroadcastEngine.BroadcastShapes(condition.Shape, outShape);

        var result = new NDArray<T>(outShape);
        var bX = x.BroadcastTo(outShape);
        var bY = y.BroadcastTo(outShape);

        int total = result.TotalLength;
        Span<int> coords = stackalloc int[outShape.Length];

        for (int i = 0; i < total; i++)
        {
            ShapeHelper.GetMultiIndex(i, outShape, coords);
            var cArr = coords.ToArray();

            bool condVal = condition[cArr];
            result[cArr] = condVal ? bX[cArr] : bY[cArr];
        }

        return result;
    }

    /// <summary>
    /// Gathers slices from a tensor along an axis according to integer indices (equivalent to np.take).
    /// </summary>
    public static NDArray<T> Take<T>(this NDArray<T> tensor, NDArray<int> indices, int axis = 0) where T : unmanaged, INumber<T>
    {
        if (axis < 0) axis += tensor.Rank;
        if (axis < 0 || axis >= tensor.Rank) throw new ArgumentOutOfRangeException(nameof(axis));

        int[] outShape = new int[tensor.Rank];
        Array.Copy(tensor.Shape, outShape, tensor.Rank);
        outShape[axis] = indices.TotalLength;

        var result = new NDArray<T>(outShape);
        int total = result.TotalLength;
        int axisSize = tensor.Shape[axis];

        Span<int> coords = stackalloc int[tensor.Rank];
        for (int i = 0; i < total; i++)
        {
            ShapeHelper.GetMultiIndex(i, outShape, coords);
            var srcCoords = (int[])coords.ToArray().Clone();

            int takeIdx = indices[coords[axis]];
            if (takeIdx < 0) takeIdx += axisSize;
            srcCoords[axis] = takeIdx;

            result[coords.ToArray()] = tensor[srcCoords];
        }

        return result;
    }

    /// <summary>
    /// Scatters values into tensor along an axis according to integer indices (equivalent to np.put_along_axis).
    /// </summary>
    public static void Put<T>(this NDArray<T> tensor, NDArray<int> indices, NDArray<T> values, int axis = 0) where T : unmanaged, INumber<T>
    {
        if (axis < 0) axis += tensor.Rank;
        int axisSize = tensor.Shape[axis];

        Span<int> coords = stackalloc int[values.Rank];
        int valTotal = values.TotalLength;

        for (int i = 0; i < valTotal; i++)
        {
            ShapeHelper.GetMultiIndex(i, values.Shape, coords);
            var dstCoords = (int[])coords.ToArray().Clone();

            int targetIdx = indices[coords[axis]];
            if (targetIdx < 0) targetIdx += axisSize;
            dstCoords[axis] = targetIdx;

            tensor[dstCoords] = values[coords.ToArray()];
        }
    }
}

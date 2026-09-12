using System;
using System.Runtime.CompilerServices;

namespace TokenVector.Numerics.Core;

/// <summary>
/// High-performance mathematics helper for tensor shapes, strides, slicing, and memory layout computations.
/// </summary>
public static class ShapeHelper
{
    /// <summary>
    /// Computes total number of elements in a tensor shape.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int ComputeTotalLength(ReadOnlySpan<int> shape)
    {
        if (shape.Length == 0) return 1; // 0-dim scalar tensor has 1 element
        int total = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] < 0) throw new ArgumentException($"Dimension size cannot be negative: shape[{i}] = {shape[i]}");
            total *= shape[i];
        }
        return total;
    }

    /// <summary>
    /// Computes standard C-contiguous row-major strides for a given shape.
    /// </summary>
    public static int[] ComputeRowMajorStrides(ReadOnlySpan<int> shape)
    {
        if (shape.Length == 0) return Array.Empty<int>();
        int[] strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    /// <summary>
    /// Computes Fortran-contiguous column-major strides for a given shape.
    /// </summary>
    public static int[] ComputeColumnMajorStrides(ReadOnlySpan<int> shape)
    {
        if (shape.Length == 0) return Array.Empty<int>();
        int[] strides = new int[shape.Length];
        int stride = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    /// <summary>
    /// Checks whether the tensor's memory layout is standard C-contiguous without holes or overlaps.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsContiguous(ReadOnlySpan<int> shape, ReadOnlySpan<int> strides)
    {
        if (shape.Length == 0) return true;
        int expectedStride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            if (shape[i] == 0) return true;
            if (shape[i] == 1) continue; // single-element dimensions don't break continuity
            if (strides[i] != expectedStride) return false;
            expectedStride *= shape[i];
        }
        return true;
    }

    /// <summary>
    /// Converts N-dimensional coordinates to flat buffer offset.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetFlatIndex(ReadOnlySpan<int> indices, ReadOnlySpan<int> strides, int offset = 0)
    {
        int flatIndex = offset;
        for (int i = 0; i < indices.Length; i++)
        {
            flatIndex += indices[i] * strides[i];
        }
        return flatIndex;
    }

    /// <summary>
    /// Converts 1D flat offset into multi-dimensional coordinates in-place.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void GetMultiIndex(int linearIndex, ReadOnlySpan<int> shape, Span<int> resultIndices)
    {
        int rem = linearIndex;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            int dim = shape[i];
            if (dim == 0)
            {
                resultIndices[i] = 0;
                continue;
            }
            resultIndices[i] = rem % dim;
            rem /= dim;
        }
    }

    /// <summary>
    /// Computes new shape and strides for permutation of axes.
    /// </summary>
    public static (int[] Shape, int[] Strides) Permute(ReadOnlySpan<int> currentShape, ReadOnlySpan<int> currentStrides, ReadOnlySpan<int> axes)
    {
        if (axes.Length != currentShape.Length)
        {
            throw new ArgumentException("Number of axes in permutation must match tensor rank.");
        }

        Span<bool> visited = stackalloc bool[currentShape.Length];
        int[] newShape = new int[currentShape.Length];
        int[] newStrides = new int[currentShape.Length];

        for (int i = 0; i < axes.Length; i++)
        {
            int axis = axes[i];
            if (axis < 0 || axis >= currentShape.Length || visited[axis])
            {
                throw new ArgumentException($"Invalid or duplicate axis {axis} in permutation.");
            }
            visited[axis] = true;
            newShape[i] = currentShape[axis];
            newStrides[i] = currentStrides[axis];
        }

        return (newShape, newStrides);
    }

    /// <summary>
    /// Computes zero-copy slice parameters (new shape, new strides, and new buffer offset).
    /// </summary>
    public static (int[] Shape, int[] Strides, int Offset) Slice(
        ReadOnlySpan<int> currentShape,
        ReadOnlySpan<int> currentStrides,
        int currentOffset,
        (int Start, int Stop, int Step)[] sliceRanges)
    {
        if (sliceRanges.Length > currentShape.Length)
        {
            throw new ArgumentException("Too many slice dimensions specified.");
        }

        int[] newShape = new int[currentShape.Length];
        int[] newStrides = new int[currentShape.Length];
        int newOffset = currentOffset;

        for (int i = 0; i < currentShape.Length; i++)
        {
            int dim = currentShape[i];
            int stride = currentStrides[i];

            if (i >= sliceRanges.Length)
            {
                newShape[i] = dim;
                newStrides[i] = stride;
                continue;
            }

            var (start, stop, step) = sliceRanges[i];
            if (step == 0) throw new ArgumentException("Slice step cannot be zero.");

            // Normalize negative indices
            if (start < 0) start = Math.Max(0, dim + start);
            if (stop < 0) stop = Math.Max(0, dim + stop);
            start = Math.Clamp(start, 0, dim);
            stop = Math.Clamp(stop, 0, dim);

            if (step > 0)
            {
                if (start >= stop)
                {
                    newShape[i] = 0;
                    newStrides[i] = stride * step;
                }
                else
                {
                    newShape[i] = (stop - start + step - 1) / step;
                    newStrides[i] = stride * step;
                    newOffset += start * stride;
                }
            }
            else
            {
                // Negative step
                if (start <= stop)
                {
                    newShape[i] = 0;
                    newStrides[i] = stride * step;
                }
                else
                {
                    newShape[i] = (start - stop - step - 1) / (-step);
                    newStrides[i] = stride * step;
                    newOffset += start * stride;
                }
            }
        }

        return (newShape, newStrides, newOffset);
    }
}

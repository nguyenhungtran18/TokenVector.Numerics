using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Engine;

namespace TokenVector.Numerics.Core;

public sealed partial class NDArray<T>
{
    #region Global Reductions

    public T Sum()
    {
        if (_isContiguous)
        {
            return SIMDKernels.SumContiguous(AsReadOnlySpan());
        }
        T total = T.Zero;
        foreach (var val in this) total += val;
        return total;
    }

    public T Mean()
    {
        if (_totalLength == 0) return T.Zero;
        return Sum() / T.CreateChecked(_totalLength);
    }

    public T Prod()
    {
        if (_totalLength == 0) return T.Zero;
        T prod = T.One;
        foreach (var val in this) prod *= val;
        return prod;
    }

    public T Min()
    {
        if (_totalLength == 0) throw new InvalidOperationException("Cannot compute Min on empty tensor.");
        T minVal = T.Zero;
        bool first = true;
        foreach (var val in this)
        {
            if (first || val < minVal)
            {
                minVal = val;
                first = false;
            }
        }
        return minVal;
    }

    public T Max()
    {
        if (_totalLength == 0) throw new InvalidOperationException("Cannot compute Max on empty tensor.");
        T maxVal = T.Zero;
        bool first = true;
        foreach (var val in this)
        {
            if (first || val > maxVal)
            {
                maxVal = val;
                first = false;
            }
        }
        return maxVal;
    }

    #endregion

    #region Axis Reductions

    /// <summary>
    /// Computes sum along the specified axis.
    /// </summary>
    public NDArray<T> Sum(int axis, bool keepdims = false) => ReduceAxis(axis, keepdims, T.Zero, (acc, val) => acc + val);

    /// <summary>
    /// Computes mean along the specified axis.
    /// </summary>
    public NDArray<T> Mean(int axis, bool keepdims = false)
    {
        int dimSize = _shape[NormalizeAxis(axis)];
        var sum = Sum(axis, keepdims);
        return sum / T.CreateChecked(dimSize);
    }

    /// <summary>
    /// Computes max along the specified axis.
    /// </summary>
    public NDArray<T> Max(int axis, bool keepdims = false) =>
        ReduceAxis(axis, keepdims, T.Zero, (acc, val) => val > acc ? val : acc, useFirstAsInitial: true);

    /// <summary>
    /// Computes min along the specified axis.
    /// </summary>
    public NDArray<T> Min(int axis, bool keepdims = false) =>
        ReduceAxis(axis, keepdims, T.Zero, (acc, val) => val < acc ? val : acc, useFirstAsInitial: true);

    /// <summary>
    /// Computes indices of maximum values along the specified axis.
    /// </summary>
    public NDArray<int> ArgMax(int axis)
    {
        int normAxis = NormalizeAxis(axis);
        int[] outShape = GetReducedShape(normAxis, keepdims: false);
        var result = new NDArray<int>(outShape);

        int axisSize = _shape[normAxis];
        int outTotal = result.TotalLength;
        Span<int> coords = stackalloc int[outShape.Length];

        for (int i = 0; i < outTotal; i++)
        {
            ShapeHelper.GetMultiIndex(i, outShape, coords);

            int maxIdx = 0;
            T maxVal = T.Zero;
            bool first = true;

            for (int a = 0; a < axisSize; a++)
            {
                int[] fullCoords = ReconstructCoords(coords, normAxis, a);
                T val = this[fullCoords];
                if (first || val > maxVal)
                {
                    maxVal = val;
                    maxIdx = a;
                    first = false;
                }
            }

            result.AsSpan()[i] = maxIdx;
        }

        return result;
    }

    /// <summary>
    /// Computes indices of minimum values along the specified axis.
    /// </summary>
    public NDArray<int> ArgMin(int axis)
    {
        int normAxis = NormalizeAxis(axis);
        int[] outShape = GetReducedShape(normAxis, keepdims: false);
        var result = new NDArray<int>(outShape);

        int axisSize = _shape[normAxis];
        int outTotal = result.TotalLength;
        Span<int> coords = stackalloc int[outShape.Length];

        for (int i = 0; i < outTotal; i++)
        {
            ShapeHelper.GetMultiIndex(i, outShape, coords);

            int minIdx = 0;
            T minVal = T.Zero;
            bool first = true;

            for (int a = 0; a < axisSize; a++)
            {
                int[] fullCoords = ReconstructCoords(coords, normAxis, a);
                T val = this[fullCoords];
                if (first || val < minVal)
                {
                    minVal = val;
                    minIdx = a;
                    first = false;
                }
            }

            result.AsSpan()[i] = minIdx;
        }

        return result;
    }

    #endregion

    #region Helper Axis Methods

    private int NormalizeAxis(int axis)
    {
        if (axis < 0) axis += Rank;
        if (axis < 0 || axis >= Rank)
        {
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} out of range for tensor rank {Rank}.");
        }
        return axis;
    }

    private int[] GetReducedShape(int normalizedAxis, bool keepdims)
    {
        if (keepdims)
        {
            int[] shape = (int[])_shape.Clone();
            shape[normalizedAxis] = 1;
            return shape;
        }

        if (Rank == 1) return Array.Empty<int>(); // scalar 0-D

        int[] newShape = new int[Rank - 1];
        int idx = 0;
        for (int i = 0; i < Rank; i++)
        {
            if (i != normalizedAxis) newShape[idx++] = _shape[i];
        }
        return newShape;
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

    private NDArray<T> ReduceAxis(int axis, bool keepdims, T initial, Func<T, T, T> op, bool useFirstAsInitial = false)
    {
        int normAxis = NormalizeAxis(axis);
        int[] outShape = GetReducedShape(normAxis, keepdims);
        var result = new NDArray<T>(outShape);

        int axisSize = _shape[normAxis];
        int outTotal = result.TotalLength;
        int[] indexLookupShape = keepdims ? GetReducedShape(normAxis, keepdims: false) : outShape;
        Span<int> coords = stackalloc int[indexLookupShape.Length];

        for (int i = 0; i < outTotal; i++)
        {
            ShapeHelper.GetMultiIndex(i, indexLookupShape, coords);

            T acc = initial;
            bool isInit = !useFirstAsInitial;

            for (int a = 0; a < axisSize; a++)
            {
                int[] fullCoords = ReconstructCoords(coords, normAxis, a);
                T val = this[fullCoords];
                if (!isInit)
                {
                    acc = val;
                    isInit = true;
                }
                else
                {
                    acc = op(acc, val);
                }
            }

            result.AsSpan()[i] = acc;
        }

        return result;
    }

    #endregion
}

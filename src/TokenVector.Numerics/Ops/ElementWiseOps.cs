using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Engine;

namespace TokenVector.Numerics.Core;

public sealed partial class NDArray<T>
{
    #region Binary Operators

    public static NDArray<T> operator +(NDArray<T> left, NDArray<T> right) => ElementWiseBinaryOp(left, right, (a, b) => a + b, SIMDKernels.AddContiguous);
    public static NDArray<T> operator -(NDArray<T> left, NDArray<T> right) => ElementWiseBinaryOp(left, right, (a, b) => a - b, SIMDKernels.SubtractContiguous);
    public static NDArray<T> operator *(NDArray<T> left, NDArray<T> right) => ElementWiseBinaryOp(left, right, (a, b) => a * b, SIMDKernels.MultiplyContiguous);
    public static NDArray<T> operator /(NDArray<T> left, NDArray<T> right) => ElementWiseBinaryOp(left, right, (a, b) => a / b, SIMDKernels.DivideContiguous);
    public static NDArray<T> operator %(NDArray<T> left, NDArray<T> right) => ElementWiseBinaryOp(left, right, (a, b) => a % b, null);

    public static NDArray<T> operator -(NDArray<T> value)
    {
        ArgumentNullException.ThrowIfNull(value);
        var result = new NDArray<T>(value.Shape);
        if (value.IsContiguous)
        {
            var src = value.AsReadOnlySpan();
            var dst = result.AsSpan();
            for (int i = 0; i < src.Length; i++) dst[i] = -src[i];
        }
        else
        {
            int total = value.TotalLength;
            Span<int> coords = stackalloc int[value.Rank];
            for (int i = 0; i < total; i++)
            {
                ShapeHelper.GetMultiIndex(i, value.Shape, coords);
                result.AsSpan()[i] = -value[coords.ToArray()];
            }
        }
        return result;
    }

    #endregion

    #region Scalar Overloads

    public static NDArray<T> operator +(NDArray<T> left, T scalar)
    {
        var result = new NDArray<T>(left.Shape);
        var dst = result.AsSpan();
        int i = 0;
        foreach (var val in left) dst[i++] = val + scalar;
        return result;
    }

    public static NDArray<T> operator +(T scalar, NDArray<T> right) => right + scalar;

    public static NDArray<T> operator -(NDArray<T> left, T scalar)
    {
        var result = new NDArray<T>(left.Shape);
        var dst = result.AsSpan();
        int i = 0;
        foreach (var val in left) dst[i++] = val - scalar;
        return result;
    }

    public static NDArray<T> operator -(T scalar, NDArray<T> right)
    {
        var result = new NDArray<T>(right.Shape);
        var dst = result.AsSpan();
        int i = 0;
        foreach (var val in right) dst[i++] = scalar - val;
        return result;
    }

    public static NDArray<T> operator *(NDArray<T> left, T scalar)
    {
        var result = new NDArray<T>(left.Shape);
        var dst = result.AsSpan();
        int i = 0;
        foreach (var val in left) dst[i++] = val * scalar;
        return result;
    }

    public static NDArray<T> operator *(T scalar, NDArray<T> right) => right * scalar;

    public static NDArray<T> operator /(NDArray<T> left, T scalar)
    {
        var result = new NDArray<T>(left.Shape);
        var dst = result.AsSpan();
        int i = 0;
        foreach (var val in left) dst[i++] = val / scalar;
        return result;
    }

    public static NDArray<T> operator /(T scalar, NDArray<T> right)
    {
        var result = new NDArray<T>(right.Shape);
        var dst = result.AsSpan();
        int i = 0;
        foreach (var val in right) dst[i++] = scalar / val;
        return result;
    }

    #endregion

    #region Helper Execution Kernel

    private delegate void ContiguousKernel<U>(ReadOnlySpan<U> a, ReadOnlySpan<U> b, Span<U> dest) where U : unmanaged, INumber<U>;

    private static NDArray<T> ElementWiseBinaryOp(
        NDArray<T> left,
        NDArray<T> right,
        Func<T, T, T> scalarOp,
        ContiguousKernel<T>? simdKernel)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        // Check if shapes are identical
        bool sameShape = left.Shape.AsSpan().SequenceEqual(right.Shape.AsSpan());
        if (sameShape)
        {
            var result = new NDArray<T>(left.Shape);
            if (left.IsContiguous && right.IsContiguous && simdKernel != null)
            {
                simdKernel(left.AsReadOnlySpan(), right.AsReadOnlySpan(), result.AsSpan());
                return result;
            }

            // General same-shape path
            int total = result.TotalLength;
            var destSpan = result.AsSpan();
            Span<int> coords = stackalloc int[left.Rank];
            for (int i = 0; i < total; i++)
            {
                ShapeHelper.GetMultiIndex(i, left.Shape, coords);
                var cArr = coords.ToArray();
                destSpan[i] = scalarOp(left[cArr], right[cArr]);
            }
            return result;
        }

        // Broadcast path
        int[] outShape = BroadcastEngine.BroadcastShapes(left.Shape, right.Shape);
        var bLeft = left.BroadcastTo(outShape);
        var bRight = right.BroadcastTo(outShape);

        var bResult = new NDArray<T>(outShape);
        int outTotal = bResult.TotalLength;
        int rank = outShape.Length;

        Parallel.For(0, outTotal, i =>
        {
            Span<int> coords = stackalloc int[rank];
            ShapeHelper.GetMultiIndex(i, outShape, coords);
            var cArr = coords.ToArray();
            bResult.Buffer[i] = scalarOp(bLeft[cArr], bRight[cArr]);
        });

        return bResult;
    }

    #endregion
}

using System;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Ops;

/// <summary>
/// Elementwise comparison operations for NDArray returning BoolNDArray.
/// </summary>
public static class ComparisonOps
{
    public static BoolNDArray GreaterThan<T>(this NDArray<T> a, T scalar) where T : unmanaged, INumber<T> =>
        CompareScalar(a, scalar, (x, y) => x > y);

    public static BoolNDArray GreaterThanOrEqual<T>(this NDArray<T> a, T scalar) where T : unmanaged, INumber<T> =>
        CompareScalar(a, scalar, (x, y) => x >= y);

    public static BoolNDArray LessThan<T>(this NDArray<T> a, T scalar) where T : unmanaged, INumber<T> =>
        CompareScalar(a, scalar, (x, y) => x < y);

    public static BoolNDArray LessThanOrEqual<T>(this NDArray<T> a, T scalar) where T : unmanaged, INumber<T> =>
        CompareScalar(a, scalar, (x, y) => x <= y);

    public static BoolNDArray Equal<T>(this NDArray<T> a, T scalar) where T : unmanaged, INumber<T> =>
        CompareScalar(a, scalar, (x, y) => x == y);

    public static BoolNDArray NotEqual<T>(this NDArray<T> a, T scalar) where T : unmanaged, INumber<T> =>
        CompareScalar(a, scalar, (x, y) => x != y);

    public static BoolNDArray GreaterThan<T>(this NDArray<T> a, NDArray<T> b) where T : unmanaged, INumber<T> =>
        CompareTensor(a, b, (x, y) => x > y);

    public static BoolNDArray LessThan<T>(this NDArray<T> a, NDArray<T> b) where T : unmanaged, INumber<T> =>
        CompareTensor(a, b, (x, y) => x < y);

    private static BoolNDArray CompareScalar<T>(NDArray<T> a, T scalar, Func<T, T, bool> pred) where T : unmanaged, INumber<T>
    {
        var result = new BoolNDArray(a.Shape);
        int i = 0;
        foreach (var val in a)
        {
            result.Buffer[i++] = pred(val, scalar);
        }
        return result;
    }

    private static BoolNDArray CompareTensor<T>(NDArray<T> a, NDArray<T> b, Func<T, T, bool> pred) where T : unmanaged, INumber<T>
    {
        int[] outShape = Engine.BroadcastEngine.BroadcastShapes(a.Shape, b.Shape);
        var res = new BoolNDArray(outShape);
        var bA = a.BroadcastTo(outShape);
        var bB = b.BroadcastTo(outShape);

        int total = res.TotalLength;
        Span<int> coords = stackalloc int[outShape.Length];

        for (int i = 0; i < total; i++)
        {
            ShapeHelper.GetMultiIndex(i, outShape, coords);
            var cArr = coords.ToArray();
            res[cArr] = pred(bA[cArr], bB[cArr]);
        }
        return res;
    }
}

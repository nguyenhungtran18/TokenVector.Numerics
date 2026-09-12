// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;
using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Engine;

namespace TokenVector.Numerics.Ops;

/// <summary>
/// Provides tolerance-based equality testing, boolean reductions, and floating-point status checks (np.isclose, np.allclose, np.all, np.any, np.isnan, np.isinf, np.isfinite).
/// </summary>
public static class LogicToleranceOps
{
    /// <summary>
    /// Returns a boolean tensor where two arrays are element-wise equal within a given tolerance (np.isclose).
    /// Condition: |a - b| &lt;= (atol + rtol * |b|)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static BoolNDArray IsClose<T>(NDArray<T> a, NDArray<T> b, double rtol = 1e-5, double atol = 1e-8) where T : unmanaged, INumber<T>
    {
        int[] outShape = BroadcastEngine.BroadcastShapes(a.Shape, b.Shape);
        var res = new BoolNDArray(outShape);

        var aContig = a.BroadcastTo(outShape).Contiguous();
        var bContig = b.BroadcastTo(outShape).Contiguous();

        Parallel.For(0, res.TotalLength, i =>
        {
            double va = double.CreateTruncating(aContig[i]);
            double vb = double.CreateTruncating(bContig[i]);

            if (double.IsNaN(va) || double.IsNaN(vb))
            {
                res[i] = false;
            }
            else if (double.IsInfinity(va) || double.IsInfinity(vb))
            {
                res[i] = va == vb;
            }
            else
            {
                double diff = Math.Abs(va - vb);
                double allowed = atol + rtol * Math.Abs(vb);
                res[i] = diff <= allowed;
            }
        });

        return res;
    }

    /// <summary>
    /// Returns true if two arrays are element-wise equal within a tolerance (np.allclose).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static bool AllClose<T>(NDArray<T> a, NDArray<T> b, double rtol = 1e-5, double atol = 1e-8) where T : unmanaged, INumber<T>
    {
        var closeMask = IsClose(a, b, rtol, atol);
        return All(closeMask);
    }

    /// <summary>
    /// Evaluates whether all elements evaluate to true over the entire tensor.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static bool All(BoolNDArray mask)
    {
        for (int i = 0; i < mask.TotalLength; i++)
        {
            if (!mask[i]) return false;
        }
        return true;
    }

    /// <summary>
    /// Evaluates whether any element evaluates to true over the entire tensor.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static bool Any(BoolNDArray mask)
    {
        for (int i = 0; i < mask.TotalLength; i++)
        {
            if (mask[i]) return true;
        }
        return false;
    }

    /// <summary>
    /// Evaluates all along a given axis.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static BoolNDArray All(BoolNDArray mask, int axis)
    {
        int normAxis = NormalizeAxis(axis, mask.Rank);
        int[] outShape = new int[mask.Rank - 1];
        int idx = 0;
        for (int i = 0; i < mask.Rank; i++)
        {
            if (i != normAxis) outShape[idx++] = mask.Shape[i];
        }
        if (outShape.Length == 0) outShape = [1];

        var result = new BoolNDArray(outShape);
        int axisDim = mask.Shape[normAxis];
        int outerSize = 1;
        for (int i = 0; i < normAxis; i++) outerSize *= mask.Shape[i];
        int innerSize = 1;
        for (int i = normAxis + 1; i < mask.Rank; i++) innerSize *= mask.Shape[i];

        Parallel.For(0, outerSize, o =>
        {
            for (int inn = 0; inn < innerSize; inn++)
            {
                bool allTrue = true;
                for (int a = 0; a < axisDim; a++)
                {
                    int flatIdx = (o * axisDim + a) * innerSize + inn;
                    if (!mask[flatIdx])
                    {
                        allTrue = false;
                        break;
                    }
                }
                int outFlatIdx = o * innerSize + inn;
                result[outFlatIdx] = allTrue;
            }
        });

        return result;
    }

    /// <summary>
    /// Evaluates any along a given axis.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static BoolNDArray Any(BoolNDArray mask, int axis)
    {
        int normAxis = NormalizeAxis(axis, mask.Rank);
        int[] outShape = new int[mask.Rank - 1];
        int idx = 0;
        for (int i = 0; i < mask.Rank; i++)
        {
            if (i != normAxis) outShape[idx++] = mask.Shape[i];
        }
        if (outShape.Length == 0) outShape = [1];

        var result = new BoolNDArray(outShape);
        int axisDim = mask.Shape[normAxis];
        int outerSize = 1;
        for (int i = 0; i < normAxis; i++) outerSize *= mask.Shape[i];
        int innerSize = 1;
        for (int i = normAxis + 1; i < mask.Rank; i++) innerSize *= mask.Shape[i];

        Parallel.For(0, outerSize, o =>
        {
            for (int inn = 0; inn < innerSize; inn++)
            {
                bool anyTrue = false;
                for (int a = 0; a < axisDim; a++)
                {
                    int flatIdx = (o * axisDim + a) * innerSize + inn;
                    if (mask[flatIdx])
                    {
                        anyTrue = true;
                        break;
                    }
                }
                int outFlatIdx = o * innerSize + inn;
                result[outFlatIdx] = anyTrue;
            }
        });

        return result;
    }

    /// <summary>
    /// Returns a boolean array indicating where elements are NaN (np.isnan).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static BoolNDArray IsNaN<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        var res = new BoolNDArray(tensor.Shape);
        var contig = tensor.Contiguous();

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double val = double.CreateTruncating(contig[i]);
            res[i] = double.IsNaN(val);
        });

        return res;
    }

    /// <summary>
    /// Returns a boolean array indicating where elements are positive or negative infinity (np.isinf).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static BoolNDArray IsInf<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        var res = new BoolNDArray(tensor.Shape);
        var contig = tensor.Contiguous();

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double val = double.CreateTruncating(contig[i]);
            res[i] = double.IsInfinity(val);
        });

        return res;
    }

    /// <summary>
    /// Returns a boolean array indicating where elements are finite (not NaN, not Inf) (np.isfinite).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static BoolNDArray IsFinite<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        var res = new BoolNDArray(tensor.Shape);
        var contig = tensor.Contiguous();

        Parallel.For(0, tensor.TotalLength, i =>
        {
            double val = double.CreateTruncating(contig[i]);
            res[i] = double.IsFinite(val);
        });

        return res;
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

// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;
using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Engine;

namespace TokenVector.Numerics.Ops;

/// <summary>
/// Provides bitwise logical and shifting operations on integer tensors (np.bitwise_and, np.bitwise_or, np.bitwise_xor, np.bitwise_not, np.left_shift, np.right_shift).
/// </summary>
public static class BitwiseOps
{
    /// <summary>
    /// Computes bitwise AND of two integer tensors element-wise (np.bitwise_and).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> BitwiseAnd<T>(NDArray<T> a, NDArray<T> b) where T : unmanaged, IBinaryInteger<T>
    {
        int[] outShape = BroadcastEngine.BroadcastShapes(a.Shape, b.Shape);
        var res = new NDArray<T>(outShape);

        var aContig = a.BroadcastTo(outShape).Contiguous();
        var bContig = b.BroadcastTo(outShape).Contiguous();

        Parallel.For(0, res.TotalLength, i =>
        {
            res[i] = aContig[i] & bContig[i];
        });

        return res;
    }

    /// <summary>
    /// Computes bitwise OR of two integer tensors element-wise (np.bitwise_or).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> BitwiseOr<T>(NDArray<T> a, NDArray<T> b) where T : unmanaged, IBinaryInteger<T>
    {
        int[] outShape = BroadcastEngine.BroadcastShapes(a.Shape, b.Shape);
        var res = new NDArray<T>(outShape);

        var aContig = a.BroadcastTo(outShape).Contiguous();
        var bContig = b.BroadcastTo(outShape).Contiguous();

        Parallel.For(0, res.TotalLength, i =>
        {
            res[i] = aContig[i] | bContig[i];
        });

        return res;
    }

    /// <summary>
    /// Computes bitwise XOR of two integer tensors element-wise (np.bitwise_xor).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> BitwiseXor<T>(NDArray<T> a, NDArray<T> b) where T : unmanaged, IBinaryInteger<T>
    {
        int[] outShape = BroadcastEngine.BroadcastShapes(a.Shape, b.Shape);
        var res = new NDArray<T>(outShape);

        var aContig = a.BroadcastTo(outShape).Contiguous();
        var bContig = b.BroadcastTo(outShape).Contiguous();

        Parallel.For(0, res.TotalLength, i =>
        {
            res[i] = aContig[i] ^ bContig[i];
        });

        return res;
    }

    /// <summary>
    /// Computes bitwise NOT (inversion) of an integer tensor element-wise (np.bitwise_not / ~a).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> BitwiseNot<T>(NDArray<T> a) where T : unmanaged, IBinaryInteger<T>
    {
        var res = new NDArray<T>(a.Shape);
        var aContig = a.Contiguous();

        Parallel.For(0, res.TotalLength, i =>
        {
            res[i] = ~aContig[i];
        });

        return res;
    }

    /// <summary>
    /// Shifts the bits of an integer tensor to the left (np.left_shift / a &lt;&lt; shift).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> LeftShift<T>(NDArray<T> a, int shift) where T : unmanaged, IBinaryInteger<T>
    {
        var res = new NDArray<T>(a.Shape);
        var aContig = a.Contiguous();

        Parallel.For(0, res.TotalLength, i =>
        {
            res[i] = aContig[i] << shift;
        });

        return res;
    }

    /// <summary>
    /// Shifts the bits of an integer tensor to the right (np.right_shift / a &gt;&gt; shift).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> RightShift<T>(NDArray<T> a, int shift) where T : unmanaged, IBinaryInteger<T>
    {
        var res = new NDArray<T>(a.Shape);
        var aContig = a.Contiguous();

        Parallel.For(0, res.TotalLength, i =>
        {
            res[i] = aContig[i] >> shift;
        });

        return res;
    }
}

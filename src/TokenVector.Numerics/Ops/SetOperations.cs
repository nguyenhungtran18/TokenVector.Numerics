// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;
using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Ops;

/// <summary>
/// Provides 1D set operations on tensors (np.intersect1d, np.union1d, np.setdiff1d, np.setxor1d, np.isin).
/// </summary>
public static class SetOperations
{
    /// <summary>
    /// Find the intersection of two arrays. Return the sorted, unique values that are in both of the input arrays.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Intersect1D<T>(NDArray<T> ar1, NDArray<T> ar2) where T : unmanaged, INumber<T>
    {
        var u1 = SortingOps.Unique(ar1.Contiguous());
        var u2 = SortingOps.Unique(ar2.Contiguous());

        var set2 = new HashSet<T>();
        for (int i = 0; i < u2.TotalLength; i++) set2.Add(u2[i]);

        var resultList = new List<T>();
        for (int i = 0; i < u1.TotalLength; i++)
        {
            if (set2.Contains(u1[i]))
            {
                resultList.Add(u1[i]);
            }
        }

        resultList.Sort();
        return NDArray<T>.FromArray(resultList.ToArray(), resultList.Count);
    }

    /// <summary>
    /// Find the union of two arrays. Return the unique, sorted array of values that are in either of the two input arrays.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Union1D<T>(NDArray<T> ar1, NDArray<T> ar2) where T : unmanaged, INumber<T>
    {
        var set = new HashSet<T>();
        var c1 = ar1.Contiguous();
        var c2 = ar2.Contiguous();

        for (int i = 0; i < c1.TotalLength; i++) set.Add(c1[i]);
        for (int i = 0; i < c2.TotalLength; i++) set.Add(c2[i]);

        var resultList = new List<T>(set);
        resultList.Sort();
        return NDArray<T>.FromArray(resultList.ToArray(), resultList.Count);
    }

    /// <summary>
    /// Find the set difference of two arrays. Return the unique values in ar1 that are not in ar2.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> SetDiff1D<T>(NDArray<T> ar1, NDArray<T> ar2) where T : unmanaged, INumber<T>
    {
        var u1 = SortingOps.Unique(ar1.Contiguous());
        var u2 = SortingOps.Unique(ar2.Contiguous());

        var set2 = new HashSet<T>();
        for (int i = 0; i < u2.TotalLength; i++) set2.Add(u2[i]);

        var resultList = new List<T>();
        for (int i = 0; i < u1.TotalLength; i++)
        {
            if (!set2.Contains(u1[i]))
            {
                resultList.Add(u1[i]);
            }
        }

        resultList.Sort();
        return NDArray<T>.FromArray(resultList.ToArray(), resultList.Count);
    }

    /// <summary>
    /// Find the set exclusive-or (symmetric difference) of two arrays.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> SetXor1D<T>(NDArray<T> ar1, NDArray<T> ar2) where T : unmanaged, INumber<T>
    {
        var diff1 = SetDiff1D(ar1, ar2);
        var diff2 = SetDiff1D(ar2, ar1);
        return Union1D(diff1, diff2);
    }

    /// <summary>
    /// Calculates element-wise whether each element of `element` is in `testElements` (np.isin).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static BoolNDArray IsIn<T>(NDArray<T> element, NDArray<T> testElements) where T : unmanaged, INumber<T>
    {
        var testSet = new HashSet<T>();
        var tContig = testElements.Contiguous();
        for (int i = 0; i < tContig.TotalLength; i++)
        {
            testSet.Add(tContig[i]);
        }

        var eContig = element.Contiguous();
        var res = new BoolNDArray(element.Shape);

        Parallel.For(0, element.TotalLength, i =>
        {
            res[i] = testSet.Contains(eContig[i]);
        });

        return res;
    }
}

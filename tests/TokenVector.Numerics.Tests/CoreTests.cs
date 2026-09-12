using System;
using TokenVector.Numerics.Core;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class CoreTests
{
    [Fact]
    public void TestNDArrayCreationAndIndexing()
    {
        var arr = new NDArray<float>(2, 3);
        Assert.Equal(2, arr.Shape[0]);
        Assert.Equal(3, arr.Shape[1]);
        Assert.Equal(6, arr.TotalLength);

        arr[0, 0] = 1.0f;
        arr[0, 1] = 2.0f;
        arr[0, 2] = 3.0f;
        arr[1, 0] = 4.0f;
        arr[1, 1] = 5.0f;
        arr[1, 2] = 6.0f;

        Assert.Equal(1.0f, arr[0, 0]);
        Assert.Equal(6.0f, arr[1, 2]);
    }

    [Fact]
    public void TestZeroCopySlicing()
    {
        var arr = NDArray<float>.Arange(0.0f, 12.0f, 1.0f).Reshape(3, 4);

        // Slice row 1..3, col 1..4 with step 2
        var slice = arr.Slice((1, 3, 1), (1, 4, 2));

        Assert.Equal(2, slice.Shape[0]);
        Assert.Equal(2, slice.Shape[1]);

        // slice[0,0] is arr[1, 1] = 5.0
        Assert.Equal(5.0f, slice[0, 0]);
        // slice[0,1] is arr[1, 3] = 7.0
        Assert.Equal(7.0f, slice[0, 1]);
        // slice[1,0] is arr[2, 1] = 9.0
        Assert.Equal(9.0f, slice[1, 0]);

        // Modify slice and check that original buffer reflects changes (Zero-Copy)
        slice[0, 0] = 99.0f;
        Assert.Equal(99.0f, arr[1, 1]);
    }

    [Fact]
    public void TestReshapeAndPermute()
    {
        var arr = NDArray<float>.Arange(0.0f, 24.0f, 1.0f).Reshape(2, 3, 4);
        Assert.Equal(3, arr.Rank);

        var permuted = arr.Permute(2, 0, 1);
        Assert.Equal(new[] { 4, 2, 3 }, permuted.Shape);

        // Transpose 2D
        var mat = new NDArray<float>(2, 3);
        mat[0, 1] = 10.0f;
        var t = mat.MatrixTranspose();
        Assert.Equal(new[] { 3, 2 }, t.Shape);
        Assert.Equal(10.0f, t[1, 0]);
    }

    [Fact]
    public void TestNativeMemoryAllocation()
    {
        using var nativeArr = NDArray<float>.AllocateNative(4, 4);
        nativeArr[2, 2] = 42.0f;
        Assert.Equal(42.0f, nativeArr[2, 2]);
        Assert.True(nativeArr.Buffer.IsNative);
    }
}

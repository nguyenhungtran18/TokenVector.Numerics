using System;
using TokenVector.Numerics.Core;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class BroadcastAndSIMDTests
{
    [Fact]
    public void TestBroadcastingAddition()
    {
        // a: (2, 1) = [[10], [20]]
        var a = NDArray<float>.FromArray(new[] { 10.0f, 20.0f }, 2, 1);
        // b: (1, 3) = [[1, 2, 3]]
        var b = NDArray<float>.FromArray(new[] { 1.0f, 2.0f, 3.0f }, 1, 3);

        // c: (2, 3)
        var c = a + b;

        Assert.Equal(new[] { 2, 3 }, c.Shape);
        Assert.Equal(11.0f, c[0, 0]);
        Assert.Equal(12.0f, c[0, 1]);
        Assert.Equal(13.0f, c[0, 2]);
        Assert.Equal(21.0f, c[1, 0]);
        Assert.Equal(22.0f, c[1, 1]);
        Assert.Equal(23.0f, c[1, 2]);
    }

    [Fact]
    public void TestSIMDContiguousArithmetic()
    {
        int size = 1024;
        var a = NDArray<float>.Ones(size) * 5.0f;
        var b = NDArray<float>.Ones(size) * 3.0f;

        var sum = a + b;
        var diff = a - b;
        var prod = a * b;
        var quot = a / b;

        Assert.Equal(8.0f, sum[500]);
        Assert.Equal(2.0f, diff[500]);
        Assert.Equal(15.0f, prod[500]);
        Assert.Equal(5.0f / 3.0f, quot[500], precision: 5);
    }

    [Fact]
    public void TestReductions()
    {
        var mat = NDArray<float>.FromArray(new[]
        {
            1.0f, 5.0f, 3.0f,
            4.0f, 2.0f, 6.0f
        }, 2, 3);

        Assert.Equal(21.0f, mat.Sum());
        Assert.Equal(3.5f, mat.Mean());
        Assert.Equal(1.0f, mat.Min());
        Assert.Equal(6.0f, mat.Max());

        // Sum along axis 0: [5, 7, 9]
        var sum0 = mat.Sum(0);
        Assert.Equal(new[] { 3 }, sum0.Shape);
        Assert.Equal(5.0f, sum0[0]);
        Assert.Equal(7.0f, sum0[1]);
        Assert.Equal(9.0f, sum0[2]);

        // ArgMax along axis 1: [1, 2]
        var argMax1 = mat.ArgMax(1);
        Assert.Equal(1, argMax1[0]);
        Assert.Equal(2, argMax1[1]);
    }
}

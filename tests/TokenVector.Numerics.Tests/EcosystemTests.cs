using System;
using System.IO;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.IO;
using TokenVector.Numerics.LinAlg;
using TokenVector.Numerics.Ops;
using TokenVector.Numerics.Random;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class EcosystemTests
{
    [Fact]
    public void TestRandomGenerators()
    {
        TensorRandom.ManualSeed(42);

        // Rand in [0, 1)
        var u = TensorRandom.Rand<float>(100);
        Assert.Equal(100, u.TotalLength);
        foreach (var val in u)
        {
            Assert.True(val >= 0.0f && val < 1.0f);
        }

        // Randn with Mean ~ 0 and Std ~ 1
        var g = TensorRandom.Randn<double>(0.0, 1.0, 1000);
        double mean = g.Mean();
        double std = g.Std()[0];
        Assert.True(Math.Abs(mean) < 0.15, $"Mean {mean} deviated from 0");
        Assert.True(Math.Abs(std - 1.0) < 0.15, $"Std {std} deviated from 1");

        // Randint in [10, 20)
        var ints = TensorRandom.Randint(10, 20, 50);
        foreach (var iVal in ints)
        {
            Assert.True(iVal >= 10 && iVal < 20);
        }

        // Choice
        var pop = NDArray<float>.Arange(0.0f, 10.0f, 1.0f);
        var sample = TensorRandom.Choice(pop, 5, replace: false);
        Assert.Equal(5, sample.TotalLength);
    }

    [Fact]
    public void TestArrayManipulation()
    {
        var a = NDArray<float>.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, 2, 2);
        var b = NDArray<float>.FromArray(new[] { 5.0f, 6.0f, 7.0f, 8.0f }, 2, 2);

        // Concatenate axis 0 -> [4, 2]
        var cat0 = ManipulationOps.Concatenate(new[] { a, b }, axis: 0);
        Assert.Equal(new[] { 4, 2 }, cat0.Shape);
        Assert.Equal(1.0f, cat0[0, 0]);
        Assert.Equal(5.0f, cat0[2, 0]);

        // Stack axis 0 -> [2, 2, 2]
        var stacked = ManipulationOps.Stack(new[] { a, b }, axis: 0);
        Assert.Equal(new[] { 2, 2, 2 }, stacked.Shape);
        Assert.Equal(1.0f, stacked[0, 0, 0]);
        Assert.Equal(5.0f, stacked[1, 0, 0]);

        // Split
        var splits = ManipulationOps.Split(cat0, 2, axis: 0);
        Assert.Equal(2, splits.Length);
        Assert.Equal(new[] { 2, 2 }, splits[0].Shape);
        Assert.Equal(1.0f, splits[0][0, 0]);

        // Tile [2, 2] by (2, 3) -> [4, 6]
        var tiled = ManipulationOps.Tile(a, 2, 3);
        Assert.Equal(new[] { 4, 6 }, tiled.Shape);
        Assert.Equal(1.0f, tiled[2, 2]); // repeats a[0, 0]

        // Pad
        var padded = ManipulationOps.Pad(a, new[] { (1, 1), (1, 1) }, constantValue: 0.0f);
        Assert.Equal(new[] { 4, 4 }, padded.Shape);
        Assert.Equal(0.0f, padded[0, 0]);
        Assert.Equal(1.0f, padded[1, 1]);

        // Roll
        var rollVec = NDArray<float>.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, 4);
        var rolled = ManipulationOps.Roll(rollVec, 1);
        Assert.Equal(4.0f, rolled[0]);
        Assert.Equal(1.0f, rolled[1]);

        // Flip
        var flipped = ManipulationOps.Flip(rollVec, 0);
        Assert.Equal(4.0f, flipped[0]);
        Assert.Equal(1.0f, flipped[3]);
    }

    [Fact]
    public void TestSortingAndSearching()
    {
        var unsorted = NDArray<float>.FromArray(new[] { 5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 2.0f }, 6);

        // Sort
        var sorted = unsorted.Sort(0);
        Assert.Equal(1.0f, sorted[0]);
        Assert.Equal(2.0f, sorted[1]);
        Assert.Equal(9.0f, sorted[5]);

        // ArgSort
        var indices = unsorted.ArgSort(0);
        Assert.Equal(3, indices[0]); // index of 1.0f is 3

        // Unique
        var u = unsorted.Unique();
        Assert.Equal(5, u.TotalLength); // {1, 2, 5, 8, 9}
        Assert.Equal(1.0f, u[0]);
        Assert.Equal(9.0f, u[4]);

        // Clip
        var clipped = unsorted.Clip(2.0f, 6.0f);
        Assert.Equal(5.0f, clipped[0]);
        Assert.Equal(2.0f, clipped[1]); // clamped
        Assert.Equal(6.0f, clipped[2]); // 8.0 clamped to 6.0
        Assert.Equal(2.0f, clipped[3]); // 1.0 clamped to 2.0
    }

    [Fact]
    public void TestStatistics()
    {
        var data = NDArray<double>.FromArray(new[] { 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 }, 8);

        // Mean = 5.0
        Assert.Equal(5.0, data.Mean(), precision: 5);

        // Var (population ddof=0) = 4.0
        var variance = data.Var(ddof: 0);
        Assert.Equal(4.0, variance[0], precision: 5);

        // Std = 2.0
        var std = data.Std(ddof: 0);
        Assert.Equal(2.0, std[0], precision: 5);

        // Median = 4.5
        var median = data.Median();
        Assert.Equal(4.5, median[0], precision: 5);

        // Covariance & Correlation between 2 variables of 3 observations
        var matrix = NDArray<double>.FromArray(new[]
        {
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0
        }, 2, 3);

        var corr = StatisticsOps.CorrCoef(matrix);
        // Variables are perfectly linearly correlated -> corr = 1.0
        Assert.Equal(1.0, corr[0, 1], precision: 5);
        Assert.Equal(1.0, corr[1, 0], precision: 5);
    }

    [Fact]
    public void TestExtendedLinAlg()
    {
        // Rectangular matrix A: 3x2
        var a = NDArray<double>.FromArray(new[]
        {
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        }, 3, 2);

        // 1. Pseudoinverse PInv: A * A^+ * A = A
        var pinv = LinAlgExtended.PInv(a);
        Assert.Equal(new[] { 2, 3 }, pinv.Shape);

        var aPinv = MatrixMultiplication.MatMul(a, pinv);
        var aPinvA = MatrixMultiplication.MatMul(aPinv, a);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(a[i, j], aPinvA[i, j], precision: 4);

        // 2. Matrix Rank of 3x2 full rank matrix = 2
        int rank = LinAlgExtended.MatrixRank(a);
        Assert.Equal(2, rank);

        // 3. Least Squares LstSq: Ax = b
        var b = NDArray<double>.FromArray(new[] { 1.0, 2.0, 3.0 }, 3);
        var x = LinAlgExtended.LstSq(a, b);
        Assert.Equal(new[] { 2 }, x.Shape);

        // 4. Condition number
        double cond = LinAlgExtended.Cond(a);
        Assert.True(cond > 1.0);
    }

    [Fact]
    public void TestNpyAndBinaryIO()
    {
        string tmpNpy = Path.Combine(Path.GetTempPath(), $"test_tensor_{Guid.NewGuid():N}.npy");
        string tmpBin = Path.Combine(Path.GetTempPath(), $"test_tensor_{Guid.NewGuid():N}.bin");

        try
        {
            var original = NDArray<float>.FromArray(new[]
            {
                1.5f, 2.5f, 3.5f,
                4.5f, 5.5f, 6.5f
            }, 2, 3);

            // Test NumPy .npy Save & Load
            TensorIO.SaveNpy(tmpNpy, original);
            Assert.True(File.Exists(tmpNpy));

            var loadedNpy = TensorIO.LoadNpy<float>(tmpNpy);
            Assert.Equal(original.Shape, loadedNpy.Shape);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    Assert.Equal(original[i, j], loadedNpy[i, j]);

            // Test Raw Binary Save & Load
            TensorIO.SaveBinary(tmpBin, original);
            var loadedBin = TensorIO.LoadBinary<float>(tmpBin);
            Assert.Equal(original.Shape, loadedBin.Shape);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    Assert.Equal(original[i, j], loadedBin[i, j]);
        }
        finally
        {
            if (File.Exists(tmpNpy)) File.Delete(tmpNpy);
            if (File.Exists(tmpBin)) File.Delete(tmpBin);
        }
    }
}

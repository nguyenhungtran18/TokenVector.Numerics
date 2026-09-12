// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.IO;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.IO;
using TokenVector.Numerics.LinAlg;
using TokenVector.Numerics.Ops;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class FinalParityTests
{
    [Fact]
    public void Test_CumulativeOps_CumSum_And_CumProd()
    {
        var x = NDArray<float>.FromArray([1f, 2f, 3f, 4f], 2, 2);

        // Flat CumSum: [1, 3, 6, 10]
        var csFlat = CumulativeOps.CumSum(x);
        Assert.Equal(4, csFlat.TotalLength);
        Assert.Equal(1f, csFlat[0]);
        Assert.Equal(3f, csFlat[1]);
        Assert.Equal(6f, csFlat[2]);
        Assert.Equal(10f, csFlat[3]);

        // Axis 0 CumSum:
        // [1, 2]
        // [1+3, 2+4] = [4, 6]
        var csAxis0 = CumulativeOps.CumSum(x, axis: 0);
        Assert.Equal(1f, csAxis0[0, 0]);
        Assert.Equal(2f, csAxis0[0, 1]);
        Assert.Equal(4f, csAxis0[1, 0]);
        Assert.Equal(6f, csAxis0[1, 1]);

        // Axis 1 CumProd:
        // [1, 1*2] = [1, 2]
        // [3, 3*4] = [3, 12]
        var cpAxis1 = CumulativeOps.CumProd(x, axis: 1);
        Assert.Equal(1f, cpAxis1[0, 0]);
        Assert.Equal(2f, cpAxis1[0, 1]);
        Assert.Equal(3f, cpAxis1[1, 0]);
        Assert.Equal(12f, cpAxis1[1, 1]);
    }

    [Fact]
    public void Test_CumulativeOps_Diff()
    {
        var x = NDArray<float>.FromArray([1f, 2f, 4f, 7f, 11f], 5);

        // 1st diff: [2-1, 4-2, 7-4, 11-7] = [1, 2, 3, 4]
        var d1 = CumulativeOps.Diff(x, n: 1);
        Assert.Equal(4, d1.TotalLength);
        Assert.Equal(1f, d1[0]);
        Assert.Equal(2f, d1[1]);
        Assert.Equal(3f, d1[2]);
        Assert.Equal(4f, d1[3]);

        // 2nd diff: [2-1, 3-2, 4-3] = [1, 1, 1]
        var d2 = CumulativeOps.Diff(x, n: 2);
        Assert.Equal(3, d2.TotalLength);
        Assert.Equal(1f, d2[0]);
        Assert.Equal(1f, d2[1]);
        Assert.Equal(1f, d2[2]);
    }

    [Fact]
    public void Test_Polynomial_PolyFit_PolyVal_Roots()
    {
        // Fit quadratic y = 2*x^2 + 3*x + 1
        var x = NDArray<double>.FromArray([0.0, 1.0, 2.0, 3.0, 4.0], 5);
        var y = NDArray<double>.FromArray([1.0, 6.0, 15.0, 28.0, 45.0], 5);

        var coeffs = Polynomial.PolyFit(x, y, deg: 2);
        Assert.Equal(3, coeffs.TotalLength);
        Assert.True(Math.Abs(coeffs[0] - 2.0) < 1e-4);
        Assert.True(Math.Abs(coeffs[1] - 3.0) < 1e-4);
        Assert.True(Math.Abs(coeffs[2] - 1.0) < 1e-4);

        // PolyVal at x = 5 => 2*(25) + 3*5 + 1 = 50 + 15 + 1 = 66
        var evalPoints = NDArray<double>.FromArray([5.0], 1);
        var evalRes = Polynomial.PolyVal(coeffs, evalPoints);
        Assert.True(Math.Abs(evalRes[0] - 66.0) < 1e-4);

        // Roots of x^2 - 5x + 6 = 0 => (x-2)(x-3) = 0 => roots 3, 2
        var quadCoeffs = NDArray<double>.FromArray([1.0, -5.0, 6.0], 3);
        var roots = Polynomial.Roots(quadCoeffs);
        Assert.Equal(2, roots.Length);
        var r1 = roots[0];
        var r2 = roots[1];
        Assert.True((Math.Abs(r1.Real - 3.0) < 1e-4 && Math.Abs(r2.Real - 2.0) < 1e-4) ||
                    (Math.Abs(r1.Real - 2.0) < 1e-4 && Math.Abs(r2.Real - 3.0) < 1e-4));
    }

    [Fact]
    public void Test_GridOps_Meshgrid_Diag_Triu_Tril()
    {
        var x = NDArray<float>.FromArray([1f, 2f, 3f], 3);
        var y = NDArray<float>.FromArray([4f, 5f], 2);

        // Meshgrid 2D
        var grids = GridOps.Meshgrid(x, y);
        Assert.Equal(2, grids.Length);
        var gx = grids[0];
        var gy = grids[1];
        Assert.Equal(new[] { 2, 3 }, gx.Shape);
        Assert.Equal(new[] { 2, 3 }, gy.Shape);

        Assert.Equal(1f, gx[0, 0]);
        Assert.Equal(2f, gx[0, 1]);
        Assert.Equal(3f, gx[0, 2]);
        Assert.Equal(4f, gy[0, 0]);
        Assert.Equal(5f, gy[1, 0]);

        // Diag 1D -> 2D
        var diagMat = GridOps.Diag(x);
        Assert.Equal(new[] { 3, 3 }, diagMat.Shape);
        Assert.Equal(1f, diagMat[0, 0]);
        Assert.Equal(2f, diagMat[1, 1]);
        Assert.Equal(3f, diagMat[2, 2]);
        Assert.Equal(0f, diagMat[0, 1]);

        // Diag 2D -> 1D
        var diagVec = GridOps.Diag(diagMat);
        Assert.Equal(3, diagVec.TotalLength);
        Assert.Equal(1f, diagVec[0]);
        Assert.Equal(2f, diagVec[1]);
        Assert.Equal(3f, diagVec[2]);

        // Triu / Tril
        var full = NDArray<float>.FromArray([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f], 3, 3);
        var upper = GridOps.Triu(full);
        Assert.Equal(1f, upper[0, 0]);
        Assert.Equal(2f, upper[0, 1]);
        Assert.Equal(0f, upper[1, 0]);
        Assert.Equal(0f, upper[2, 0]);

        var lower = GridOps.Tril(full);
        Assert.Equal(1f, lower[0, 0]);
        Assert.Equal(0f, lower[0, 1]);
        Assert.Equal(4f, lower[1, 0]);
        Assert.Equal(5f, lower[1, 1]);
    }

    [Fact]
    public void Test_LogicToleranceOps_IsClose_AllClose_FiniteChecks()
    {
        var a = NDArray<float>.FromArray([1.000001f, 2.0f, 3.0f], 3);
        var b = NDArray<float>.FromArray([1.000002f, 2.0f, 3.0f], 3);

        var close = LogicToleranceOps.IsClose(a, b, rtol: 1e-4, atol: 1e-4);
        Assert.True(close[0]);
        Assert.True(close[1]);
        Assert.True(close[2]);

        bool allClose = LogicToleranceOps.AllClose(a, b, rtol: 1e-4, atol: 1e-4);
        Assert.True(allClose);

        var nanInfArr = NDArray<float>.FromArray([float.NaN, float.PositiveInfinity, 42.0f], 3);
        var isNan = LogicToleranceOps.IsNaN(nanInfArr);
        Assert.True(isNan[0]);
        Assert.False(isNan[1]);
        Assert.False(isNan[2]);

        var isInf = LogicToleranceOps.IsInf(nanInfArr);
        Assert.False(isInf[0]);
        Assert.True(isInf[1]);
        Assert.False(isInf[2]);

        var isFinite = LogicToleranceOps.IsFinite(nanInfArr);
        Assert.False(isFinite[0]);
        Assert.False(isFinite[1]);
        Assert.True(isFinite[2]);
    }

    [Fact]
    public void Test_TensorArchive_Npz_And_MemMap()
    {
        string npzPath = Path.Combine(Path.GetTempPath(), $"test_archive_{Guid.NewGuid():N}.npz");
        string mmapPath = Path.Combine(Path.GetTempPath(), $"test_mmap_{Guid.NewGuid():N}.bin");

        try
        {
            // 1. NPZ Archive save & load
            var weights = NDArray<float>.FromArray([1.0f, 2.0f, 3.0f, 4.0f], 2, 2);
            var bias = NDArray<float>.FromArray([0.5f, -0.5f], 2);

            var dict = new Dictionary<string, NDArray<float>>
            {
                ["weights"] = weights,
                ["bias"] = bias
            };

            TensorArchive.SaveNpz(npzPath, dict, compress: true);
            Assert.True(File.Exists(npzPath));

            var loaded = TensorArchive.LoadNpz<float>(npzPath);
            Assert.Equal(2, loaded.Count);
            Assert.True(loaded.ContainsKey("weights"));
            Assert.True(loaded.ContainsKey("bias"));

            var lWeights = loaded["weights"];
            Assert.Equal(weights.Shape, lWeights.Shape);
            Assert.Equal(1.0f, lWeights[0, 0]);
            Assert.Equal(4.0f, lWeights[1, 1]);

            var lBias = loaded["bias"];
            Assert.Equal(bias.Shape, lBias.Shape);
            Assert.Equal(0.5f, lBias[0]);
            Assert.Equal(-0.5f, lBias[1]);

            // 2. MemoryMappedNDArray
            using (var mmap = new MemoryMappedNDArray<float>(mmapPath, [10, 10]))
            {
                mmap.SetValue(999.0f, 3, 4);
                mmap.Flush();
                Assert.Equal(999.0f, mmap.GetValue(3, 4));

                var mat = mmap.ToNDArray();
                Assert.Equal(new[] { 10, 10 }, mat.Shape);
                Assert.Equal(999.0f, mat[3, 4]);
            }
        }
        finally
        {
            if (File.Exists(npzPath)) File.Delete(npzPath);
            if (File.Exists(mmapPath)) File.Delete(mmapPath);
        }
    }
}

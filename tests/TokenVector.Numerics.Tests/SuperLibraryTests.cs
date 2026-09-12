// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;
using TokenVector.Numerics.Ops;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class SuperLibraryTests
{
    [Fact]
    public void Test_SignalProcessing_Convolve_Correlate_Windows()
    {
        var a = NDArray<float>.FromArray([1f, 2f, 3f], 3);
        var v = NDArray<float>.FromArray([0f, 1f, 0.5f], 3);

        // Full convolution: [0, 1, 2.5, 4, 1.5]
        var convFull = SignalProcessing.Convolve(a, v, ConvMode.Full);
        Assert.Equal(5, convFull.TotalLength);
        Assert.Equal(0f, convFull[0]);
        Assert.Equal(1f, convFull[1]);
        Assert.Equal(2.5f, convFull[2]);
        Assert.Equal(4f, convFull[3]);
        Assert.Equal(1.5f, convFull[4]);

        // Same convolution: length max(3, 3) = 3 -> [1, 2.5, 4]
        var convSame = SignalProcessing.Convolve(a, v, ConvMode.Same);
        Assert.Equal(3, convSame.TotalLength);

        // Valid convolution: length 1 -> [2.5]
        var convValid = SignalProcessing.Convolve(a, v, ConvMode.Valid);
        Assert.Equal(1, convValid.TotalLength);
        Assert.Equal(2.5f, convValid[0]);

        // Window functions
        var hann = SignalProcessing.Hanning(5);
        Assert.Equal(5, hann.TotalLength);
        Assert.Equal(0.0, hann[0], precision: 5);
        Assert.Equal(1.0, hann[2], precision: 5);
        Assert.Equal(0.0, hann[4], precision: 5);

        var hamm = SignalProcessing.Hamming(5);
        Assert.Equal(5, hamm.TotalLength);
        Assert.True(hamm[0] > 0.0);

        var black = SignalProcessing.Blackman(5);
        Assert.Equal(5, black.TotalLength);

        var bart = SignalProcessing.Bartlett(5);
        Assert.Equal(5, bart.TotalLength);
        Assert.Equal(0.0, bart[0], precision: 5);
        Assert.Equal(1.0, bart[2], precision: 5);
    }

    [Fact]
    public void Test_SpecialFunctions_Erf_Gamma_Bessel()
    {
        var x = NDArray<double>.FromArray([0.0, 1.0, -1.0], 3);

        // Erf: erf(0) = 0, erf(1) ~ 0.8427, erf(-1) ~ -0.8427
        var erfX = SpecialFunctions.Erf(x);
        Assert.Equal(0.0, erfX[0], precision: 4);
        Assert.Equal(0.8427, erfX[1], precision: 3);
        Assert.Equal(-0.8427, erfX[2], precision: 3);

        // Erfc: erfc(0) = 1.0
        var erfcX = SpecialFunctions.Erfc(x);
        Assert.Equal(1.0, erfcX[0], precision: 4);

        // Gamma: Gamma(1) = 1, Gamma(2) = 1, Gamma(3) = 2, Gamma(4) = 6, Gamma(5) = 24
        var gInput = NDArray<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0], 5);
        var gOutput = SpecialFunctions.Gamma(gInput);
        Assert.Equal(1.0, gOutput[0], precision: 4);
        Assert.Equal(1.0, gOutput[1], precision: 4);
        Assert.Equal(2.0, gOutput[2], precision: 4);
        Assert.Equal(6.0, gOutput[3], precision: 4);
        Assert.Equal(24.0, gOutput[4], precision: 4);

        // LogGamma: ln(Gamma(5)) = ln(24) ~ 3.17805
        var lgInput = NDArray<double>.FromArray([5.0], 1);
        var lgOutput = SpecialFunctions.LogGamma(lgInput);
        Assert.Equal(Math.Log(24.0), lgOutput[0], precision: 4);

        // Bessel I0(0) = 1, J0(0) = 1
        var bInput = NDArray<double>.FromArray([0.0], 1);
        var i0 = SpecialFunctions.BesselI0(bInput);
        var j0 = SpecialFunctions.BesselJ0(bInput);
        Assert.Equal(1.0, i0[0], precision: 4);
        Assert.Equal(1.0, j0[0], precision: 4);
    }

    [Fact]
    public void Test_SetOperations_Intersect_Union_Diff_IsIn()
    {
        var ar1 = NDArray<int>.FromArray([1, 2, 2, 3, 4], 5);
        var ar2 = NDArray<int>.FromArray([2, 4, 6], 3);

        // Intersect: [2, 4]
        var inter = SetOperations.Intersect1D(ar1, ar2);
        Assert.Equal(2, inter.TotalLength);
        Assert.Equal(2, inter[0]);
        Assert.Equal(4, inter[1]);

        // Union: [1, 2, 3, 4, 6]
        var un = SetOperations.Union1D(ar1, ar2);
        Assert.Equal(5, un.TotalLength);
        Assert.Equal(1, un[0]);
        Assert.Equal(6, un[4]);

        // SetDiff: [1, 3]
        var diff = SetOperations.SetDiff1D(ar1, ar2);
        Assert.Equal(2, diff.TotalLength);
        Assert.Equal(1, diff[0]);
        Assert.Equal(3, diff[1]);

        // SetXor: [1, 3, 6]
        var xor = SetOperations.SetXor1D(ar1, ar2);
        Assert.Equal(3, xor.TotalLength);
        Assert.Equal(1, xor[0]);
        Assert.Equal(3, xor[1]);
        Assert.Equal(6, xor[2]);

        // IsIn:
        var isIn = SetOperations.IsIn(ar1, ar2);
        Assert.False(isIn[0]); // 1 not in ar2
        Assert.True(isIn[1]);  // 2 in ar2
        Assert.True(isIn[2]);  // 2 in ar2
        Assert.False(isIn[3]); // 3 not in ar2
        Assert.True(isIn[4]);  // 4 in ar2
    }

    [Fact]
    public void Test_MatrixOps_Kron_Outer_MatrixPower_Norm()
    {
        // Kron
        var a = NDArray<float>.FromArray([1f, 2f, 3f, 4f], 2, 2);
        var b = NDArray<float>.FromArray([0f, 5f, 6f, 7f], 2, 2);
        var kron = MatrixOps.Kron(a, b);
        Assert.Equal(new[] { 4, 4 }, kron.Shape);
        Assert.Equal(0f, kron[0, 0]);
        Assert.Equal(5f, kron[0, 1]);
        Assert.Equal(0f, kron[0, 2]);
        Assert.Equal(10f, kron[0, 3]);

        // Outer product: [1, 2] outer [3, 4, 5] -> shape (2, 3)
        var u = NDArray<float>.FromArray([1f, 2f], 2);
        var v = NDArray<float>.FromArray([3f, 4f, 5f], 3);
        var outer = MatrixOps.Outer(u, v);
        Assert.Equal(new[] { 2, 3 }, outer.Shape);
        Assert.Equal(3f, outer[0, 0]);
        Assert.Equal(4f, outer[0, 1]);
        Assert.Equal(5f, outer[0, 2]);
        Assert.Equal(6f, outer[1, 0]);
        Assert.Equal(8f, outer[1, 1]);
        Assert.Equal(10f, outer[1, 2]);

        // MatrixPower: A^3
        var mat = NDArray<double>.FromArray([1.0, 1.0, 0.0, 1.0], 2, 2);
        var matCubed = MatrixOps.MatrixPower(mat, 3);
        Assert.Equal(1.0, matCubed[0, 0]);
        Assert.Equal(3.0, matCubed[0, 1]);
        Assert.Equal(0.0, matCubed[1, 0]);
        Assert.Equal(1.0, matCubed[1, 1]);

        // Norm
        var vec = NDArray<double>.FromArray([3.0, 4.0], 2);
        Assert.Equal(5.0, MatrixOps.Norm(vec, "2"), precision: 5);
        Assert.Equal(7.0, MatrixOps.Norm(vec, "1"), precision: 5);
        Assert.Equal(4.0, MatrixOps.Norm(vec, "inf"), precision: 5);
    }

    [Fact]
    public void Test_BitwiseOps()
    {
        var x = NDArray<int>.FromArray([0b1100, 0b1010], 2);
        var y = NDArray<int>.FromArray([0b1010, 0b1100], 2);

        // Bitwise AND: 0b1000 = 8
        var andRes = BitwiseOps.BitwiseAnd(x, y);
        Assert.Equal(8, andRes[0]);
        Assert.Equal(8, andRes[1]);

        // Bitwise OR: 0b1110 = 14
        var orRes = BitwiseOps.BitwiseOr(x, y);
        Assert.Equal(14, orRes[0]);
        Assert.Equal(14, orRes[1]);

        // Bitwise XOR: 0b0110 = 6
        var xorRes = BitwiseOps.BitwiseXor(x, y);
        Assert.Equal(6, xorRes[0]);
        Assert.Equal(6, xorRes[1]);

        // Shifts:
        var shl = BitwiseOps.LeftShift(x, 1);
        Assert.Equal(24, shl[0]); // 12 << 1 = 24
        Assert.Equal(20, shl[1]); // 10 << 1 = 20

        var shr = BitwiseOps.RightShift(x, 1);
        Assert.Equal(6, shr[0]);  // 12 >> 1 = 6
        Assert.Equal(5, shr[1]);  // 10 >> 1 = 5
    }
}

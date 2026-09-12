using System;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;
using TokenVector.Numerics.Ops;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class AdvancedMathTests
{
    [Fact]
    public void TestSVDDecomposition()
    {
        // 3x2 Rectangular matrix A
        var a = NDArray<double>.FromArray(new[]
        {
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        }, 3, 2);

        var (u, s, vt) = SVD.Decompose(a);

        Assert.Equal(new[] { 3, 2 }, u.Shape);
        Assert.Equal(new[] { 2 }, s.Shape);
        Assert.Equal(new[] { 2, 2 }, vt.Shape);

        // Singular values must be strictly positive and decreasing
        Assert.True(s[0] > s[1]);
        Assert.True(s[1] > 0.0);

        // Reconstruct A = U * diag(S) * Vt
        var diagS = NDArray<double>.Zeros(2, 2);
        diagS[0, 0] = s[0];
        diagS[1, 1] = s[1];

        var uS = MatrixMultiplication.MatMul(u, diagS);
        var reconstructed = MatrixMultiplication.MatMul(uS, vt);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(a[i, j], reconstructed[i, j], precision: 4);
            }
        }
    }

    [Fact]
    public void TestEigenSymmetricEigh()
    {
        // Symmetric matrix: [[2, 1], [1, 2]]
        var a = NDArray<double>.FromArray(new[]
        {
            2.0, 1.0,
            1.0, 2.0
        }, 2, 2);

        var (eigvals, eigvecs) = Eigen.Eigh(a);

        // Eigenvalues of [[2, 1], [1, 2]] are 1.0 and 3.0
        Assert.Equal(1.0, eigvals[0], precision: 5);
        Assert.Equal(3.0, eigvals[1], precision: 5);

        // Check A * v_i = lambda_i * v_i
        for (int i = 0; i < 2; i++)
        {
            var v_i = new NDArray<double>(2, 1);
            v_i[0, 0] = eigvecs[0, i];
            v_i[1, 0] = eigvecs[1, i];

            var av = MatrixMultiplication.MatMul(a, v_i);
            var lambdaV = v_i * eigvals[i];

            Assert.Equal(lambdaV[0, 0], av[0, 0], precision: 5);
            Assert.Equal(lambdaV[1, 0], av[1, 0], precision: 5);
        }
    }

    [Fact]
    public void TestArbitraryLengthBluesteinFFT()
    {
        // Odd prime length signal N = 7 (Cannot be processed by pure Radix-2 without Bluestein)
        var signal = new Complex<double>[]
        {
            new(1.0, 0.0),
            new(2.0, 0.0),
            new(3.0, 0.0),
            new(4.0, 0.0),
            new(5.0, 0.0),
            new(6.0, 0.0),
            new(7.0, 0.0)
        };

        var spectrum = FFT.FFT1D<double>(signal);
        Assert.Equal(7, spectrum.Length);

        // DC Component = 1+2+3+4+5+6+7 = 28
        Assert.Equal(28.0, spectrum[0].Real, precision: 4);
        Assert.Equal(0.0, spectrum[0].Imaginary, precision: 4);

        // IFFT back to original
        var restored = FFT.IFFT1D<double>(spectrum);
        for (int i = 0; i < 7; i++)
        {
            Assert.Equal(signal[i].Real, restored[i].Real, precision: 4);
            Assert.Equal(signal[i].Imaginary, restored[i].Imaginary, precision: 4);
        }
    }

    [Fact]
    public void TestEinSumContraction()
    {
        // 1. Matrix Multiplication via EinSum: "ij,jk->ik"
        var a = NDArray<float>.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, 2, 2);
        var b = NDArray<float>.FromArray(new[] { 5.0f, 6.0f, 7.0f, 8.0f }, 2, 2);

        var c = EinSum.Evaluate("ij,jk->ik", a, b);
        Assert.Equal(new[] { 2, 2 }, c.Shape);
        Assert.Equal(19.0f, c[0, 0]); // 1*5 + 2*7 = 19
        Assert.Equal(22.0f, c[0, 1]); // 1*6 + 2*8 = 22
        Assert.Equal(43.0f, c[1, 0]); // 3*5 + 4*7 = 43
        Assert.Equal(50.0f, c[1, 1]); // 3*6 + 4*8 = 50

        // 2. Trace via EinSum: "ii->"
        var trace = EinSum.Evaluate("ii->", a);
        Assert.Equal(1.0f + 4.0f, trace[0]);

        // 3. Transpose via EinSum: "ij->ji"
        var aT = EinSum.Evaluate("ij->ji", a);
        Assert.Equal(2.0f, aT[1, 0]);
        Assert.Equal(3.0f, aT[0, 1]);
    }

    [Fact]
    public void TestBooleanMaskingAndWhere()
    {
        // arr: [1, 2, 3, 4, 5, 6]
        var arr = NDArray<float>.Arange(1.0f, 7.0f, 1.0f);

        // Mask arr > 3
        var mask = arr.GreaterThan(3.0f);
        var filtered = arr.Filter(mask);

        Assert.Equal(3, filtered.TotalLength);
        Assert.Equal(4.0f, filtered[0]);
        Assert.Equal(5.0f, filtered[1]);
        Assert.Equal(6.0f, filtered[2]);

        // Where: if arr > 3 choose 100, else choose 0
        var zeros = NDArray<float>.Zeros(6);
        var hundreds = NDArray<float>.Ones(6) * 100.0f;
        var whereResult = IndexingOps.Where(mask, hundreds, zeros);

        Assert.Equal(0.0f, whereResult[0]);
        Assert.Equal(0.0f, whereResult[2]);
        Assert.Equal(100.0f, whereResult[3]);
        Assert.Equal(100.0f, whereResult[5]);
    }

    [Fact]
    public void TestTakeAndPut()
    {
        // Tensor [3, 4]
        var mat = NDArray<float>.Arange(0.0f, 12.0f, 1.0f).Reshape(3, 4);

        // Take rows [2, 0]
        var indices = NDArray<int>.FromArray(new[] { 2, 0 }, 2);
        var gathered = mat.Take(indices, axis: 0);

        Assert.Equal(new[] { 2, 4 }, gathered.Shape);
        Assert.Equal(8.0f, gathered[0, 0]);  // row 2 col 0
        Assert.Equal(0.0f, gathered[1, 0]);  // row 0 col 0
    }
}

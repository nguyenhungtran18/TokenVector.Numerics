using System;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class LinAlgTests
{
    [Fact]
    public void TestMatrixMultiplication2D()
    {
        // A: 2x3, B: 3x2
        var a = NDArray<float>.FromArray(new[]
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        }, 2, 3);

        var b = NDArray<float>.FromArray(new[]
        {
            7.0f, 8.0f,
            9.0f, 1.0f,
            2.0f, 3.0f
        }, 3, 2);

        var c = MatrixMultiplication.MatMul(a, b);

        Assert.Equal(new[] { 2, 2 }, c.Shape);
        // c[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
        Assert.Equal(31.0f, c[0, 0]);
        // c[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
        Assert.Equal(19.0f, c[0, 1]);
        // c[1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
        Assert.Equal(85.0f, c[1, 0]);
        // c[1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
        Assert.Equal(55.0f, c[1, 1]);
    }

    [Fact]
    public void TestLUDecompositionAndSolve()
    {
        // A = [[4, 3], [6, 3]]
        var a = NDArray<double>.FromArray(new[]
        {
            4.0, 3.0,
            6.0, 3.0
        }, 2, 2);

        var (p, l, u) = Decomposition.LU(a);

        // Check P*A = L*U
        var pa = MatrixMultiplication.MatMul(p, a);
        var lu = MatrixMultiplication.MatMul(l, u);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(pa[i, j], lu[i, j], precision: 5);

        // Solve Ax = b where b = [10, 12]
        var b = NDArray<double>.FromArray(new[] { 10.0, 12.0 }, 2);
        var x = Decomposition.Solve(a, b);

        // 4x_0 + 3x_1 = 10; 6x_0 + 3x_1 = 12 -> 2x_0 = 2 -> x_0 = 1, x_1 = 2
        Assert.Equal(1.0, x[0], precision: 5);
        Assert.Equal(2.0, x[1], precision: 5);
    }

    [Fact]
    public void TestDeterminantAndInverse()
    {
        var a = NDArray<double>.FromArray(new[]
        {
            1.0, 2.0,
            3.0, 4.0
        }, 2, 2);

        // Det(A) = 1*4 - 2*3 = -2
        double det = Decomposition.Det(a);
        Assert.Equal(-2.0, det, precision: 5);

        // A * A^-1 = I
        var inv = Decomposition.Inverse(a);
        var ident = MatrixMultiplication.MatMul(a, inv);

        Assert.Equal(1.0, ident[0, 0], precision: 5);
        Assert.Equal(0.0, ident[0, 1], precision: 5);
        Assert.Equal(0.0, ident[1, 0], precision: 5);
        Assert.Equal(1.0, ident[1, 1], precision: 5);
    }

    [Fact]
    public void TestFFT1D()
    {
        // 4-point signal: [1, 2, 3, 4]
        var signal = new Complex<double>[]
        {
            new(1.0, 0.0),
            new(2.0, 0.0),
            new(3.0, 0.0),
            new(4.0, 0.0)
        };

        var spectrum = FFT.FFT1D<double>(signal);
        // DC component (index 0) = 1 + 2 + 3 + 4 = 10
        Assert.Equal(10.0, spectrum[0].Real, precision: 5);
        Assert.Equal(0.0, spectrum[0].Imaginary, precision: 5);

        // IFFT back to original
        var restored = FFT.IFFT1D<double>(spectrum);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(signal[i].Real, restored[i].Real, precision: 5);
            Assert.Equal(signal[i].Imaginary, restored[i].Imaginary, precision: 5);
        }
    }
}

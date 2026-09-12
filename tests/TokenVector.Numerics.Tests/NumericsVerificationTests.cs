using System;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;
using TokenVector.Numerics.Neural;
using Xunit;

namespace TokenVector.Numerics.Tests;

/// <summary>
/// Benchmark & End-to-End Verification Test Suite required by TokenVector Numerics spec.
/// </summary>
public class NumericsVerificationTests
{
    [Fact]
    public void Test3DZeroCopySlicing()
    {
        // 1. Slicing mảng 3D không copy RAM
        // Shape: [2, 3, 4] = 24 elements
        var tensor = NDArray<float>.Arange(0.0f, 24.0f, 1.0f).Reshape(2, 3, 4);

        // Slice batch 0..2, rows 1..3, cols 1..4 step 2 -> Shape: [2, 2, 2]
        var slice = tensor.Slice((0, 2, 1), (1, 3, 1), (1, 4, 2));

        Assert.Equal(new[] { 2, 2, 2 }, slice.Shape);

        // Modifying slice must reflect in the original tensor without cloning RAM
        float originalVal = tensor[1, 1, 1];
        slice[1, 0, 0] = 777.0f; // slice[1, 0, 0] points to tensor[1, 1, 1]

        Assert.Equal(777.0f, tensor[1, 1, 1]);
        Assert.NotEqual(originalVal, tensor[1, 1, 1]);
    }

    [Fact]
    public void TestBroadcasting2x1And1x3()
    {
        // 2. Broadcasting giữa ma trận (2, 1) và (1, 3) -> (2, 3)
        var a = NDArray<float>.FromArray(new[] { 10.0f, 20.0f }, 2, 1);
        var b = NDArray<float>.FromArray(new[] { 1.0f, 2.0f, 3.0f }, 1, 3);

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
    public void TestMatMulSoftmaxAttentionAndLUDecomposition()
    {
        // 3. Độ chính xác của MatMul
        var m1 = NDArray<float>.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, 2, 2);
        var m2 = NDArray<float>.FromArray(new[] { 2.0f, 0.0f, 1.0f, 2.0f }, 2, 2);
        var matmulRes = MatrixMultiplication.MatMul(m1, m2);

        // [1*2 + 2*1, 1*0 + 2*2] = [4, 4]
        // [3*2 + 4*1, 3*0 + 4*2] = [10, 8]
        Assert.Equal(4.0f, matmulRes[0, 0]);
        Assert.Equal(4.0f, matmulRes[0, 1]);
        Assert.Equal(10.0f, matmulRes[1, 0]);
        Assert.Equal(8.0f, matmulRes[1, 1]);

        // 4. Độ chính xác của Softmax (với Log-Sum-Exp trick chống tràn số)
        var largeLogits = NDArray<float>.FromArray(new[] { 500.0f, 501.0f, 502.0f }, 1, 3);
        var probs = Activations.Softmax(largeLogits, axis: -1);
        Assert.Equal(1.0f, probs.Sum(), precision: 5);
        Assert.True(probs[0, 2] > probs[0, 1] && probs[0, 1] > probs[0, 0]);

        // 5. Độ chính xác của Scaled Dot-Product Attention
        var q = NDArray<float>.FromArray(new[] { 1.0f, 0.0f, 0.0f, 1.0f }, 1, 2, 2);
        var k = NDArray<float>.FromArray(new[] { 1.0f, 0.0f, 0.0f, 1.0f }, 1, 2, 2);
        var v = NDArray<float>.FromArray(new[] { 5.0f, 0.0f, 0.0f, 10.0f }, 1, 2, 2);

        var (attnOut, attnWeights) = AttentionEngine.ScaledDotProductAttention(q, k, v);
        Assert.Equal(new[] { 1, 2, 2 }, attnOut.Shape);
        Assert.Equal(1.0f, attnWeights[0, 0, 0] + attnWeights[0, 0, 1], precision: 5);

        // 6. Phân rã ma trận LU
        var aMat = NDArray<double>.FromArray(new[]
        {
            2.0, 1.0,
            6.0, 8.0
        }, 2, 2);

        var (p, l, u) = Decomposition.LU(aMat);
        var reconstructed = MatrixMultiplication.MatMul(l, u);
        var pA = MatrixMultiplication.MatMul(p, aMat);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(pA[i, j], reconstructed[i, j], precision: 5);
            }
        }
    }
}

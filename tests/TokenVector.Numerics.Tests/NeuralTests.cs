using System;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Neural;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class NeuralTests
{
    [Fact]
    public void TestSoftmaxNumericalStability()
    {
        // Extreme values that would overflow naive exp(x)
        var logits = NDArray<float>.FromArray(new[] { 1000.0f, 1001.0f, 1002.0f }, 1, 3);
        var probs = Activations.Softmax(logits, axis: -1);

        // Sum of probabilities must be 1.0
        Assert.Equal(1.0f, probs.Sum(), precision: 5);
        Assert.False(float.IsNaN(probs[0, 0]));
        Assert.False(float.IsInfinity(probs[0, 0]));
    }

    [Fact]
    public void TestGELUActivation()
    {
        var x = NDArray<float>.FromArray(new[] { -2.0f, 0.0f, 2.0f }, 3);
        var gelu = Activations.GELU(x);

        // GELU(0) = 0
        Assert.Equal(0.0f, gelu[1], precision: 5);
        // GELU(2) ≈ 1.95459
        Assert.True(MathF.Abs(gelu[2] - 1.95459f) < 1e-3f);
        // GELU(-2) ≈ -0.0454
        Assert.True(MathF.Abs(gelu[0] - (-0.0454f)) < 1e-3f);
    }

    [Fact]
    public void TestIm2ColAndConv2D()
    {
        // 1 image, 1 channel, 3x3
        var img = NDArray<float>.FromArray(new[]
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f
        }, 1, 1, 3, 3);

        // 1 filter, 1 channel, 2x2 filter of all ones
        var weights = NDArray<float>.Ones(1, 1, 2, 2);

        var output = ConvEngine.Conv2D(img, weights, strideH: 1, strideW: 1, padH: 0, padW: 0);

        // Output size: [1, 1, 2, 2]
        Assert.Equal(new[] { 1, 1, 2, 2 }, output.Shape);
        // output[0,0,0,0] = 1 + 2 + 4 + 5 = 12
        Assert.Equal(12.0f, output[0, 0, 0, 0]);
        // output[0,0,0,1] = 2 + 3 + 5 + 6 = 16
        Assert.Equal(16.0f, output[0, 0, 0, 1]);
        // output[0,0,1,0] = 4 + 5 + 7 + 8 = 24
        Assert.Equal(24.0f, output[0, 0, 1, 0]);
        // output[0,0,1,1] = 5 + 6 + 8 + 9 = 28
        Assert.Equal(28.0f, output[0, 0, 1, 1]);
    }

    [Fact]
    public void TestScaledDotProductAttention()
    {
        // Batch 1, SeqLen 2, Dim 2
        var q = NDArray<float>.FromArray(new[]
        {
            1.0f, 0.0f,
            0.0f, 1.0f
        }, 1, 2, 2);

        var k = NDArray<float>.FromArray(new[]
        {
            1.0f, 0.0f,
            0.0f, 1.0f
        }, 1, 2, 2);

        var v = NDArray<float>.FromArray(new[]
        {
            10.0f, 0.0f,
            0.0f, 20.0f
        }, 1, 2, 2);

        var (output, attnWeights) = AttentionEngine.ScaledDotProductAttention(q, k, v);

        Assert.Equal(new[] { 1, 2, 2 }, output.Shape);
        Assert.Equal(new[] { 1, 2, 2 }, attnWeights.Shape);

        // Sum of attention weights across key dimension must be 1
        Assert.Equal(1.0f, attnWeights[0, 0, 0] + attnWeights[0, 0, 1], precision: 5);
        Assert.Equal(1.0f, attnWeights[0, 1, 0] + attnWeights[0, 1, 1], precision: 5);
    }

    [Fact]
    public void TestLayerNorm()
    {
        // Tensor: [1, 4]
        var x = NDArray<float>.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f }, 1, 4);
        var (normX, mean, invStd) = Normalization.LayerNorm(x);

        // Mean should be 2.5
        Assert.Equal(2.5f, mean[0, 0], precision: 5);
        // Normalized tensor mean should be 0.0 and std should be 1.0
        Assert.Equal(0.0f, normX.Mean(), precision: 5);
    }
}

using System;
using TokenVector.Numerics.Autograd;
using TokenVector.Numerics.Autograd.NN;
using TokenVector.Numerics.Autograd.Nodes;
using TokenVector.Numerics.Autograd.Optim;
using TokenVector.Numerics.Core;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class AutogradTests
{
    [Fact]
    public void Test_ScalarArithmetic_Autograd()
    {
        // z = (x + y) * (x - y) = x^2 - y^2
        // dz/dx = 2x, dz/dy = -2y
        var x = new Tensor<double>(3.0, requiresGrad: true);
        var y = new Tensor<double>(2.0, requiresGrad: true);

        var a = x + y; // 5.0
        var b = x - y; // 1.0
        var z = a * b; // 5.0

        Assert.Equal(5.0, z.Data.Buffer[0], precision: 6);

        z.Backward();

        Assert.NotNull(x.Grad);
        Assert.NotNull(y.Grad);
        Assert.Equal(6.0, x.Grad.Buffer[0], precision: 6);   // 2 * 3 = 6
        Assert.Equal(-4.0, y.Grad.Buffer[0], precision: 6);  // -2 * 2 = -4
    }

    [Fact]
    public void Test_MultiBranch_Autograd()
    {
        // z = x * x * x = x^3
        // dz/dx = 3 * x^2
        var x = new Tensor<double>(4.0, requiresGrad: true);
        var z = x * x * x;

        Assert.Equal(64.0, z.Data.Buffer[0], precision: 6);

        z.Backward();

        Assert.NotNull(x.Grad);
        Assert.Equal(48.0, x.Grad.Buffer[0], precision: 6); // 3 * 4^2 = 48
    }

    [Fact]
    public void Test_MatrixMultiplication_And_Unbroadcasting_Autograd()
    {
        // X is (2, 3), W is (3, 2), b is (1, 2)
        var xArr = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var wArr = new double[] { 0.5, -0.5, 1.0, 2.0, -1.5, 0.5 };
        var bArr = new double[] { 0.1, 0.2 };

        var X = new Tensor<double>(xArr, new[] { 2, 3 }, requiresGrad: true);
        var W = new Tensor<double>(wArr, new[] { 3, 2 }, requiresGrad: true);
        var b = new Tensor<double>(bArr, new[] { 1, 2 }, requiresGrad: true);

        // Y = X * W + b
        var Y = X.MatMul(W) + b;
        Assert.Equal(new[] { 2, 2 }, Y.Shape);

        var loss = Y.Sum();
        loss.Backward();

        Assert.NotNull(X.Grad);
        Assert.NotNull(W.Grad);
        Assert.NotNull(b.Grad);

        // Gradient of b should be unbroadcasted to (1, 2) and equal to [2, 2] (since 2 rows)
        Assert.Equal(new[] { 1, 2 }, b.Grad.Shape);
        Assert.Equal(2.0, b.Grad.Buffer[0], precision: 6);
        Assert.Equal(2.0, b.Grad.Buffer[1], precision: 6);

        // Shapes of gradients must match source tensors
        Assert.Equal(X.Shape, X.Grad.Shape);
        Assert.Equal(W.Shape, W.Grad.Shape);
    }

    [Fact]
    public void Test_ActivationFunctions_Autograd()
    {
        // ReLU
        var xRelu = new Tensor<double>(new[] { -2.0, 0.5, 3.0 }, new[] { 3 }, requiresGrad: true);
        var yRelu = xRelu.Relu().Sum();
        yRelu.Backward();
        Assert.Equal(new[] { 0.0, 1.0, 1.0 }, xRelu.Grad!.AsSpan().ToArray());

        // Sigmoid at 0: sigma(0) = 0.5, grad = 0.5 * (1 - 0.5) = 0.25
        var xSig = new Tensor<double>(0.0, requiresGrad: true);
        var ySig = xSig.Sigmoid();
        ySig.Backward();
        Assert.Equal(0.25, xSig.Grad!.Buffer[0], precision: 6);

        // Tanh at 0: tanh(0) = 0, grad = 1 - 0^2 = 1.0
        var xTanh = new Tensor<double>(0.0, requiresGrad: true);
        var yTanh = xTanh.Tanh();
        yTanh.Backward();
        Assert.Equal(1.0, xTanh.Grad!.Buffer[0], precision: 6);
    }

    [Fact]
    public void Test_LossFunctions_Autograd()
    {
        var yPred = new Tensor<double>(new[] { 2.0, 4.0 }, new[] { 2, 1 }, requiresGrad: true);
        var yTrue = new Tensor<double>(new[] { 1.0, 3.0 }, new[] { 2, 1 }, requiresGrad: false);

        // MSE = Mean((yPred - yTrue)^2) = Mean([1, 1]) = 1.0
        var mse = ActivationAndReductionNodes<double>.MSELoss(yPred, yTrue);
        Assert.Equal(1.0, mse.Data.Buffer[0], precision: 6);

        mse.Backward();
        // grad_yPred = 2/N * (yPred - yTrue) = 2/2 * [1, 1] = [1, 1]
        Assert.NotNull(yPred.Grad);
        Assert.Equal(1.0, yPred.Grad.Buffer[0], precision: 6);
        Assert.Equal(1.0, yPred.Grad.Buffer[1], precision: 6);
    }

    [Fact]
    public void Test_EndToEnd_XOR_NeuralNetwork_Training()
    {
        // 4 samples XOR dataset:
        // [0, 0] -> 0
        // [0, 1] -> 1
        // [1, 0] -> 1
        // [1, 1] -> 0
        var xData = new double[]
        {
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0
        };
        var yData = new double[]
        {
            0.0,
            1.0,
            1.0,
            0.0
        };

        var X = new Tensor<double>(xData, new[] { 4, 2 });
        var Y = new Tensor<double>(yData, new[] { 4, 1 });

        // Build 2-layer MLP: 2 -> 8 -> 1
        var l1 = new Linear<double>(2, 8, hasBias: true, seed: 42);
        var l2 = new Linear<double>(8, 1, hasBias: true, seed: 43);

        var model = new Sequential<double>(l1, l2);
        var optimizer = new AdamW<double>(model.Parameters(), lr: 0.1, weightDecay: 0.0);

        double initialLoss = 0.0;
        double finalLoss = 0.0;

        // Train for 250 epochs
        for (int epoch = 0; epoch < 250; epoch++)
        {
            optimizer.ZeroGrad();

            // Forward: L1 -> Tanh -> L2 -> Sigmoid
            var h = l1.Forward(X).Tanh();
            var outPred = l2.Forward(h).Sigmoid();

            var loss = ActivationAndReductionNodes<double>.MSELoss(outPred, Y);

            if (epoch == 0) initialLoss = loss.Data.Buffer[0];
            finalLoss = loss.Data.Buffer[0];

            loss.Backward();
            optimizer.Step();
        }

        // Loss should decrease significantly
        Assert.True(finalLoss < initialLoss);
        Assert.True(finalLoss < 0.05, $"Final loss {finalLoss} is too high for XOR convergence.");

        // Check predictions
        var finalH = l1.Forward(X).Tanh();
        var finalPred = l2.Forward(finalH).Sigmoid();
        var p = finalPred.Data.AsSpan();

        Assert.True(p[0] < 0.2, $"Expected ~0 for [0, 0], got {p[0]}");
        Assert.True(p[1] > 0.8, $"Expected ~1 for [0, 1], got {p[1]}");
        Assert.True(p[2] > 0.8, $"Expected ~1 for [1, 0], got {p[2]}");
        Assert.True(p[3] < 0.2, $"Expected ~0 for [1, 1], got {p[3]}");
    }
}

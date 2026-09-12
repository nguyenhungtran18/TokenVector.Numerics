using System;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Autograd.Nodes;

/// <summary>
/// Backward nodes and factory methods for activation functions, reductions, and loss functions.
/// </summary>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class ActivationAndReductionNodes<T> where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
{
    #region ReLU

    internal sealed class ReluNode : AutogradNode<T>
    {
        public ReluNode(Tensor<T> a)
        {
            Inputs = new[] { a };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad)
            {
                var input = Inputs[0].Data;
                var gradA = new NDArray<T>(input.Shape);
                var inSpan = input.AsReadOnlySpan();
                var gSpan = gradOutput.AsReadOnlySpan();
                var outSpan = gradA.AsSpan();

                for (int i = 0; i < inSpan.Length; i++)
                {
                    outSpan[i] = inSpan[i] > T.Zero ? gSpan[i] : T.Zero;
                }

                Inputs[0].AccumulateGrad(gradA);
            }
        }
    }

    public static Tensor<T> Relu(Tensor<T> a)
    {
        ArgumentNullException.ThrowIfNull(a);
        var inSpan = a.Data.AsReadOnlySpan();
        var outND = new NDArray<T>(a.Shape);
        var outSpan = outND.AsSpan();

        for (int i = 0; i < inSpan.Length; i++)
        {
            outSpan[i] = inSpan[i] > T.Zero ? inSpan[i] : T.Zero;
        }

        ReluNode? node = a.RequiresGrad ? new ReluNode(a) : null;
        var result = new Tensor<T>(outND, a.RequiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    #endregion

    #region Sigmoid

    internal sealed class SigmoidNode : AutogradNode<T>
    {
        public SigmoidNode(Tensor<T> a)
        {
            Inputs = new[] { a };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad && Output != null)
            {
                // d/dx (sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x)) = Output * (1 - Output)
                var outData = Output.Data;
                var gradA = gradOutput * outData * (T.One - outData);
                Inputs[0].AccumulateGrad(gradA);
            }
        }
    }

    public static Tensor<T> Sigmoid(Tensor<T> a)
    {
        ArgumentNullException.ThrowIfNull(a);
        var inSpan = a.Data.AsReadOnlySpan();
        var outND = new NDArray<T>(a.Shape);
        var outSpan = outND.AsSpan();

        for (int i = 0; i < inSpan.Length; i++)
        {
            outSpan[i] = T.One / (T.One + T.Exp(-inSpan[i]));
        }

        SigmoidNode? node = a.RequiresGrad ? new SigmoidNode(a) : null;
        var result = new Tensor<T>(outND, a.RequiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    #endregion

    #region Tanh

    internal sealed class TanhNode : AutogradNode<T>
    {
        public TanhNode(Tensor<T> a)
        {
            Inputs = new[] { a };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad && Output != null)
            {
                // d/dx tanh(x) = 1 - tanh(x)^2 = 1 - Output^2
                var outData = Output.Data;
                var gradA = gradOutput * (T.One - (outData * outData));
                Inputs[0].AccumulateGrad(gradA);
            }
        }
    }

    public static Tensor<T> Tanh(Tensor<T> a)
    {
        ArgumentNullException.ThrowIfNull(a);
        var inSpan = a.Data.AsReadOnlySpan();
        var outND = new NDArray<T>(a.Shape);
        var outSpan = outND.AsSpan();

        for (int i = 0; i < inSpan.Length; i++)
        {
            outSpan[i] = T.Tanh(inSpan[i]);
        }

        TanhNode? node = a.RequiresGrad ? new TanhNode(a) : null;
        var result = new Tensor<T>(outND, a.RequiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    #endregion

    #region GELU

    internal sealed class GeluNode : AutogradNode<T>
    {
        public GeluNode(Tensor<T> a)
        {
            Inputs = new[] { a };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad)
            {
                var inSpan = Inputs[0].Data.AsReadOnlySpan();
                var gSpan = gradOutput.AsReadOnlySpan();
                var gradND = new NDArray<T>(Inputs[0].Shape);
                var gradSpan = gradND.AsSpan();

                T sqrt2OverPi = T.CreateChecked(Math.Sqrt(2.0 / Math.PI));
                T coeff = T.CreateChecked(0.044715);
                T half = T.CreateChecked(0.5);

                for (int i = 0; i < inSpan.Length; i++)
                {
                    T x = inSpan[i];
                    T x3 = x * x * x;
                    T inner = sqrt2OverPi * (x + coeff * x3);
                    T tanhInner = T.Tanh(inner);
                    T dInner = sqrt2OverPi * (T.One + T.CreateChecked(3.0) * coeff * x * x);
                    T dGelu = half * (T.One + tanhInner) + half * x * (T.One - tanhInner * tanhInner) * dInner;

                    gradSpan[i] = gSpan[i] * dGelu;
                }

                Inputs[0].AccumulateGrad(gradND);
            }
        }
    }

    public static Tensor<T> Gelu(Tensor<T> a)
    {
        ArgumentNullException.ThrowIfNull(a);
        var inSpan = a.Data.AsReadOnlySpan();
        var outND = new NDArray<T>(a.Shape);
        var outSpan = outND.AsSpan();

        T sqrt2OverPi = T.CreateChecked(Math.Sqrt(2.0 / Math.PI));
        T coeff = T.CreateChecked(0.044715);
        T half = T.CreateChecked(0.5);

        for (int i = 0; i < inSpan.Length; i++)
        {
            T x = inSpan[i];
            T x3 = x * x * x;
            T inner = sqrt2OverPi * (x + coeff * x3);
            outSpan[i] = half * x * (T.One + T.Tanh(inner));
        }

        GeluNode? node = a.RequiresGrad ? new GeluNode(a) : null;
        var result = new Tensor<T>(outND, a.RequiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    #endregion

    #region Softmax

    internal sealed class SoftmaxNode : AutogradNode<T>
    {
        private readonly int _axis;

        public SoftmaxNode(Tensor<T> a, int axis)
        {
            Inputs = new[] { a };
            _axis = axis;
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad && Output != null)
            {
                // Softmax backward: gradInput = S * (gradOutput - sum(gradOutput * S, axis, keepdims=True))
                var s = Output.Data;
                var dotSum = (gradOutput * s).Sum(_axis, keepdims: true);
                var gradA = s * (gradOutput - dotSum);
                Inputs[0].AccumulateGrad(gradA);
            }
        }
    }

    public static Tensor<T> Softmax(Tensor<T> a, int axis = -1)
    {
        ArgumentNullException.ThrowIfNull(a);
        int normAxis = axis < 0 ? a.Rank + axis : axis;

        // Numerically stable Softmax: S = exp(x - max(x)) / sum(exp(x - max(x)))
        var maxND = a.Data.Max(normAxis, keepdims: true);
        var diff = a.Data - maxND;
        var inSpan = diff.AsReadOnlySpan();
        var expND = new NDArray<T>(diff.Shape);
        var expSpan = expND.AsSpan();

        for (int i = 0; i < inSpan.Length; i++)
        {
            expSpan[i] = T.Exp(inSpan[i]);
        }

        var sumND = expND.Sum(normAxis, keepdims: true);
        var outND = expND / sumND;

        SoftmaxNode? node = a.RequiresGrad ? new SoftmaxNode(a, normAxis) : null;
        var result = new Tensor<T>(outND, a.RequiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    #endregion

    #region Sum & Mean Reductions

    internal sealed class SumNode : AutogradNode<T>
    {
        public SumNode(Tensor<T> a)
        {
            Inputs = new[] { a };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad)
            {
                var gradND = new NDArray<T>(Inputs[0].Shape);
                if (gradOutput.TotalLength == 1)
                {
                    gradND = NDArray<T>.Full(gradOutput.Buffer[0], Inputs[0].Shape);
                }
                else
                {
                    gradND = gradND + gradOutput;
                }
                Inputs[0].AccumulateGrad(gradND);
            }
        }
    }

    public static Tensor<T> Sum(Tensor<T> a, int axis = -1, bool keepdims = false)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (axis == -1 && !keepdims)
        {
            T sumVal = a.Data.Sum();
            var outND = NDArray<T>.Full(sumVal, 1);
            SumNode? node = a.RequiresGrad ? new SumNode(a) : null;
            var result = new Tensor<T>(outND, a.RequiresGrad, node);
            if (node != null) node.Output = result;
            return result;
        }

        int normAxis = axis < 0 ? a.Rank + axis : axis;
        var axisSum = a.Data.Sum(normAxis, keepdims);
        SumNode? axisNode = a.RequiresGrad ? new SumNode(a) : null;
        var axisResult = new Tensor<T>(axisSum, a.RequiresGrad, axisNode);
        if (axisNode != null) axisNode.Output = axisResult;
        return axisResult;
    }

    internal sealed class MeanNode : AutogradNode<T>
    {
        private readonly T _scale;

        public MeanNode(Tensor<T> a, T scale)
        {
            Inputs = new[] { a };
            _scale = scale;
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad)
            {
                var gradND = new NDArray<T>(Inputs[0].Shape);
                if (gradOutput.TotalLength == 1)
                {
                    gradND = NDArray<T>.Full(gradOutput.Buffer[0] / _scale, Inputs[0].Shape);
                }
                else
                {
                    gradND = (gradND + gradOutput) / _scale;
                }
                Inputs[0].AccumulateGrad(gradND);
            }
        }
    }

    public static Tensor<T> Mean(Tensor<T> a, int axis = -1, bool keepdims = false)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (axis == -1 && !keepdims)
        {
            T meanVal = a.Data.Mean();
            var outND = NDArray<T>.Full(meanVal, 1);
            T scale = T.CreateChecked(a.TotalLength);
            MeanNode? node = a.RequiresGrad ? new MeanNode(a, scale) : null;
            var result = new Tensor<T>(outND, a.RequiresGrad, node);
            if (node != null) node.Output = result;
            return result;
        }

        int normAxis = axis < 0 ? a.Rank + axis : axis;
        var axisMean = a.Data.Mean(normAxis, keepdims);
        T dimScale = T.CreateChecked(a.Shape[normAxis]);
        MeanNode? axisNode = a.RequiresGrad ? new MeanNode(a, dimScale) : null;
        var axisResult = new Tensor<T>(axisMean, a.RequiresGrad, axisNode);
        if (axisNode != null) axisNode.Output = axisResult;
        return axisResult;
    }

    #endregion

    #region Losses

    /// <summary>
    /// Computes Mean Squared Error (MSE) loss: L = Mean((yPred - yTrue)^2).
    /// </summary>
    public static Tensor<T> MSELoss(Tensor<T> yPred, Tensor<T> yTrue)
    {
        ArgumentNullException.ThrowIfNull(yPred);
        ArgumentNullException.ThrowIfNull(yTrue);

        var diff = yPred - yTrue;
        var sq = diff * diff;
        return sq.Mean();
    }

    /// <summary>
    /// Computes Cross Entropy Loss for classification: L = -Mean(sum(target * log_softmax(pred), axis=-1)).
    /// </summary>
    public static Tensor<T> CrossEntropyLoss(Tensor<T> logits, Tensor<T> targetOneHot)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(targetOneHot);

        var probs = Softmax(logits, axis: -1);
        T eps = T.CreateChecked(1e-12);
        var logProbs = (probs + eps).Log();
        var loss = -(targetOneHot * logProbs).Sum(axis: -1).Mean();
        return loss;
    }

    #endregion
}

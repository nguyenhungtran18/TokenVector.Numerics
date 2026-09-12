using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using TokenVector.Numerics.Autograd.Nodes;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Autograd;

/// <summary>
/// Differentiable Tensor engine with Dynamic Compute Graph (Reverse-Mode Autograd).
/// </summary>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public sealed class Tensor<T> where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
{
    public NDArray<T> Data { get; set; }
    public NDArray<T>? Grad { get; set; }
    public bool RequiresGrad { get; set; }
    public AutogradNode<T>? GradFn { get; set; }
    public bool IsLeaf { get; set; }

    public int[] Shape => Data.Shape;
    public int[] Strides => Data.Strides;
    public int Rank => Data.Rank;
    public int TotalLength => Data.TotalLength;

    #region Constructors

    public Tensor(NDArray<T> data, bool requiresGrad = false, AutogradNode<T>? gradFn = null)
    {
        ArgumentNullException.ThrowIfNull(data);
        Data = data;
        RequiresGrad = requiresGrad;
        GradFn = gradFn;
        IsLeaf = gradFn == null;
    }

    public Tensor(T[] values, int[] shape, bool requiresGrad = false)
        : this(NDArray<T>.FromArray(values, shape), requiresGrad)
    {
    }

    public Tensor(T scalar, bool requiresGrad = false)
        : this(NDArray<T>.Full(scalar, 1), requiresGrad)
    {
    }

    #endregion

    #region Indexers

    public ref T this[params int[] indices] => ref Data[indices];
    public ref T this[int index] => ref Data[index];

    #endregion

    #region Factory Methods

    public static Tensor<T> FromNDArray(NDArray<T> data, bool requiresGrad = false)
        => new(data, requiresGrad);

    public static Tensor<T> Zeros(params int[] shape)
        => new(NDArray<T>.Zeros(shape), requiresGrad: false);

    public static Tensor<T> Ones(params int[] shape)
        => new(NDArray<T>.Ones(shape), requiresGrad: false);

    public static Tensor<T> Full(T value, params int[] shape)
        => new(NDArray<T>.Full(value, shape), requiresGrad: false);

    public static Tensor<T> Randn(int[] shape, double mean = 0.0, double stdDev = 1.0, int? seed = null, bool requiresGrad = false)
    {
        var rnd = seed.HasValue ? new System.Random(seed.Value) : System.Random.Shared;
        int len = ShapeHelper.ComputeTotalLength(shape);
        var arr = new T[len];

        for (int i = 0; i < len; i += 2)
        {
            double u1 = 1.0 - rnd.NextDouble();
            double u2 = 1.0 - rnd.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            arr[i] = T.CreateChecked(mean + stdDev * randStdNormal);

            if (i + 1 < len)
            {
                double randStdNormal2 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                arr[i + 1] = T.CreateChecked(mean + stdDev * randStdNormal2);
            }
        }

        return new Tensor<T>(NDArray<T>.FromArray(arr, shape), requiresGrad: requiresGrad);
    }

    public static Tensor<T> Uniform(int[] shape, double min = 0.0, double max = 1.0, int? seed = null, bool requiresGrad = false)
    {
        var rnd = seed.HasValue ? new System.Random(seed.Value) : System.Random.Shared;
        int len = ShapeHelper.ComputeTotalLength(shape);
        var arr = new T[len];
        double range = max - min;

        for (int i = 0; i < len; i++)
        {
            arr[i] = T.CreateChecked(min + rnd.NextDouble() * range);
        }

        return new Tensor<T>(NDArray<T>.FromArray(arr, shape), requiresGrad: requiresGrad);
    }

    #endregion

    #region Backward & Gradient Management

    /// <summary>
    /// Accumulates incoming gradient to this tensor's Grad buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AccumulateGrad(NDArray<T> grad)
    {
        if (!RequiresGrad) return;

        if (Grad == null)
        {
            Grad = grad.Clone();
        }
        else
        {
            Grad = Grad + grad;
        }
    }

    /// <summary>
    /// Resets the gradient buffer to null.
    /// </summary>
    public void ZeroGrad()
    {
        Grad = null;
    }

    /// <summary>
    /// Detaches this tensor from the computation graph.
    /// </summary>
    public Tensor<T> Detach()
    {
        return new Tensor<T>(Data.Clone(), requiresGrad: false, gradFn: null);
    }

    /// <summary>
    /// Executes Reverse-Mode Automatic Differentiation via topological sort on the dynamic DAG.
    /// </summary>
    /// <param name="gradient">Optional external gradient (defaults to 1.0 for scalar output).</param>
    public void Backward(Tensor<T>? gradient = null)
    {
        NDArray<T> initGrad;
        if (gradient != null)
        {
            initGrad = gradient.Data;
        }
        else
        {
            if (TotalLength != 1)
            {
                throw new InvalidOperationException($"Implicit backward() can only be called on scalar tensors (TotalLength = 1), but tensor shape is [{string.Join(", ", Shape)}].");
            }
            initGrad = NDArray<T>.Full(T.One, Shape);
        }

        AccumulateGrad(initGrad);

        // Build topological order of nodes in DAG
        var topoOrder = new List<AutogradNode<T>>();
        var visited = new HashSet<AutogradNode<T>>();

        void BuildTopo(AutogradNode<T>? node)
        {
            if (node == null || visited.Contains(node)) return;
            visited.Add(node);
            foreach (var parent in node.GetParentNodes())
            {
                BuildTopo(parent);
            }
            topoOrder.Add(node);
        }

        BuildTopo(GradFn);

        // Execute backward in reverse topological order (from output back to leaf nodes)
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.Output != null && node.Output.Grad != null)
            {
                node.Backward(node.Output.Grad);
            }
        }
    }

    #endregion

    #region Operators

    public static Tensor<T> operator +(Tensor<T> a, Tensor<T> b) => MathAndMatrixNodes<T>.Add(a, b);
    public static Tensor<T> operator +(Tensor<T> a, T b) => MathAndMatrixNodes<T>.AddScalar(a, b);
    public static Tensor<T> operator +(T a, Tensor<T> b) => MathAndMatrixNodes<T>.AddScalar(b, a);

    public static Tensor<T> operator -(Tensor<T> a, Tensor<T> b) => MathAndMatrixNodes<T>.Subtract(a, b);
    public static Tensor<T> operator -(Tensor<T> a, T b) => MathAndMatrixNodes<T>.SubtractScalar(a, b);
    public static Tensor<T> operator -(T a, Tensor<T> b) => MathAndMatrixNodes<T>.SubtractFromScalar(b, a);

    public static Tensor<T> operator *(Tensor<T> a, Tensor<T> b) => MathAndMatrixNodes<T>.Multiply(a, b);
    public static Tensor<T> operator *(Tensor<T> a, T b) => MathAndMatrixNodes<T>.MultiplyScalar(a, b);
    public static Tensor<T> operator *(T a, Tensor<T> b) => MathAndMatrixNodes<T>.MultiplyScalar(b, a);

    public static Tensor<T> operator /(Tensor<T> a, Tensor<T> b) => MathAndMatrixNodes<T>.Divide(a, b);
    public static Tensor<T> operator /(Tensor<T> a, T b) => MathAndMatrixNodes<T>.DivideScalar(a, b);
    public static Tensor<T> operator /(T a, Tensor<T> b) => MathAndMatrixNodes<T>.DivideFromScalar(b, a);

    public static Tensor<T> operator -(Tensor<T> a) => MathAndMatrixNodes<T>.Negate(a);

    #endregion

    #region Tensor Operations & Autograd Methods

    public Tensor<T> MatMul(Tensor<T> other) => MathAndMatrixNodes<T>.MatMul(this, other);

    public Tensor<T> Transpose(int axis0 = 0, int axis1 = 1) => MathAndMatrixNodes<T>.Transpose(this, axis0, axis1);

    public Tensor<T> Reshape(params int[] newShape) => MathAndMatrixNodes<T>.Reshape(this, newShape);

    public Tensor<T> Sum(int axis = -1, bool keepdims = false) => ActivationAndReductionNodes<T>.Sum(this, axis, keepdims);

    public Tensor<T> Mean(int axis = -1, bool keepdims = false) => ActivationAndReductionNodes<T>.Mean(this, axis, keepdims);

    public Tensor<T> Relu() => ActivationAndReductionNodes<T>.Relu(this);

    public Tensor<T> Sigmoid() => ActivationAndReductionNodes<T>.Sigmoid(this);

    public Tensor<T> Tanh() => ActivationAndReductionNodes<T>.Tanh(this);

    public Tensor<T> Gelu() => ActivationAndReductionNodes<T>.Gelu(this);

    public Tensor<T> Softmax(int axis = -1) => ActivationAndReductionNodes<T>.Softmax(this, axis);

    public Tensor<T> Pow(T power) => MathAndMatrixNodes<T>.Pow(this, power);

    public Tensor<T> Exp() => MathAndMatrixNodes<T>.Exp(this);

    public Tensor<T> Log() => MathAndMatrixNodes<T>.Log(this);

    #endregion

    public override string ToString()
    {
        return $"Tensor(shape=[{string.Join(", ", Shape)}], requires_grad={RequiresGrad}, dtype={typeof(T).Name})\n{Data}";
    }
}

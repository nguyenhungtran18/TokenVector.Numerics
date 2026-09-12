using System;
using System.Collections.Generic;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Autograd.NN;

/// <summary>
/// Base class for all Neural Network layers and composite modules.
/// </summary>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public abstract class Module<T> where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
{
    /// <summary>
    /// Executes the forward pass of this module.
    /// </summary>
    public abstract Tensor<T> Forward(Tensor<T> input);

    /// <summary>
    /// Collects all learnable parameter tensors belonging to this module and its submodules.
    /// </summary>
    public abstract IEnumerable<Tensor<T>> Parameters();

    /// <summary>
    /// Resets all parameter gradients to null.
    /// </summary>
    public void ZeroGrad()
    {
        foreach (var p in Parameters())
        {
            p.ZeroGrad();
        }
    }
}

/// <summary>
/// Fully connected linear (dense) layer: Y = X * W + b.
/// </summary>
public sealed class Linear<T> : Module<T> where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
{
    public Tensor<T> Weight { get; }
    public Tensor<T>? Bias { get; }
    public int InFeatures { get; }
    public int OutFeatures { get; }

    public Linear(int inFeatures, int outFeatures, bool hasBias = true, int? seed = null)
    {
        InFeatures = inFeatures;
        OutFeatures = outFeatures;

        // Kaiming / He uniform initialization: bound = 1 / sqrt(inFeatures)
        double bound = 1.0 / Math.Sqrt(inFeatures);
        Weight = Tensor<T>.Uniform(new[] { inFeatures, outFeatures }, -bound, bound, seed, requiresGrad: true);

        if (hasBias)
        {
            Bias = Tensor<T>.Uniform(new[] { 1, outFeatures }, -bound, bound, seed.HasValue ? seed.Value + 1 : null, requiresGrad: true);
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        var output = input.MatMul(Weight);
        if (Bias != null)
        {
            output = output + Bias;
        }
        return output;
    }

    public override IEnumerable<Tensor<T>> Parameters()
    {
        yield return Weight;
        if (Bias != null)
        {
            yield return Bias;
        }
    }
}

/// <summary>
/// Sequential container module that cascades multiple modules in order.
/// </summary>
public sealed class Sequential<T> : Module<T> where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
{
    private readonly List<Module<T>> _modules = new();

    public IReadOnlyList<Module<T>> Modules => _modules;

    public Sequential(params Module<T>[] modules)
    {
        if (modules != null)
        {
            _modules.AddRange(modules);
        }
    }

    public void Add(Module<T> module)
    {
        ArgumentNullException.ThrowIfNull(module);
        _modules.Add(module);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        var current = input;
        foreach (var m in _modules)
        {
            current = m.Forward(current);
        }
        return current;
    }

    public override IEnumerable<Tensor<T>> Parameters()
    {
        foreach (var m in _modules)
        {
            foreach (var p in m.Parameters())
            {
                yield return p;
            }
        }
    }
}

/// <summary>
/// Root Mean Square Layer Normalization (RMSNorm) for LLMs &amp; Transformers.
/// </summary>
public sealed class RMSNorm<T> : Module<T> where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
{
    public Tensor<T> Weight { get; }
    public double Epsilon { get; }

    public RMSNorm(int dim, double eps = 1e-6)
    {
        Epsilon = eps;
        Weight = Tensor<T>.Ones(1, dim);
        Weight.RequiresGrad = true;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        // rms = sqrt(mean(x^2, axis=-1, keepdims=true) + eps)
        var x2 = input * input;
        var meanX2 = x2.Mean(axis: -1, keepdims: true);
        var rms = (meanX2 + T.CreateChecked(Epsilon)).Pow(T.CreateChecked(0.5));
        var norm = input / rms;
        return norm * Weight;
    }

    public override IEnumerable<Tensor<T>> Parameters()
    {
        yield return Weight;
    }
}

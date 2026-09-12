using System;
using System.Collections.Generic;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Autograd.Optim;

/// <summary>
/// Base class for all tensor gradient optimization algorithms.
/// </summary>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public abstract class Optimizer<T> where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
{
    protected readonly List<Tensor<T>> _parameters;

    public IReadOnlyList<Tensor<T>> Parameters => _parameters;

    protected Optimizer(IEnumerable<Tensor<T>> parameters)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        _parameters = new List<Tensor<T>>(parameters);
    }

    /// <summary>
    /// Resets all gradients of registered parameters to null.
    /// </summary>
    public void ZeroGrad()
    {
        foreach (var p in _parameters)
        {
            p.ZeroGrad();
        }
    }

    /// <summary>
    /// Performs a single optimization parameter update step.
    /// </summary>
    public abstract void Step();
}

/// <summary>
/// Stochastic Gradient Descent (SGD) optimizer with optional Momentum and Weight Decay.
/// </summary>
public sealed class SGD<T> : Optimizer<T> where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
{
    private readonly T _lr;
    private readonly T _momentum;
    private readonly T _weightDecay;
    private readonly Dictionary<Tensor<T>, NDArray<T>> _velocities = new();

    public SGD(IEnumerable<Tensor<T>> parameters, double lr = 0.01, double momentum = 0.0, double weightDecay = 0.0)
        : base(parameters)
    {
        _lr = T.CreateChecked(lr);
        _momentum = T.CreateChecked(momentum);
        _weightDecay = T.CreateChecked(weightDecay);
    }

    public override void Step()
    {
        foreach (var p in _parameters)
        {
            if (p.Grad == null) continue;

            var dSpan = p.Data.AsSpan();
            var gSpan = p.Grad.AsReadOnlySpan();

            if (_momentum > T.Zero)
            {
                if (!_velocities.TryGetValue(p, out var v))
                {
                    v = new NDArray<T>(p.Shape);
                    _velocities[p] = v;
                }

                var vSpan = v.AsSpan();
                for (int i = 0; i < dSpan.Length; i++)
                {
                    T grad = gSpan[i];
                    if (_weightDecay > T.Zero)
                    {
                        grad += _weightDecay * dSpan[i];
                    }

                    vSpan[i] = _momentum * vSpan[i] + grad;
                    dSpan[i] -= _lr * vSpan[i];
                }
            }
            else
            {
                for (int i = 0; i < dSpan.Length; i++)
                {
                    T grad = gSpan[i];
                    if (_weightDecay > T.Zero)
                    {
                        grad += _weightDecay * dSpan[i];
                    }

                    dSpan[i] -= _lr * grad;
                }
            }
        }
    }
}

/// <summary>
/// AdamW optimizer with decoupled weight decay (Loshchilov &amp; Hutter).
/// </summary>
public sealed class AdamW<T> : Optimizer<T> where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
{
    private readonly T _lr;
    private readonly T _beta1;
    private readonly T _beta2;
    private readonly T _eps;
    private readonly T _weightDecay;
    private int _stepCount;

    private readonly Dictionary<Tensor<T>, NDArray<T>> _expAvg = new();
    private readonly Dictionary<Tensor<T>, NDArray<T>> _expAvgSq = new();

    public AdamW(IEnumerable<Tensor<T>> parameters, double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8, double weightDecay = 0.01)
        : base(parameters)
    {
        _lr = T.CreateChecked(lr);
        _beta1 = T.CreateChecked(beta1);
        _beta2 = T.CreateChecked(beta2);
        _eps = T.CreateChecked(eps);
        _weightDecay = T.CreateChecked(weightDecay);
        _stepCount = 0;
    }

    public override void Step()
    {
        _stepCount++;
        T beta1T = _beta1;
        T beta2T = _beta2;
        T oneMinusBeta1 = T.One - beta1T;
        T oneMinusBeta2 = T.One - beta2T;

        // Bias correction factors
        T biasCorrection1 = T.One - T.CreateChecked(Math.Pow(double.CreateChecked(beta1T), _stepCount));
        T biasCorrection2 = T.One - T.CreateChecked(Math.Pow(double.CreateChecked(beta2T), _stepCount));

        foreach (var p in _parameters)
        {
            if (p.Grad == null) continue;

            if (!_expAvg.TryGetValue(p, out var m))
            {
                m = new NDArray<T>(p.Shape);
                _expAvg[p] = m;
            }

            if (!_expAvgSq.TryGetValue(p, out var v))
            {
                v = new NDArray<T>(p.Shape);
                _expAvgSq[p] = v;
            }

            var dSpan = p.Data.AsSpan();
            var gSpan = p.Grad.AsReadOnlySpan();
            var mSpan = m.AsSpan();
            var vSpan = v.AsSpan();

            for (int i = 0; i < dSpan.Length; i++)
            {
                // 1. Decoupled weight decay
                if (_weightDecay > T.Zero)
                {
                    dSpan[i] -= _lr * _weightDecay * dSpan[i];
                }

                T grad = gSpan[i];

                // 2. Update biased 1st moment estimate
                mSpan[i] = beta1T * mSpan[i] + oneMinusBeta1 * grad;

                // 3. Update biased 2nd raw moment estimate
                vSpan[i] = beta2T * vSpan[i] + oneMinusBeta2 * (grad * grad);

                // 4. Compute bias-corrected moments
                T mHat = mSpan[i] / biasCorrection1;
                T vHat = vSpan[i] / biasCorrection2;

                // 5. Update parameter
                T step = _lr * mHat / (T.Sqrt(vHat) + _eps);
                dSpan[i] -= step;
            }
        }
    }
}

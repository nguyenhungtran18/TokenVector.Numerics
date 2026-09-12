using System;
using System.Collections.Generic;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Autograd;

/// <summary>
/// Base abstract class for all backward execution nodes in the Dynamic Compute Graph (DAG).
/// </summary>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public abstract class AutogradNode<T> where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
{
    /// <summary>
    /// The input tensors that fed into this operation.
    /// </summary>
    public Tensor<T>[] Inputs { get; protected set; } = Array.Empty<Tensor<T>>();

    /// <summary>
    /// The output tensor produced by this operation.
    /// </summary>
    public Tensor<T>? Output { get; set; }

    /// <summary>
    /// Executes the vector-jacobian product and propagates gradients backward to parent tensors.
    /// </summary>
    /// <param name="gradOutput">Incoming gradient from child nodes.</param>
    public abstract void Backward(NDArray<T> gradOutput);

    /// <summary>
    /// Retrieves all parent AutogradNodes feeding into the inputs.
    /// </summary>
    public IEnumerable<AutogradNode<T>> GetParentNodes()
    {
        foreach (var input in Inputs)
        {
            if (input?.GradFn != null)
            {
                yield return input.GradFn;
            }
        }
    }

    /// <summary>
    /// Reduces and unbroadcasts an incoming gradient tensor to match the original tensor shape.
    /// </summary>
    /// <param name="grad">Incoming gradient NDArray.</param>
    /// <param name="targetShape">Original target shape before forward broadcast.</param>
    /// <returns>Unbroadcasted gradient matching targetShape.</returns>
    public static NDArray<T> Unbroadcast(NDArray<T> grad, int[] targetShape)
    {
        ArgumentNullException.ThrowIfNull(grad);
        ArgumentNullException.ThrowIfNull(targetShape);

        if (grad.Shape.AsSpan().SequenceEqual(targetShape))
        {
            return grad.Clone();
        }

        if (targetShape.Length == 0 || (targetShape.Length == 1 && targetShape[0] == 1 && grad.TotalLength > 1))
        {
            T sumVal = grad.Sum();
            return NDArray<T>.Full(sumVal, targetShape);
        }

        var currentGrad = grad;

        // 1. Sum over extra leading dimensions if grad has higher rank than target
        while (currentGrad.Rank > targetShape.Length)
        {
            currentGrad = currentGrad.Sum(axis: 0, keepdims: false);
        }

        // 2. Sum along dimensions where targetShape has size 1 but grad has size > 1
        for (int i = 0; i < targetShape.Length; i++)
        {
            if (targetShape[i] == 1 && currentGrad.Shape[i] > 1)
            {
                currentGrad = currentGrad.Sum(axis: i, keepdims: true);
            }
        }

        if (!currentGrad.Shape.AsSpan().SequenceEqual(targetShape))
        {
            currentGrad = currentGrad.Reshape(targetShape);
        }

        return currentGrad.IsContiguous ? currentGrad : currentGrad.Contiguous();
    }
}

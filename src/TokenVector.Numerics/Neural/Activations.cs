using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Neural;

/// <summary>
/// Numerically stable activation functions with forward and backward gradient (VJP) kernels for Autograd.
/// </summary>
public static class Activations
{
    #region ReLU & LeakyReLU

    public static NDArray<T> ReLU<T>(NDArray<T> x) where T : unmanaged, IFloatingPoint<T>
    {
        var result = new NDArray<T>(x.Shape);
        var src = x.IsContiguous ? x.AsReadOnlySpan() : x.Contiguous().AsReadOnlySpan();
        var dst = result.AsSpan();

        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = src[i] > T.Zero ? src[i] : T.Zero;
        }
        return result;
    }

    public static NDArray<T> ReLUBackward<T>(NDArray<T> gradOutput, NDArray<T> x) where T : unmanaged, IFloatingPoint<T>
    {
        var gradInput = new NDArray<T>(x.Shape);
        var gOut = gradOutput.IsContiguous ? gradOutput.AsReadOnlySpan() : gradOutput.Contiguous().AsReadOnlySpan();
        var src = x.IsContiguous ? x.AsReadOnlySpan() : x.Contiguous().AsReadOnlySpan();
        var gIn = gradInput.AsSpan();

        for (int i = 0; i < src.Length; i++)
        {
            gIn[i] = src[i] > T.Zero ? gOut[i] : T.Zero;
        }
        return gradInput;
    }

    public static NDArray<T> LeakyReLU<T>(NDArray<T> x, T negativeSlope) where T : unmanaged, IFloatingPoint<T>
    {
        var result = new NDArray<T>(x.Shape);
        var src = x.IsContiguous ? x.AsReadOnlySpan() : x.Contiguous().AsReadOnlySpan();
        var dst = result.AsSpan();

        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = src[i] > T.Zero ? src[i] : negativeSlope * src[i];
        }
        return result;
    }

    public static NDArray<T> LeakyReLUBackward<T>(NDArray<T> gradOutput, NDArray<T> x, T negativeSlope) where T : unmanaged, IFloatingPoint<T>
    {
        var gradInput = new NDArray<T>(x.Shape);
        var gOut = gradOutput.IsContiguous ? gradOutput.AsReadOnlySpan() : gradOutput.Contiguous().AsReadOnlySpan();
        var src = x.IsContiguous ? x.AsReadOnlySpan() : x.Contiguous().AsReadOnlySpan();
        var gIn = gradInput.AsSpan();

        for (int i = 0; i < src.Length; i++)
        {
            gIn[i] = src[i] > T.Zero ? gOut[i] : negativeSlope * gOut[i];
        }
        return gradInput;
    }

    #endregion

    #region Sigmoid & Tanh

    public static NDArray<T> Sigmoid<T>(NDArray<T> x) where T : unmanaged, IFloatingPoint<T>, IExponentialFunctions<T>
    {
        var result = new NDArray<T>(x.Shape);
        var src = x.IsContiguous ? x.AsReadOnlySpan() : x.Contiguous().AsReadOnlySpan();
        var dst = result.AsSpan();

        for (int i = 0; i < src.Length; i++)
        {
            // Numerically stable sigmoid
            if (src[i] >= T.Zero)
            {
                T z = T.Exp(-src[i]);
                dst[i] = T.One / (T.One + z);
            }
            else
            {
                T z = T.Exp(src[i]);
                dst[i] = z / (T.One + z);
            }
        }
        return result;
    }

    public static NDArray<T> SigmoidBackward<T>(NDArray<T> gradOutput, NDArray<T> outputSigmoid) where T : unmanaged, IFloatingPoint<T>
    {
        var gradInput = new NDArray<T>(outputSigmoid.Shape);
        var gOut = gradOutput.IsContiguous ? gradOutput.AsReadOnlySpan() : gradOutput.Contiguous().AsReadOnlySpan();
        var sig = outputSigmoid.IsContiguous ? outputSigmoid.AsReadOnlySpan() : outputSigmoid.Contiguous().AsReadOnlySpan();
        var gIn = gradInput.AsSpan();

        for (int i = 0; i < sig.Length; i++)
        {
            gIn[i] = gOut[i] * sig[i] * (T.One - sig[i]);
        }
        return gradInput;
    }

    public static NDArray<T> Tanh<T>(NDArray<T> x) where T : unmanaged, IFloatingPoint<T>, IHyperbolicFunctions<T>
    {
        var result = new NDArray<T>(x.Shape);
        var src = x.IsContiguous ? x.AsReadOnlySpan() : x.Contiguous().AsReadOnlySpan();
        var dst = result.AsSpan();

        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = T.Tanh(src[i]);
        }
        return result;
    }

    public static NDArray<T> TanhBackward<T>(NDArray<T> gradOutput, NDArray<T> outputTanh) where T : unmanaged, IFloatingPoint<T>
    {
        var gradInput = new NDArray<T>(outputTanh.Shape);
        var gOut = gradOutput.IsContiguous ? gradOutput.AsReadOnlySpan() : gradOutput.Contiguous().AsReadOnlySpan();
        var th = outputTanh.IsContiguous ? outputTanh.AsReadOnlySpan() : outputTanh.Contiguous().AsReadOnlySpan();
        var gIn = gradInput.AsSpan();

        for (int i = 0; i < th.Length; i++)
        {
            gIn[i] = gOut[i] * (T.One - th[i] * th[i]);
        }
        return gradInput;
    }

    #endregion

    #region GELU (Gaussian Error Linear Unit)

    /// <summary>
    /// Computes GELU activation using high-precision tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    /// </summary>
    public static NDArray<T> GELU<T>(NDArray<T> x) where T : unmanaged, IFloatingPoint<T>, IHyperbolicFunctions<T>
    {
        var result = new NDArray<T>(x.Shape);
        var src = x.IsContiguous ? x.AsReadOnlySpan() : x.Contiguous().AsReadOnlySpan();
        var dst = result.AsSpan();

        T sqrt2OverPi = T.CreateChecked(Math.Sqrt(2.0 / Math.PI));
        T c0044715 = T.CreateChecked(0.044715);
        T half = T.CreateChecked(0.5);

        for (int i = 0; i < src.Length; i++)
        {
            T val = src[i];
            T cube = val * val * val;
            T inner = sqrt2OverPi * (val + c0044715 * cube);
            dst[i] = half * val * (T.One + T.Tanh(inner));
        }
        return result;
    }

    #endregion

    #region Softmax & Log-Sum-Exp Trick

    /// <summary>
    /// Computes Softmax along the specified axis using Log-Sum-Exp trick to eliminate floating-point overflow.
    /// </summary>
    public static NDArray<T> Softmax<T>(NDArray<T> x, int axis = -1) where T : unmanaged, IFloatingPoint<T>, IExponentialFunctions<T>
    {
        if (axis < 0) axis += x.Rank;

        // 1. Compute Max along axis for numerical shift (subtract max)
        var maxVal = x.Max(axis, keepdims: true);
        var shifted = x - maxVal;

        // 2. Compute Exp of shifted values
        var expResult = new NDArray<T>(shifted.Shape);
        var sSpan = shifted.IsContiguous ? shifted.AsReadOnlySpan() : shifted.Contiguous().AsReadOnlySpan();
        var eSpan = expResult.AsSpan();
        for (int i = 0; i < sSpan.Length; i++)
        {
            eSpan[i] = T.Exp(sSpan[i]);
        }

        // 3. Normalize by Sum(Exp) along axis
        var sumExp = expResult.Sum(axis, keepdims: true);
        return expResult / sumExp;
    }

    /// <summary>
    /// Computes Softmax gradient (Vector-Jacobian Product): gradInput = softmax * (gradOutput - sum(gradOutput * softmax, axis)).
    /// </summary>
    public static NDArray<T> SoftmaxBackward<T>(NDArray<T> gradOutput, NDArray<T> outputSoftmax, int axis = -1) where T : unmanaged, IFloatingPoint<T>
    {
        if (axis < 0) axis += outputSoftmax.Rank;
        var dot = (gradOutput * outputSoftmax).Sum(axis, keepdims: true);
        return outputSoftmax * (gradOutput - dot);
    }

    #endregion
}

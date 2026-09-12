using System;
using System.Numerics;
using System.Threading.Tasks;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Neural;

/// <summary>
/// High-performance Convolutional Neural Network engine based on im2col, MatMul, and Pooling operators.
/// </summary>
public static class ConvEngine
{
    /// <summary>
    /// Transforms image tensor [N, C, H, W] into column matrix for fast MatMul convolution.
    /// Output shape: [C * Kh * Kw, N * OutH * OutW].
    /// </summary>
    public static NDArray<T> Im2Col<T>(
        NDArray<T> input,
        int kernelH, int kernelW,
        int strideH = 1, int strideW = 1,
        int padH = 0, int padW = 0) where T : unmanaged, INumber<T>
    {
        if (input.Rank != 4) throw new ArgumentException("Im2Col requires 4D tensor [N, C, H, W].");

        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int inH = input.Shape[2];
        int inW = input.Shape[3];

        int outH = (inH + 2 * padH - kernelH) / strideH + 1;
        int outW = (inW + 2 * padW - kernelW) / strideW + 1;

        int numCols = batchSize * outH * outW;
        int colRows = channels * kernelH * kernelW;

        var colMatrix = new NDArray<T>(colRows, numCols);
        var inContig = input.Contiguous();

        Parallel.For(0, channels, c =>
        {
            for (int kh = 0; kh < kernelH; kh++)
            {
                for (int kw = 0; kw < kernelW; kw++)
                {
                    int rowIdx = c * kernelH * kernelW + kh * kernelW + kw;

                    for (int b = 0; b < batchSize; b++)
                    {
                        for (int oh = 0; oh < outH; oh++)
                        {
                            int ih = oh * strideH - padH + kh;

                            for (int ow = 0; ow < outW; ow++)
                            {
                                int iw = ow * strideW - padW + kw;
                                int colIdx = b * (outH * outW) + oh * outW + ow;

                                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                {
                                    colMatrix[rowIdx, colIdx] = inContig[b, c, ih, iw];
                                }
                                else
                                {
                                    colMatrix[rowIdx, colIdx] = T.Zero;
                                }
                            }
                        }
                    }
                }
            }
        });

        return colMatrix;
    }

    /// <summary>
    /// Performs 2D Convolution Forward pass: Output = Weights * Im2Col(Input) + Bias.
    /// Input: [N, C_in, H, W], Weights: [C_out, C_in, Kh, Kw], Bias: [C_out] (optional).
    /// Output: [N, C_out, OutH, OutW].
    /// </summary>
    public static NDArray<T> Conv2D<T>(
        NDArray<T> input,
        NDArray<T> weights,
        NDArray<T>? bias = null,
        int strideH = 1, int strideW = 1,
        int padH = 0, int padW = 0) where T : unmanaged, IFloatingPoint<T>
    {
        if (input.Rank != 4 || weights.Rank != 4)
        {
            throw new ArgumentException("Conv2D requires 4D input [N, C, H, W] and 4D weights [C_out, C_in, Kh, Kw].");
        }

        int batchSize = input.Shape[0];
        int inC = input.Shape[1];
        int inH = input.Shape[2];
        int inW = input.Shape[3];

        int outC = weights.Shape[0];
        int kernelH = weights.Shape[2];
        int kernelW = weights.Shape[3];

        if (weights.Shape[1] != inC)
        {
            throw new ArgumentException($"Channel mismatch: input has {inC} channels, weights expect {weights.Shape[1]}.");
        }

        int outH = (inH + 2 * padH - kernelH) / strideH + 1;
        int outW = (inW + 2 * padW - kernelW) / strideW + 1;

        // 1. Flatten weights to [C_out, C_in * Kh * Kw]
        var flatWeights = weights.Reshape(outC, inC * kernelH * kernelW);

        // 2. Im2Col: [C_in * Kh * Kw, N * OutH * OutW]
        var colMatrix = Im2Col(input, kernelH, kernelW, strideH, strideW, padH, padW);

        // 3. MatMul: [C_out, N * OutH * OutW]
        var outMatrix = MatrixMultiplication.MatMul(flatWeights, colMatrix);

        // 4. Reshape to [C_out, N, OutH, OutW] then permute to [N, C_out, OutH, OutW]
        var shaped = outMatrix.Reshape(outC, batchSize, outH, outW);
        var permuted = shaped.Permute(1, 0, 2, 3).Contiguous();

        // 5. Add bias if present
        if (bias != null)
        {
            var reshapedBias = bias.Reshape(1, outC, 1, 1);
            return permuted + reshapedBias;
        }

        return permuted;
    }

    /// <summary>
    /// Performs 2D Max Pooling on tensor [N, C, H, W].
    /// </summary>
    public static (NDArray<T> Output, NDArray<int> ArgMaxIndices) MaxPool2D<T>(
        NDArray<T> input,
        int poolH, int poolW,
        int strideH = 2, int strideW = 2) where T : unmanaged, IFloatingPoint<T>
    {
        if (input.Rank != 4) throw new ArgumentException("MaxPool2D requires 4D tensor [N, C, H, W].");

        int n = input.Shape[0];
        int c = input.Shape[1];
        int h = input.Shape[2];
        int w = input.Shape[3];

        int outH = (h - poolH) / strideH + 1;
        int outW = (w - poolW) / strideW + 1;

        var output = new NDArray<T>(n, c, outH, outW);
        var indices = new NDArray<int>(n, c, outH, outW);
        var inContig = input.Contiguous();

        Parallel.For(0, n, b =>
        {
            for (int ch = 0; ch < c; ch++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    int startH = oh * strideH;
                    for (int ow = 0; ow < outW; ow++)
                    {
                        int startW = ow * strideW;

                        T maxVal = inContig[b, ch, startH, startW];
                        int maxIdx = startH * w + startW;

                        for (int kh = 0; kh < poolH; kh++)
                        {
                            for (int kw = 0; kw < poolW; kw++)
                            {
                                int curH = startH + kh;
                                int curW = startW + kw;
                                T val = inContig[b, ch, curH, curW];
                                if (val > maxVal)
                                {
                                    maxVal = val;
                                    maxIdx = curH * w + curW;
                                }
                            }
                        }

                        output[b, ch, oh, ow] = maxVal;
                        indices[b, ch, oh, ow] = maxIdx;
                    }
                }
            }
        });

        return (output, indices);
    }
}

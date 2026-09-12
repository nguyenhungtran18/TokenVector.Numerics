using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Engine;

namespace TokenVector.Numerics.LinAlg;

/// <summary>
/// High-performance matrix multiplication engine featuring Cache Tiling (L1/L2 Blocking) and Batch MatMul.
/// </summary>
public static unsafe class MatrixMultiplication
{
    private const int BlockSize = 32;

    /// <summary>
    /// Multiplies two matrices (2D or Batched N-D).
    /// </summary>
    public static NDArray<T> MatMul<T>(NDArray<T> a, NDArray<T> b) where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        if (a.Rank < 2 || b.Rank < 2)
        {
            throw new ArgumentException("MatMul requires tensors with at least 2 dimensions.");
        }

        int m = a.Shape[^2];
        int kA = a.Shape[^1];
        int kB = b.Shape[^2];
        int n = b.Shape[^1];

        if (kA != kB)
        {
            throw new ArgumentException($"Inner matrix dimensions must agree: {kA} != {kB}.");
        }

        // 2D Matrix multiplication
        if (a.Rank == 2 && b.Rank == 2)
        {
            return MatMul2D(a, b, m, kA, n);
        }

        // Batch Matrix multiplication
        return BatchMatMul(a, b, m, kA, n);
    }

    private static NDArray<T> MatMul2D<T>(NDArray<T> a, NDArray<T> b, int m, int k, int n) where T : unmanaged, INumber<T>
    {
        var result = new NDArray<T>(m, n);
        var aContig = a.Contiguous();
        var bContig = b.Contiguous();

        // Optimized Tiled MatMul for float
        if (typeof(T) == typeof(float))
        {
            float* pA = (float*)aContig.GetUnsafePointer();
            float* pB = (float*)bContig.GetUnsafePointer();
            float* pC = (float*)result.GetUnsafePointer();

            Parallel.For(0, (m + BlockSize - 1) / BlockSize, bi =>
            {
                int iStart = bi * BlockSize;
                int iEnd = Math.Min(iStart + BlockSize, m);

                for (int bk = 0; bk < k; bk += BlockSize)
                {
                    int kEnd = Math.Min(bk + BlockSize, k);

                    for (int bj = 0; bj < n; bj += BlockSize)
                    {
                        int jEnd = Math.Min(bj + BlockSize, n);

                        for (int i = iStart; i < iEnd; i++)
                        {
                            float* aRow = pA + i * k;
                            float* cRow = pC + i * n;

                            for (int kk = bk; kk < kEnd; kk++)
                            {
                                float aVal = aRow[kk];
                                float* bRow = pB + kk * n;

                                int j = bj;
                                if (Vector256.IsHardwareAccelerated)
                                {
                                    var va = Vector256.Create(aVal);
                                    int vCount = Vector256<float>.Count;
                                    for (; j <= jEnd - vCount; j += vCount)
                                    {
                                        var vb = Vector256.Load(bRow + j);
                                        var vc = Vector256.Load(cRow + j);
                                        Vector256.Store(vc + va * vb, cRow + j);
                                    }
                                }

                                for (; j < jEnd; j++)
                                {
                                    cRow[j] += aVal * bRow[j];
                                }
                            }
                        }
                    }
                }
            });

            return result;
        }

        // Generic fallback with tiling
        Parallel.For(0, (m + BlockSize - 1) / BlockSize, bi =>
        {
            int iStart = bi * BlockSize;
            int iEnd = Math.Min(iStart + BlockSize, m);

            for (int bk = 0; bk < k; bk += BlockSize)
            {
                int kEnd = Math.Min(bk + BlockSize, k);

                for (int bj = 0; bj < n; bj += BlockSize)
                {
                    int jEnd = Math.Min(bj + BlockSize, n);

                    for (int i = iStart; i < iEnd; i++)
                    {
                        for (int kk = bk; kk < kEnd; kk++)
                        {
                            T aVal = aContig[i, kk];
                            for (int j = bj; j < jEnd; j++)
                            {
                                result[i, j] += aVal * bContig[kk, j];
                            }
                        }
                    }
                }
            }
        });

        return result;
    }

    private static NDArray<T> BatchMatMul<T>(NDArray<T> a, NDArray<T> b, int m, int k, int n) where T : unmanaged, INumber<T>
    {
        // Batch dimensions are all dimensions except the last two
        var aBatchShape = a.Shape[..^2];
        var bBatchShape = b.Shape[..^2];
        var outBatchShape = BroadcastEngine.BroadcastShapes(aBatchShape, bBatchShape);

        int[] outShape = new int[outBatchShape.Length + 2];
        Array.Copy(outBatchShape, outShape, outBatchShape.Length);
        outShape[^2] = m;
        outShape[^1] = n;

        var result = new NDArray<T>(outShape);
        int totalBatches = ShapeHelper.ComputeTotalLength(outBatchShape);

        int aBatchStride = a.Shape[^2] * a.Shape[^1];
        int bBatchStride = b.Shape[^2] * b.Shape[^1];
        int outBatchStride = m * n;

        var aContig = a.Contiguous();
        var bContig = b.Contiguous();

        Parallel.For(0, totalBatches, batchIdx =>
        {
            Span<int> batchCoords = stackalloc int[outBatchShape.Length];
            ShapeHelper.GetMultiIndex(batchIdx, outBatchShape, batchCoords);

            // Compute slice offsets for a and b
            int aOffset = 0;
            int bOffset = 0;
            for (int d = 0; d < outBatchShape.Length; d++)
            {
                int aDimIdx = (d < aBatchShape.Length) ? (aBatchShape[d] == 1 ? 0 : batchCoords[d]) : 0;
                int bDimIdx = (d < bBatchShape.Length) ? (bBatchShape[d] == 1 ? 0 : batchCoords[d]) : 0;

                int aDimStride = (d < aBatchShape.Length) ? aContig.Strides[d] : 0;
                int bDimStride = (d < bBatchShape.Length) ? bContig.Strides[d] : 0;

                aOffset += aDimIdx * aDimStride;
                bOffset += bDimIdx * bDimStride;
            }

            int outOffset = batchIdx * outBatchStride;

            // Perform single 2D matrix multiplication for this batch slice
            for (int i = 0; i < m; i++)
            {
                for (int kk = 0; kk < k; kk++)
                {
                    T aVal = aContig.Buffer[aOffset + i * k + kk];
                    for (int j = 0; j < n; j++)
                    {
                        T bVal = bContig.Buffer[bOffset + kk * n + j];
                        result.Buffer[outOffset + i * n + j] += aVal * bVal;
                    }
                }
            }
        });

        return result;
    }
}

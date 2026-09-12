// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Numerics;
using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Ops;

/// <summary>
/// Provides grid generation and structured matrix extraction operations (np.meshgrid, np.diag, np.triu, np.tril).
/// </summary>
public static class GridOps
{
    /// <summary>
    /// Generates coordinate matrices from coordinate vectors (np.meshgrid).
    /// Defaults to 'xy' indexing for 2D vectors: (len(y), len(x)).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T>[] Meshgrid<T>(params NDArray<T>[] vectors) where T : unmanaged, INumber<T>
    {
        if (vectors == null || vectors.Length == 0)
            throw new ArgumentException("At least one vector must be provided.", nameof(vectors));

        for (int i = 0; i < vectors.Length; i++)
        {
            if (vectors[i].Rank != 1)
                throw new ArgumentException($"Vector at index {i} must be 1D.");
        }

        int n = vectors.Length;
        int[] outShape = new int[n];

        if (n == 2)
        {
            // Standard 'xy' Cartesian indexing: shape (ny, nx)
            outShape[0] = vectors[1].Shape[0];
            outShape[1] = vectors[0].Shape[0];

            var gridX = new NDArray<T>(outShape);
            var gridY = new NDArray<T>(outShape);

            var xContig = vectors[0].Contiguous();
            var yContig = vectors[1].Contiguous();

            int ny = outShape[0];
            int nx = outShape[1];

            Parallel.For(0, ny, r =>
            {
                T yr = yContig[r];
                for (int c = 0; c < nx; c++)
                {
                    gridX[r, c] = xContig[c];
                    gridY[r, c] = yr;
                }
            });

            return [gridX, gridY];
        }
        else
        {
            // 'ij' matrix indexing for ND
            for (int i = 0; i < n; i++)
            {
                outShape[i] = vectors[i].Shape[0];
            }

            var grids = new NDArray<T>[n];
            for (int i = 0; i < n; i++)
            {
                grids[i] = new NDArray<T>(outShape);
            }

            int totalElements = 1;
            for (int i = 0; i < n; i++) totalElements *= outShape[i];

            int[] strides = ShapeHelper.ComputeRowMajorStrides(outShape);

            Parallel.For(0, totalElements, flatIdx =>
            {
                int rem = flatIdx;
                int[] coords = new int[n];
                for (int d = 0; d < n; d++)
                {
                    coords[d] = rem / strides[d];
                    rem %= strides[d];
                }

                for (int d = 0; d < n; d++)
                {
                    T val = vectors[d][coords[d]];
                    grids[d][flatIdx] = val;
                }
            });

            return grids;
        }
    }

    /// <summary>
    /// Extracts a diagonal or constructs a diagonal array (np.diag).
    /// If input is 1D vector: returns 2D square matrix with vector on k-th diagonal.
    /// If input is 2D matrix: returns 1D vector containing elements of k-th diagonal.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Diag<T>(NDArray<T> tensor, int k = 0) where T : unmanaged, INumber<T>
    {
        if (tensor.Rank == 1)
        {
            int len = tensor.Shape[0];
            int n = len + Math.Abs(k);
            var mat = new NDArray<T>(n, n);
            var contig = tensor.Contiguous();

            for (int i = 0; i < len; i++)
            {
                int r = k >= 0 ? i : i - k;
                int c = k >= 0 ? i + k : i;
                mat[r, c] = contig[i];
            }
            return mat;
        }
        else if (tensor.Rank == 2)
        {
            return Diagonal(tensor, k);
        }
        else
        {
            throw new ArgumentException("Input tensor must be 1D or 2D for Diag.");
        }
    }

    /// <summary>
    /// Returns specified diagonals of a 2D matrix (np.diagonal).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Diagonal<T>(NDArray<T> mat, int offset = 0) where T : unmanaged, INumber<T>
    {
        if (mat.Rank != 2)
            throw new ArgumentException("Matrix must be 2D for Diagonal extraction.");

        int rows = mat.Shape[0];
        int cols = mat.Shape[1];

        int startR = offset >= 0 ? 0 : -offset;
        int startC = offset >= 0 ? offset : 0;

        int diagLen = Math.Max(0, Math.Min(rows - startR, cols - startC));
        var diag = new NDArray<T>(diagLen);

        for (int i = 0; i < diagLen; i++)
        {
            diag[i] = mat[startR + i, startC + i];
        }

        return diag;
    }

    /// <summary>
    /// Upper triangle of an array (np.triu).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Triu<T>(NDArray<T> mat, int k = 0) where T : unmanaged, INumber<T>
    {
        if (mat.Rank < 2)
            throw new ArgumentException("Tensor must be at least 2D for Triu.");

        var res = mat.Clone();
        int rows = mat.Shape[^2];
        int cols = mat.Shape[^1];
        int batchSize = mat.TotalLength / (rows * cols);

        Parallel.For(0, batchSize, b =>
        {
            int batchOffset = b * rows * cols;
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    if (c < r + k)
                    {
                        res[batchOffset + r * cols + c] = T.Zero;
                    }
                }
            }
        });

        return res;
    }

    /// <summary>
    /// Lower triangle of an array (np.tril).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static NDArray<T> Tril<T>(NDArray<T> mat, int k = 0) where T : unmanaged, INumber<T>
    {
        if (mat.Rank < 2)
            throw new ArgumentException("Tensor must be at least 2D for Tril.");

        var res = mat.Clone();
        int rows = mat.Shape[^2];
        int cols = mat.Shape[^1];
        int batchSize = mat.TotalLength / (rows * cols);

        Parallel.For(0, batchSize, b =>
        {
            int batchOffset = b * rows * cols;
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    if (c > r + k)
                    {
                        res[batchOffset + r * cols + c] = T.Zero;
                    }
                }
            }
        });

        return res;
    }
}

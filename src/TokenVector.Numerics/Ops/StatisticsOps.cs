using System;
using System.Numerics;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Ops;

/// <summary>
/// Statistical and probabilistic functions: Variance, Standard Deviation, Median, Percentile, Covariance, and Correlation Coefficients.
/// </summary>
public static class StatisticsOps
{
    /// <summary>
    /// Computes variance along the specified axis (equivalent to np.var).
    /// </summary>
    public static NDArray<T> Var<T>(this NDArray<T> tensor, int? axis = null, bool keepdims = false, int ddof = 0)
        where T : unmanaged, IFloatingPoint<T>
    {
        if (axis == null)
        {
            T mean = tensor.Mean();
            T sumSq = T.Zero;
            foreach (var val in tensor)
            {
                T diff = val - mean;
                sumSq += diff * diff;
            }
            int divisor = Math.Max(1, tensor.TotalLength - ddof);
            T varVal = sumSq / T.CreateChecked(divisor);
            return NDArray<T>.FromArray(new[] { varVal }, keepdims ? new[] { 1 } : Array.Empty<int>());
        }

        int ax = axis.Value;
        var meanTensor = tensor.Mean(ax, keepdims: true);
        var diffTensor = tensor - meanTensor;
        var sqDiff = diffTensor * diffTensor;

        int dimSize = tensor.Shape[ax < 0 ? ax + tensor.Rank : ax];
        int denom = Math.Max(1, dimSize - ddof);

        var sumSqAxis = sqDiff.Sum(ax, keepdims);
        return sumSqAxis / T.CreateChecked(denom);
    }

    /// <summary>
    /// Computes standard deviation along the specified axis (equivalent to np.std).
    /// </summary>
    public static NDArray<T> Std<T>(this NDArray<T> tensor, int? axis = null, bool keepdims = false, int ddof = 0)
        where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        var variance = Var(tensor, axis, keepdims, ddof);
        var result = new NDArray<T>(variance.Shape);
        var src = variance.IsContiguous ? variance.AsReadOnlySpan() : variance.Contiguous().AsReadOnlySpan();
        var dst = result.AsSpan();

        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = T.Sqrt(src[i]);
        }
        return result;
    }

    /// <summary>
    /// Computes median along the specified axis (equivalent to np.median).
    /// </summary>
    public static NDArray<T> Median<T>(this NDArray<T> tensor, int? axis = null, bool keepdims = false)
        where T : unmanaged, IFloatingPoint<T>
    {
        return Percentile(tensor, 50.0, axis, keepdims);
    }

    /// <summary>
    /// Computes the q-th percentile of data along the specified axis (q in [0, 100]).
    /// </summary>
    public static NDArray<T> Percentile<T>(this NDArray<T> tensor, double q, int? axis = null, bool keepdims = false)
        where T : unmanaged, IFloatingPoint<T>
    {
        if (q < 0.0 || q > 100.0) throw new ArgumentOutOfRangeException(nameof(q), "Percentile q must be in [0, 100].");

        if (axis == null)
        {
            var flat = tensor.Contiguous();
            T[] sorted = flat.ToArray();
            Array.Sort(sorted);

            int n = sorted.Length;
            double index = (q / 100.0) * (n - 1);
            int low = (int)Math.Floor(index);
            int high = (int)Math.Ceiling(index);
            double frac = index - low;

            T val = sorted[low] + T.CreateChecked(frac) * (sorted[high] - sorted[low]);
            return NDArray<T>.FromArray(new[] { val }, keepdims ? new[] { 1 } : Array.Empty<int>());
        }

        int ax = axis.Value;
        if (ax < 0) ax += tensor.Rank;
        int axisSize = tensor.Shape[ax];

        var sortedTensor = tensor.Sort(ax);
        int[] outShape = GetReducedShape(tensor.Shape, ax, keepdims);
        var result = new NDArray<T>(outShape);

        double targetIdx = (q / 100.0) * (axisSize - 1);
        int iLow = (int)Math.Floor(targetIdx);
        int iHigh = (int)Math.Ceiling(targetIdx);
        T fracT = T.CreateChecked(targetIdx - iLow);

        int outTotal = result.TotalLength;
        int[] indexLookupShape = keepdims ? GetReducedShape(tensor.Shape, ax, keepdims: false) : outShape;

        Span<int> coords = stackalloc int[indexLookupShape.Length];
        for (int i = 0; i < outTotal; i++)
        {
            ShapeHelper.GetMultiIndex(i, indexLookupShape, coords);

            int[] lowCoords = ReconstructCoords(coords, ax, iLow);
            int[] highCoords = ReconstructCoords(coords, ax, iHigh);

            T lowVal = sortedTensor[lowCoords];
            T highVal = sortedTensor[highCoords];

            result.AsSpan()[i] = lowVal + fracT * (highVal - lowVal);
        }

        return result;
    }

    /// <summary>
    /// Computes sample covariance matrix between variables (equivalent to np.cov).
    /// Input: [NumVariables, NumObservations].
    /// </summary>
    public static NDArray<T> Cov<T>(NDArray<T> x) where T : unmanaged, IFloatingPoint<T>
    {
        if (x.Rank != 2) throw new ArgumentException("Cov requires 2D matrix [Variables, Observations].");

        int nVars = x.Shape[0];
        int nObs = x.Shape[1];
        if (nObs < 2) throw new ArgumentException("Cov requires at least 2 observations.");

        // Subtract mean along observations axis
        var mean = x.Mean(1, keepdims: true);
        var centered = x - mean;

        // Cov = (centered * centered^T) / (N - 1)
        var cov = LinAlg.MatrixMultiplication.MatMul(centered, centered.MatrixTranspose());
        T scale = T.One / T.CreateChecked(nObs - 1);

        return cov * scale;
    }

    /// <summary>
    /// Computes Pearson correlation coefficients matrix (equivalent to np.corrcoef).
    /// </summary>
    public static NDArray<T> CorrCoef<T>(NDArray<T> x) where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        var cov = Cov(x);
        int n = cov.Shape[0];
        var corr = new NDArray<T>(n, n);

        T[] stds = new T[n];
        for (int i = 0; i < n; i++)
        {
            stds[i] = T.Sqrt(cov[i, i]);
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T denom = stds[i] * stds[j];
                corr[i, j] = denom > T.Zero ? cov[i, j] / denom : T.Zero;
            }
        }

        return corr;
    }

    private static int[] GetReducedShape(ReadOnlySpan<int> shape, int normalizedAxis, bool keepdims)
    {
        if (keepdims)
        {
            int[] res = shape.ToArray();
            res[normalizedAxis] = 1;
            return res;
        }

        if (shape.Length == 1) return Array.Empty<int>();

        int[] newShape = new int[shape.Length - 1];
        int idx = 0;
        for (int i = 0; i < shape.Length; i++)
        {
            if (i != normalizedAxis) newShape[idx++] = shape[i];
        }
        return newShape;
    }

    private static int[] ReconstructCoords(ReadOnlySpan<int> reducedCoords, int axis, int axisVal)
    {
        int rank = reducedCoords.Length + 1;
        int[] coords = new int[rank];
        int rIdx = 0;
        for (int i = 0; i < rank; i++)
        {
            if (i == axis) coords[i] = axisVal;
            else coords[i] = reducedCoords[rIdx++];
        }
        return coords;
    }
}

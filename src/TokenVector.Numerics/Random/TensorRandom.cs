using System;
using System.Numerics;
using System.Threading;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Random;

/// <summary>
/// Pseudo-random number generator engine for tensors supporting Uniform, Gaussian Normal, Discrete Integer, Choice, and Shuffle.
/// </summary>
public static class TensorRandom
{
    private static readonly AsyncLocal<System.Random?> _threadRandom = new();
    private static int _globalSeed = Environment.TickCount;

    private static System.Random GetRng()
    {
        return _threadRandom.Value ??= new System.Random(Interlocked.Increment(ref _globalSeed));
    }

    /// <summary>
    /// Sets the global seed for reproducible random tensor generation.
    /// </summary>
    public static void ManualSeed(int seed)
    {
        _globalSeed = seed;
        _threadRandom.Value = new System.Random(seed);
    }

    /// <summary>
    /// Generates tensor with random values drawn uniformly from [0, 1).
    /// </summary>
    public static NDArray<T> Rand<T>(params int[] shape) where T : unmanaged, IFloatingPoint<T>
    {
        var result = new NDArray<T>(shape);
        var rng = GetRng();
        var span = result.AsSpan();

        for (int i = 0; i < span.Length; i++)
        {
            span[i] = T.CreateChecked(rng.NextDouble());
        }
        return result;
    }

    /// <summary>
    /// Generates tensor with random values drawn from Standard Normal distribution N(mean, std^2) using the Box-Muller transform.
    /// </summary>
    public static NDArray<T> Randn<T>(T mean, T std, params int[] shape)
        where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>
    {
        var result = new NDArray<T>(shape);
        var rng = GetRng();
        var span = result.AsSpan();
        int n = span.Length;

        T twoPi = T.CreateChecked(2.0 * Math.PI);
        T minusTwo = T.CreateChecked(-2.0);

        for (int i = 0; i < n; i += 2)
        {
            double u1 = Math.Max(1e-15, rng.NextDouble());
            double u2 = rng.NextDouble();

            T tU1 = T.CreateChecked(u1);
            T tU2 = T.CreateChecked(u2);

            T radius = T.Sqrt(minusTwo * T.Log(tU1));
            T theta = twoPi * tU2;

            T z0 = radius * T.Cos(theta);
            T z1 = radius * T.Sin(theta);

            span[i] = mean + z0 * std;
            if (i + 1 < n)
            {
                span[i + 1] = mean + z1 * std;
            }
        }

        return result;
    }

    /// <summary>
    /// Generates standard normal tensor N(0, 1).
    /// </summary>
    public static NDArray<T> Randn<T>(params int[] shape)
        where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>
    {
        return Randn(T.Zero, T.One, shape);
    }

    /// <summary>
    /// Generates tensor with random integer values drawn uniformly from [low, high).
    /// </summary>
    public static NDArray<int> Randint(int low, int high, params int[] shape)
    {
        if (low >= high) throw new ArgumentException("low must be strictly less than high.");
        var result = new NDArray<int>(shape);
        var rng = GetRng();
        var span = result.AsSpan();

        for (int i = 0; i < span.Length; i++)
        {
            span[i] = rng.Next(low, high);
        }
        return result;
    }

    /// <summary>
    /// Randomly permutes a tensor in-place along axis 0 using the Fisher-Yates algorithm.
    /// </summary>
    public static void Shuffle<T>(NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        int n = tensor.Shape[0];
        var rng = GetRng();

        if (tensor.Rank == 1)
        {
            var span = tensor.AsSpan();
            for (int i = n - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (span[i], span[j]) = (span[j], span[i]);
            }
        }
        else
        {
            // Swap slices along axis 0
            for (int i = n - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                if (i == j) continue;

                var sliceI = tensor.Slice((i, i + 1, 1)).Contiguous();
                var sliceJ = tensor.Slice((j, j + 1, 1)).Contiguous();

                // Swap
                var temp = sliceI.Clone();
                sliceI.AsSpan().CopyTo(sliceJ.AsSpan());
                temp.AsSpan().CopyTo(sliceI.AsSpan());
            }
        }
    }

    /// <summary>
    /// Generates a random sample from a given 1D tensor.
    /// </summary>
    public static NDArray<T> Choice<T>(NDArray<T> a, int size, bool replace = true) where T : unmanaged, INumber<T>
    {
        if (a.Rank != 1) throw new ArgumentException("Choice requires a 1D tensor.");
        if (!replace && size > a.TotalLength)
        {
            throw new ArgumentException("Cannot take a larger sample than population when replace = false.");
        }

        var rng = GetRng();
        var result = new NDArray<T>(size);

        if (replace)
        {
            for (int i = 0; i < size; i++)
            {
                int idx = rng.Next(a.TotalLength);
                result[i] = a[idx];
            }
        }
        else
        {
            int[] indices = new int[a.TotalLength];
            for (int i = 0; i < indices.Length; i++) indices[i] = i;

            // Partial Fisher-Yates shuffle
            for (int i = 0; i < size; i++)
            {
                int j = rng.Next(i, indices.Length);
                (indices[i], indices[j]) = (indices[j], indices[i]);
                result[i] = a[indices[i]];
            }
        }

        return result;
    }
}

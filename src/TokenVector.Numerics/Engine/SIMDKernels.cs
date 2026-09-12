using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace TokenVector.Numerics.Engine;

/// <summary>
/// Hardware-accelerated SIMD kernel engine with AVX2/FMA fast paths and generic Vector&lt;T&gt; fallbacks.
/// </summary>
public static unsafe class SIMDKernels
{
    private const int ParallelThreshold = 65536;

    #region Contiguous Elementwise Binary Ops

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddContiguous<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> destination) where T : unmanaged, INumber<T>
    {
        int length = destination.Length;
        if (typeof(T) == typeof(float) && Vector256.IsHardwareAccelerated)
        {
            fixed (float* pA = &Unsafe.As<T, float>(ref MemoryMarshal.GetReference(a)))
            fixed (float* pB = &Unsafe.As<T, float>(ref MemoryMarshal.GetReference(b)))
            fixed (float* pDest = &Unsafe.As<T, float>(ref MemoryMarshal.GetReference(destination)))
            {
                int vCount = Vector256<float>.Count;
                int i = 0;
                for (; i <= length - vCount; i += vCount)
                {
                    var va = Vector256.Load(pA + i);
                    var vb = Vector256.Load(pB + i);
                    Vector256.Store(va + vb, pDest + i);
                }
                for (; i < length; i++) pDest[i] = pA[i] + pB[i];
                return;
            }
        }

        if (typeof(T) == typeof(double) && Vector256.IsHardwareAccelerated)
        {
            fixed (double* pA = &Unsafe.As<T, double>(ref MemoryMarshal.GetReference(a)))
            fixed (double* pB = &Unsafe.As<T, double>(ref MemoryMarshal.GetReference(b)))
            fixed (double* pDest = &Unsafe.As<T, double>(ref MemoryMarshal.GetReference(destination)))
            {
                int vCount = Vector256<double>.Count;
                int i = 0;
                for (; i <= length - vCount; i += vCount)
                {
                    var va = Vector256.Load(pA + i);
                    var vb = Vector256.Load(pB + i);
                    Vector256.Store(va + vb, pDest + i);
                }
                for (; i < length; i++) pDest[i] = pA[i] + pB[i];
                return;
            }
        }

        // Generic Vector<T> fallback
        int vecCount = Vector<T>.Count;
        int idx = 0;
        for (; idx <= length - vecCount; idx += vecCount)
        {
            var va = new Vector<T>(a.Slice(idx, vecCount));
            var vb = new Vector<T>(b.Slice(idx, vecCount));
            (va + vb).CopyTo(destination.Slice(idx, vecCount));
        }
        for (; idx < length; idx++)
        {
            destination[idx] = a[idx] + b[idx];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SubtractContiguous<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> destination) where T : unmanaged, INumber<T>
    {
        int length = destination.Length;
        if (typeof(T) == typeof(float) && Vector256.IsHardwareAccelerated)
        {
            fixed (float* pA = &Unsafe.As<T, float>(ref MemoryMarshal.GetReference(a)))
            fixed (float* pB = &Unsafe.As<T, float>(ref MemoryMarshal.GetReference(b)))
            fixed (float* pDest = &Unsafe.As<T, float>(ref MemoryMarshal.GetReference(destination)))
            {
                int vCount = Vector256<float>.Count;
                int i = 0;
                for (; i <= length - vCount; i += vCount)
                {
                    var va = Vector256.Load(pA + i);
                    var vb = Vector256.Load(pB + i);
                    Vector256.Store(va - vb, pDest + i);
                }
                for (; i < length; i++) pDest[i] = pA[i] - pB[i];
                return;
            }
        }

        if (typeof(T) == typeof(double) && Vector256.IsHardwareAccelerated)
        {
            fixed (double* pA = &Unsafe.As<T, double>(ref MemoryMarshal.GetReference(a)))
            fixed (double* pB = &Unsafe.As<T, double>(ref MemoryMarshal.GetReference(b)))
            fixed (double* pDest = &Unsafe.As<T, double>(ref MemoryMarshal.GetReference(destination)))
            {
                int vCount = Vector256<double>.Count;
                int i = 0;
                for (; i <= length - vCount; i += vCount)
                {
                    var va = Vector256.Load(pA + i);
                    var vb = Vector256.Load(pB + i);
                    Vector256.Store(va - vb, pDest + i);
                }
                for (; i < length; i++) pDest[i] = pA[i] - pB[i];
                return;
            }
        }

        int vecCount = Vector<T>.Count;
        int idx = 0;
        for (; idx <= length - vecCount; idx += vecCount)
        {
            var va = new Vector<T>(a.Slice(idx, vecCount));
            var vb = new Vector<T>(b.Slice(idx, vecCount));
            (va - vb).CopyTo(destination.Slice(idx, vecCount));
        }
        for (; idx < length; idx++)
        {
            destination[idx] = a[idx] - b[idx];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MultiplyContiguous<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> destination) where T : unmanaged, INumber<T>
    {
        int length = destination.Length;
        if (typeof(T) == typeof(float) && Vector256.IsHardwareAccelerated)
        {
            fixed (float* pA = &Unsafe.As<T, float>(ref MemoryMarshal.GetReference(a)))
            fixed (float* pB = &Unsafe.As<T, float>(ref MemoryMarshal.GetReference(b)))
            fixed (float* pDest = &Unsafe.As<T, float>(ref MemoryMarshal.GetReference(destination)))
            {
                int vCount = Vector256<float>.Count;
                int i = 0;
                for (; i <= length - vCount; i += vCount)
                {
                    var va = Vector256.Load(pA + i);
                    var vb = Vector256.Load(pB + i);
                    Vector256.Store(va * vb, pDest + i);
                }
                for (; i < length; i++) pDest[i] = pA[i] * pB[i];
                return;
            }
        }

        if (typeof(T) == typeof(double) && Vector256.IsHardwareAccelerated)
        {
            fixed (double* pA = &Unsafe.As<T, double>(ref MemoryMarshal.GetReference(a)))
            fixed (double* pB = &Unsafe.As<T, double>(ref MemoryMarshal.GetReference(b)))
            fixed (double* pDest = &Unsafe.As<T, double>(ref MemoryMarshal.GetReference(destination)))
            {
                int vCount = Vector256<double>.Count;
                int i = 0;
                for (; i <= length - vCount; i += vCount)
                {
                    var va = Vector256.Load(pA + i);
                    var vb = Vector256.Load(pB + i);
                    Vector256.Store(va * vb, pDest + i);
                }
                for (; i < length; i++) pDest[i] = pA[i] * pB[i];
                return;
            }
        }

        int vecCount = Vector<T>.Count;
        int idx = 0;
        for (; idx <= length - vecCount; idx += vecCount)
        {
            var va = new Vector<T>(a.Slice(idx, vecCount));
            var vb = new Vector<T>(b.Slice(idx, vecCount));
            (va * vb).CopyTo(destination.Slice(idx, vecCount));
        }
        for (; idx < length; idx++)
        {
            destination[idx] = a[idx] * b[idx];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DivideContiguous<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> destination) where T : unmanaged, INumber<T>
    {
        int length = destination.Length;
        if (typeof(T) == typeof(float) && Vector256.IsHardwareAccelerated)
        {
            fixed (float* pA = &Unsafe.As<T, float>(ref MemoryMarshal.GetReference(a)))
            fixed (float* pB = &Unsafe.As<T, float>(ref MemoryMarshal.GetReference(b)))
            fixed (float* pDest = &Unsafe.As<T, float>(ref MemoryMarshal.GetReference(destination)))
            {
                int vCount = Vector256<float>.Count;
                int i = 0;
                for (; i <= length - vCount; i += vCount)
                {
                    var va = Vector256.Load(pA + i);
                    var vb = Vector256.Load(pB + i);
                    Vector256.Store(va / vb, pDest + i);
                }
                for (; i < length; i++) pDest[i] = pA[i] / pB[i];
                return;
            }
        }

        if (typeof(T) == typeof(double) && Vector256.IsHardwareAccelerated)
        {
            fixed (double* pA = &Unsafe.As<T, double>(ref MemoryMarshal.GetReference(a)))
            fixed (double* pB = &Unsafe.As<T, double>(ref MemoryMarshal.GetReference(b)))
            fixed (double* pDest = &Unsafe.As<T, double>(ref MemoryMarshal.GetReference(destination)))
            {
                int vCount = Vector256<double>.Count;
                int i = 0;
                for (; i <= length - vCount; i += vCount)
                {
                    var va = Vector256.Load(pA + i);
                    var vb = Vector256.Load(pB + i);
                    Vector256.Store(va / vb, pDest + i);
                }
                for (; i < length; i++) pDest[i] = pA[i] / pB[i];
                return;
            }
        }

        int vecCount = Vector<T>.Count;
        int idx = 0;
        for (; idx <= length - vecCount; idx += vecCount)
        {
            var va = new Vector<T>(a.Slice(idx, vecCount));
            var vb = new Vector<T>(b.Slice(idx, vecCount));
            (va / vb).CopyTo(destination.Slice(idx, vecCount));
        }
        for (; idx < length; idx++)
        {
            destination[idx] = a[idx] / b[idx];
        }
    }

    #endregion

    #region Reductions

    public static T SumContiguous<T>(ReadOnlySpan<T> source) where T : unmanaged, INumber<T>
    {
        int length = source.Length;
        if (length == 0) return T.Zero;

        if (typeof(T) == typeof(float) && Vector256.IsHardwareAccelerated)
        {
            fixed (float* pSrc = &Unsafe.As<T, float>(ref MemoryMarshal.GetReference(source)))
            {
                var vAcc = Vector256<float>.Zero;
                int vCount = Vector256<float>.Count;
                int i = 0;
                for (; i <= length - vCount; i += vCount)
                {
                    vAcc += Vector256.Load(pSrc + i);
                }
                float sum = Vector256.Sum(vAcc);
                for (; i < length; i++) sum += pSrc[i];
                return Unsafe.As<float, T>(ref sum);
            }
        }

        if (typeof(T) == typeof(double) && Vector256.IsHardwareAccelerated)
        {
            fixed (double* pSrc = &Unsafe.As<T, double>(ref MemoryMarshal.GetReference(source)))
            {
                var vAcc = Vector256<double>.Zero;
                int vCount = Vector256<double>.Count;
                int i = 0;
                for (; i <= length - vCount; i += vCount)
                {
                    vAcc += Vector256.Load(pSrc + i);
                }
                double sum = Vector256.Sum(vAcc);
                for (; i < length; i++) sum += pSrc[i];
                return Unsafe.As<double, T>(ref sum);
            }
        }

        int vecCount = Vector<T>.Count;
        var acc = Vector<T>.Zero;
        int idx = 0;
        for (; idx <= length - vecCount; idx += vecCount)
        {
            acc += new Vector<T>(source.Slice(idx, vecCount));
        }
        T total = T.Zero;
        for (int k = 0; k < vecCount; k++) total += acc[k];
        for (; idx < length; idx++) total += source[idx];
        return total;
    }

    #endregion
}

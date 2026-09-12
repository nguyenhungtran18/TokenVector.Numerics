using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace TokenVector.Numerics.Core;

/// <summary>
/// Unified hybrid tensor storage supporting Managed GC arrays, Unmanaged NativeMemory pointers, and Zero-Copy Views.
/// </summary>
/// <typeparam name="T">Unmanaged numeric type.</typeparam>
public sealed unsafe class TensorBuffer<T> : IDisposable where T : unmanaged
{
    private readonly T[]? _managedArray;
    private readonly T* _nativePtr;
    private readonly int _elementCount;
    private readonly bool _isNative;
    private readonly TensorBuffer<T>? _parentBuffer;
    private int _refCount;
    private int _isDisposed;

    /// <summary>
    /// Gets whether this buffer is backed by native unmanaged memory.
    /// </summary>
    public bool IsNative => _isNative;

    /// <summary>
    /// Total element capacity of this buffer.
    /// </summary>
    public int Length => _elementCount;

    /// <summary>
    /// Gets whether this buffer has been disposed.
    /// </summary>
    public bool IsDisposed => Volatile.Read(ref _isDisposed) != 0;

    /// <summary>
    /// Creates a managed GC-backed tensor buffer.
    /// </summary>
    public TensorBuffer(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        _elementCount = length;
        _managedArray = GC.AllocateArray<T>(length, pinned: false);
        _nativePtr = null;
        _isNative = false;
        _parentBuffer = null;
        _refCount = 1;
    }

    /// <summary>
    /// Wraps an existing managed array.
    /// </summary>
    public TensorBuffer(T[] array)
    {
        ArgumentNullException.ThrowIfNull(array);
        _elementCount = array.Length;
        _managedArray = array;
        _nativePtr = null;
        _isNative = false;
        _parentBuffer = null;
        _refCount = 1;
    }

    /// <summary>
    /// Allocates an unmanaged native memory buffer.
    /// </summary>
    public static TensorBuffer<T> AllocateNative(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        nuint byteCount = (nuint)length * (nuint)sizeof(T);
        void* ptr = NativeMemory.AllocZeroed(byteCount);
        return new TensorBuffer<T>((T*)ptr, length, isOwner: true);
    }

    private TensorBuffer(T* nativePtr, int length, bool isOwner)
    {
        _elementCount = length;
        _nativePtr = nativePtr;
        _managedArray = null;
        _isNative = true;
        _parentBuffer = null;
        _refCount = isOwner ? 1 : 0;
    }

    /// <summary>
    /// Creates a view buffer sharing the same underlying storage of a parent buffer.
    /// </summary>
    public TensorBuffer(TensorBuffer<T> parent, int offset, int length)
    {
        ArgumentNullException.ThrowIfNull(parent);
        if (offset < 0 || length < 0 || offset + length > parent.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), "Invalid view offset or length for parent buffer.");
        }

        _parentBuffer = parent;
        _parentBuffer.AddRef();
        _elementCount = length;
        _isNative = parent._isNative;

        if (_isNative)
        {
            _nativePtr = parent._nativePtr + offset;
            _managedArray = null;
        }
        else
        {
            _managedArray = parent._managedArray;
            _nativePtr = null;
        }
        _refCount = 1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddRef()
    {
        Interlocked.Increment(ref _refCount);
    }

    /// <summary>
    /// Gets a Span over the buffer memory with a specific offset and length.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan(int offset = 0, int length = -1)
    {
        if (IsDisposed) throw new ObjectDisposedException(nameof(TensorBuffer<T>));
        int actualLength = length < 0 ? _elementCount - offset : length;
        if (offset < 0 || actualLength < 0 || offset + actualLength > _elementCount)
        {
            throw new ArgumentOutOfRangeException(nameof(offset));
        }

        if (_isNative)
        {
            return new Span<T>(_nativePtr + offset, actualLength);
        }
        return new Span<T>(_managedArray!, offset, actualLength);
    }

    /// <summary>
    /// Gets a ReadOnlySpan over the buffer memory.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> AsReadOnlySpan(int offset = 0, int length = -1)
    {
        return AsSpan(offset, length);
    }

    /// <summary>
    /// Direct element indexer with aggressive inlining.
    /// </summary>
    public ref T this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if (IsDisposed) throw new ObjectDisposedException(nameof(TensorBuffer<T>));
            if ((uint)index >= (uint)_elementCount)
            {
                throw new IndexOutOfRangeException($"Index {index} is out of buffer range [0, {_elementCount}).");
            }

            if (_isNative)
            {
                return ref *(_nativePtr + index);
            }
            return ref _managedArray![index];
        }
    }

    /// <summary>
    /// Gets direct raw pointer to underlying memory (if pinned or native).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T* GetUnsafePointer()
    {
        if (IsDisposed) throw new ObjectDisposedException(nameof(TensorBuffer<T>));
        if (_isNative) return _nativePtr;
        return (T*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_managedArray!));
    }

    /// <summary>
    /// Copies contents to a managed array.
    /// </summary>
    public T[] ToArray(int offset = 0, int length = -1)
    {
        int actualLength = length < 0 ? _elementCount - offset : length;
        T[] result = new T[actualLength];
        AsReadOnlySpan(offset, actualLength).CopyTo(result);
        return result;
    }

    /// <summary>
    /// Creates a detached full clone of this buffer.
    /// </summary>
    public TensorBuffer<T> Clone()
    {
        var clone = _isNative ? AllocateNative(_elementCount) : new TensorBuffer<T>(_elementCount);
        AsReadOnlySpan().CopyTo(clone.AsSpan());
        return clone;
    }

    ~TensorBuffer()
    {
        Dispose(false);
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (Interlocked.Exchange(ref _isDisposed, 1) != 0) return;

        if (Interlocked.Decrement(ref _refCount) <= 0)
        {
            if (_parentBuffer != null)
            {
                _parentBuffer.Dispose();
            }
            else if (_isNative && _nativePtr != null)
            {
                NativeMemory.Free(_nativePtr);
            }
        }
    }
}

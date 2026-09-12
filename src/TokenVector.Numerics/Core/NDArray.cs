using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using TokenVector.Numerics.Engine;

namespace TokenVector.Numerics.Core;

/// <summary>
/// N-Dimensional Array (Tensor) high-performance engine for TokenVector.
/// </summary>
/// <typeparam name="T">Unmanaged numeric type.</typeparam>
public sealed unsafe partial class NDArray<T> : IDisposable, IEnumerable<T> where T : unmanaged, INumber<T>
{
    private readonly TensorBuffer<T> _buffer;
    private readonly int[] _shape;
    private readonly int[] _strides;
    private readonly int _offset;
    private readonly int _totalLength;
    private readonly bool _isContiguous;

    public TensorBuffer<T> Buffer => _buffer;
    public int[] Shape => _shape;
    public int[] Strides => _strides;
    public int Offset => _offset;
    public int TotalLength => _totalLength;
    public int Rank => _shape.Length;
    public bool IsContiguous => _isContiguous;
    public TypeCode DType => Type.GetTypeCode(typeof(T));

    #region Constructors

    /// <summary>
    /// Creates a newly allocated tensor with given shape (C-contiguous).
    /// </summary>
    public NDArray(params int[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        _shape = (int[])shape.Clone();
        _totalLength = ShapeHelper.ComputeTotalLength(_shape);
        _strides = ShapeHelper.ComputeRowMajorStrides(_shape);
        _offset = 0;
        _buffer = new TensorBuffer<T>(_totalLength);
        _isContiguous = true;
    }

    /// <summary>
    /// Internal constructor for views and transformations.
    /// </summary>
    public NDArray(TensorBuffer<T> buffer, int[] shape, int[] strides, int offset)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentNullException.ThrowIfNull(strides);

        _buffer = buffer;
        _shape = (int[])shape.Clone();
        _strides = (int[])strides.Clone();
        _offset = offset;
        _totalLength = ShapeHelper.ComputeTotalLength(_shape);
        _isContiguous = ShapeHelper.IsContiguous(_shape, _strides);
    }

    #endregion

    #region Indexers

    /// <summary>
    /// N-Dimensional indexer with aggressive inlining.
    /// </summary>
    public ref T this[params int[] indices]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if (indices.Length != _shape.Length)
            {
                throw new ArgumentException($"Index rank {indices.Length} does not match tensor rank {_shape.Length}.");
            }
            int flatIndex = _offset;
            for (int i = 0; i < indices.Length; i++)
            {
                int idx = indices[i];
                if ((uint)idx >= (uint)_shape[i])
                {
                    throw new IndexOutOfRangeException($"Index {idx} out of range for axis {i} with size {_shape[i]}.");
                }
                flatIndex += idx * _strides[i];
            }
            return ref _buffer[flatIndex];
        }
    }

    /// <summary>
    /// Fast 1D indexer.
    /// </summary>
    public ref T this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if (_shape.Length == 1)
            {
                if ((uint)index >= (uint)_shape[0]) throw new IndexOutOfRangeException();
                return ref _buffer[_offset + index * _strides[0]];
            }
            if (_isContiguous)
            {
                if ((uint)index >= (uint)_totalLength) throw new IndexOutOfRangeException();
                return ref _buffer[_offset + index];
            }
            // Multi-dimensional non-contiguous fallback
            Span<int> coords = stackalloc int[_shape.Length];
            ShapeHelper.GetMultiIndex(index, _shape, coords);
            return ref this[coords.ToArray()];
        }
    }

    /// <summary>
    /// Fast 2D indexer.
    /// </summary>
    public ref T this[int i0, int i1]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if (_shape.Length != 2 || (uint)i0 >= (uint)_shape[0] || (uint)i1 >= (uint)_shape[1])
            {
                throw new IndexOutOfRangeException();
            }
            return ref _buffer[_offset + i0 * _strides[0] + i1 * _strides[1]];
        }
    }

    /// <summary>
    /// Fast 3D indexer.
    /// </summary>
    public ref T this[int i0, int i1, int i2]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if (_shape.Length != 3 || (uint)i0 >= (uint)_shape[0] || (uint)i1 >= (uint)_shape[1] || (uint)i2 >= (uint)_shape[2])
            {
                throw new IndexOutOfRangeException();
            }
            return ref _buffer[_offset + i0 * _strides[0] + i1 * _strides[1] + i2 * _strides[2]];
        }
    }

    /// <summary>
    /// Fast 4D indexer.
    /// </summary>
    public ref T this[int i0, int i1, int i2, int i3]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if (_shape.Length != 4 || (uint)i0 >= (uint)_shape[0] || (uint)i1 >= (uint)_shape[1] || (uint)i2 >= (uint)_shape[2] || (uint)i3 >= (uint)_shape[3])
            {
                throw new IndexOutOfRangeException();
            }
            return ref _buffer[_offset + i0 * _strides[0] + i1 * _strides[1] + i2 * _strides[2] + i3 * _strides[3]];
        }
    }

    #endregion

    #region Zero-Copy Views & Transformations

    /// <summary>
    /// Creates a zero-copy slice of this tensor.
    /// </summary>
    public NDArray<T> Slice(params (int Start, int Stop, int Step)[] ranges)
    {
        var (newShape, newStrides, newOffset) = ShapeHelper.Slice(_shape, _strides, _offset, ranges);
        _buffer.AddRef();
        return new NDArray<T>(_buffer, newShape, newStrides, newOffset);
    }

    /// <summary>
    /// Permutes axes of this tensor (Zero-copy).
    /// </summary>
    public NDArray<T> Permute(params int[] axes)
    {
        var (newShape, newStrides) = ShapeHelper.Permute(_shape, _strides, axes);
        _buffer.AddRef();
        return new NDArray<T>(_buffer, newShape, newStrides, _offset);
    }

    /// <summary>
    /// Swaps two axes of this tensor (Zero-copy).
    /// </summary>
    public NDArray<T> Transpose(int axis0 = 0, int axis1 = 1)
    {
        int[] axes = new int[Rank];
        for (int i = 0; i < Rank; i++) axes[i] = i;
        (axes[axis0], axes[axis1]) = (axes[axis1], axes[axis0]);
        return Permute(axes);
    }

    /// <summary>
    /// 2D Matrix Transpose convenience method (swaps last two axes).
    /// </summary>
    public NDArray<T> MatrixTranspose()
    {
        if (Rank < 2) return this;
        return Transpose(Rank - 2, Rank - 1);
    }

    /// <summary>
    /// Reshapes the tensor to a new shape. If contiguous, returns a zero-copy view; otherwise copies to contiguous buffer.
    /// </summary>
    public NDArray<T> Reshape(params int[] newShape)
    {
        int newTotal = ShapeHelper.ComputeTotalLength(newShape);
        if (newTotal != _totalLength)
        {
            throw new ArgumentException($"Cannot reshape tensor of size {_totalLength} into shape {string.Join("x", newShape)} ({newTotal} elements).");
        }

        if (_isContiguous)
        {
            int[] newStrides = ShapeHelper.ComputeRowMajorStrides(newShape);
            _buffer.AddRef();
            return new NDArray<T>(_buffer, newShape, newStrides, _offset);
        }

        // Make contiguous copy then reshape
        var contig = Contiguous();
        return contig.Reshape(newShape);
    }

    /// <summary>
    /// Returns a C-contiguous version of this tensor. If already contiguous, returns this or a reference.
    /// </summary>
    public NDArray<T> Contiguous()
    {
        if (_isContiguous && _offset == 0 && _buffer.Length == _totalLength)
        {
            return this;
        }

        var result = new NDArray<T>(_shape);
        Span<T> destSpan = result.AsSpan();

        if (_shape.Length == 1)
        {
            for (int i = 0; i < _totalLength; i++)
            {
                destSpan[i] = _buffer[_offset + i * _strides[0]];
            }
        }
        else
        {
            Span<int> coords = stackalloc int[Rank];
            for (int i = 0; i < _totalLength; i++)
            {
                ShapeHelper.GetMultiIndex(i, _shape, coords);
                destSpan[i] = this[coords.ToArray()];
            }
        }

        return result;
    }

    /// <summary>
    /// Broadcasts this tensor to a target shape (Zero-copy with Stride-0 trick).
    /// </summary>
    public NDArray<T> BroadcastTo(params int[] targetShape)
    {
        int[] targetStrides = BroadcastEngine.ComputeBroadcastStrides(_shape, _strides, targetShape);
        _buffer.AddRef();
        return new NDArray<T>(_buffer, targetShape, targetStrides, _offset);
    }

    /// <summary>
    /// Creates a deep copy clone of this tensor.
    /// </summary>
    public NDArray<T> Clone()
    {
        var contig = Contiguous();
        var cloneBuffer = contig._buffer.Clone();
        return new NDArray<T>(cloneBuffer, _shape, ShapeHelper.ComputeRowMajorStrides(_shape), 0);
    }

    #endregion

    #region Direct Memory Access

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan()
    {
        if (!_isContiguous)
        {
            throw new InvalidOperationException("AsSpan() can only be called on contiguous tensors. Call .Contiguous() first.");
        }
        return _buffer.AsSpan(_offset, _totalLength);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> AsReadOnlySpan()
    {
        if (!_isContiguous)
        {
            throw new InvalidOperationException("AsReadOnlySpan() can only be called on contiguous tensors. Call .Contiguous() first.");
        }
        return _buffer.AsReadOnlySpan(_offset, _totalLength);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T* GetUnsafePointer()
    {
        return _buffer.GetUnsafePointer() + _offset;
    }

    #endregion

    #region Factory Methods

    public static NDArray<T> Zeros(params int[] shape)
    {
        return new NDArray<T>(shape);
    }

    public static NDArray<T> Ones(params int[] shape)
    {
        var result = new NDArray<T>(shape);
        result.AsSpan().Fill(T.One);
        return result;
    }

    public static NDArray<T> Full(T value, params int[] shape)
    {
        var result = new NDArray<T>(shape);
        result.AsSpan().Fill(value);
        return result;
    }

    public static NDArray<T> Eye(int size) => Eye(size, size);

    public static NDArray<T> Eye(int rows, int cols)
    {
        var result = new NDArray<T>(rows, cols);
        int min = Math.Min(rows, cols);
        for (int i = 0; i < min; i++)
        {
            result[i, i] = T.One;
        }
        return result;
    }

    public static NDArray<T> Arange(T start, T stop, T step)
    {
        if (step == T.Zero) throw new ArgumentException("Step cannot be zero.");
        List<T> values = new();
        if (step > T.Zero)
        {
            for (T v = start; v < stop; v += step) values.Add(v);
        }
        else
        {
            for (T v = start; v > stop; v += step) values.Add(v);
        }
        return FromArray(values.ToArray(), values.Count);
    }

    public static NDArray<T> Linspace(T start, T stop, int num)
    {
        if (num <= 0) throw new ArgumentException("num must be positive.");
        if (num == 1) return FromArray(new[] { start }, 1);

        T[] values = new T[num];
        T step = (stop - start) / T.CreateChecked(num - 1);
        for (int i = 0; i < num; i++)
        {
            values[i] = start + T.CreateChecked(i) * step;
        }
        return FromArray(values, num);
    }

    public static NDArray<T> FromArray(T[] array, params int[] shape)
    {
        ArgumentNullException.ThrowIfNull(array);
        int total = ShapeHelper.ComputeTotalLength(shape);
        if (array.Length != total)
        {
            throw new ArgumentException($"Array length {array.Length} does not match shape total size {total}.");
        }
        var buffer = new TensorBuffer<T>(array);
        return new NDArray<T>(buffer, shape, ShapeHelper.ComputeRowMajorStrides(shape), 0);
    }

    public static NDArray<T> AllocateNative(params int[] shape)
    {
        int total = ShapeHelper.ComputeTotalLength(shape);
        var buffer = TensorBuffer<T>.AllocateNative(total);
        return new NDArray<T>(buffer, shape, ShapeHelper.ComputeRowMajorStrides(shape), 0);
    }

    #endregion

    #region Interfaces & Formatting

    public IEnumerator<T> GetEnumerator()
    {
        for (int i = 0; i < _totalLength; i++)
        {
            yield return this[i];
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append($"NDArray<{typeof(T).Name}>(shape=[{string.Join(", ", _shape)}], dtype={DType}):\n[");
        int count = Math.Min(_totalLength, 100);
        int i = 0;
        foreach (var val in this)
        {
            if (i > 0) sb.Append(", ");
            if (i >= count)
            {
                sb.Append("...");
                break;
            }
            sb.Append(val);
            i++;
        }
        sb.Append(']');
        return sb.ToString();
    }

    public void Dispose()
    {
        _buffer.Dispose();
    }

    #endregion
}

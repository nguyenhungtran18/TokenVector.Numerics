using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using TokenVector.Numerics.Engine;

namespace TokenVector.Numerics.Core;

/// <summary>
/// N-Dimensional Boolean Array for tensor masking, logical indexing, and conditional selections.
/// </summary>
public sealed class BoolNDArray : IEnumerable<bool>
{
    private readonly bool[] _buffer;
    private readonly int[] _shape;
    private readonly int[] _strides;
    private readonly int _offset;
    private readonly int _totalLength;

    public bool[] Buffer => _buffer;
    public int[] Shape => _shape;
    public int[] Strides => _strides;
    public int Offset => _offset;
    public int TotalLength => _totalLength;
    public int Rank => _shape.Length;

    public BoolNDArray(params int[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        _shape = (int[])shape.Clone();
        _totalLength = ShapeHelper.ComputeTotalLength(_shape);
        _strides = ShapeHelper.ComputeRowMajorStrides(_shape);
        _offset = 0;
        _buffer = new bool[_totalLength];
    }

    public BoolNDArray(bool[] buffer, int[] shape, int[] strides, int offset)
    {
        _buffer = buffer;
        _shape = (int[])shape.Clone();
        _strides = (int[])strides.Clone();
        _offset = offset;
        _totalLength = ShapeHelper.ComputeTotalLength(_shape);
    }

    public ref bool this[params int[] indices]
    {
        get
        {
            int flatIndex = _offset;
            for (int i = 0; i < indices.Length; i++)
            {
                flatIndex += indices[i] * _strides[i];
            }
            return ref _buffer[flatIndex];
        }
    }

    public ref bool this[int index] => ref _buffer[_offset + index];

    public static BoolNDArray operator &(BoolNDArray a, BoolNDArray b) => BinaryLogicalOp(a, b, (x, y) => x && y);
    public static BoolNDArray operator |(BoolNDArray a, BoolNDArray b) => BinaryLogicalOp(a, b, (x, y) => x || y);
    public static BoolNDArray operator !(BoolNDArray a)
    {
        var result = new BoolNDArray(a.Shape);
        for (int i = 0; i < a.TotalLength; i++) result._buffer[i] = !a._buffer[a._offset + i];
        return result;
    }

    private static BoolNDArray BinaryLogicalOp(BoolNDArray a, BoolNDArray b, Func<bool, bool, bool> op)
    {
        int[] outShape = BroadcastEngine.BroadcastShapes(a.Shape, b.Shape);
        var res = new BoolNDArray(outShape);
        int total = res.TotalLength;
        Span<int> coords = stackalloc int[outShape.Length];

        for (int i = 0; i < total; i++)
        {
            ShapeHelper.GetMultiIndex(i, outShape, coords);
            var cArr = coords.ToArray();
            res[cArr] = op(a[cArr], b[cArr]);
        }
        return res;
    }

    public IEnumerator<bool> GetEnumerator()
    {
        for (int i = 0; i < _totalLength; i++) yield return _buffer[_offset + i];
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public override string ToString() => $"BoolNDArray(shape=[{string.Join(", ", _shape)}], count={_totalLength})";
}

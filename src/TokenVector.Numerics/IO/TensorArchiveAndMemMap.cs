// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.IO.Compression;
using System.IO.MemoryMappedFiles;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.IO;

/// <summary>
/// Provides multi-tensor NPZ archive storage and Memory-Mapped tensor access for massive out-of-core datasets (np.savez, np.load, np.memmap).
/// </summary>
public static class TensorArchive
{
    /// <summary>
    /// Saves multiple named NDArrays into a single compressed or uncompressed .npz ZIP archive (np.savez / np.savez_compressed).
    /// </summary>
    public static void SaveNpz<T>(string filePath, IDictionary<string, NDArray<T>> tensors, bool compress = true) where T : unmanaged, INumber<T>
    {
        var mode = FileMode.Create;
        using var fileStream = new FileStream(filePath, mode, FileAccess.Write);
        using var archive = new ZipArchive(fileStream, ZipArchiveMode.Create);

        var compressionLevel = compress ? CompressionLevel.Optimal : CompressionLevel.NoCompression;

        foreach (var (name, tensor) in tensors)
        {
            string entryName = name.EndsWith(".npy", StringComparison.OrdinalIgnoreCase) ? name : $"{name}.npy";
            var entry = archive.CreateEntry(entryName, compressionLevel);

            using var entryStream = entry.Open();
            using var ms = new MemoryStream();
            TensorIO.SaveNpyStream(tensor, ms);
            ms.Position = 0;
            ms.CopyTo(entryStream);
        }
    }

    /// <summary>
    /// Loads all NDArrays from a .npz ZIP archive into a dictionary of named tensors.
    /// </summary>
    public static Dictionary<string, NDArray<T>> LoadNpz<T>(string filePath) where T : unmanaged, INumber<T>
    {
        using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var archive = new ZipArchive(fileStream, ZipArchiveMode.Read);

        var dict = new Dictionary<string, NDArray<T>>();

        foreach (var entry in archive.Entries)
        {
            if (!entry.Name.EndsWith(".npy", StringComparison.OrdinalIgnoreCase))
                continue;

            string key = entry.Name[..^4]; // remove .npy
            using var entryStream = entry.Open();
            using var ms = new MemoryStream();
            entryStream.CopyTo(ms);
            ms.Position = 0;

            var tensor = TensorIO.LoadNpyStream<T>(ms);
            dict[key] = tensor;
        }

        return dict;
    }
}

/// <summary>
/// Memory-mapped NDArray representation for massive out-of-core computations directly mapped to disk.
/// </summary>
public sealed unsafe class MemoryMappedNDArray<T> : IDisposable where T : unmanaged, INumber<T>
{
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    private byte* _ptr;
    private readonly int[] _shape;
    private readonly int[] _strides;
    private readonly int _totalLength;
    private bool _disposed;

    public int[] Shape => (int[])_shape.Clone();
    public int[] Strides => (int[])_strides.Clone();
    public int TotalLength => _totalLength;
    public int Rank => _shape.Length;

    public MemoryMappedNDArray(string filePath, int[] shape, FileMode fileMode = FileMode.OpenOrCreate)
    {
        _shape = (int[])shape.Clone();
        _strides = ShapeHelper.ComputeRowMajorStrides(_shape);
        _totalLength = ShapeHelper.ComputeTotalLength(_shape);

        long byteLength = checked((long)_totalLength * sizeof(T));

        var fs = new FileStream(filePath, fileMode, FileAccess.ReadWrite, FileShare.ReadWrite);
        if (fs.Length < byteLength)
        {
            fs.SetLength(byteLength);
        }

        _mmf = MemoryMappedFile.CreateFromFile(fs, null, byteLength, MemoryMappedFileAccess.ReadWrite, HandleInheritability.None, false);
        _accessor = _mmf.CreateViewAccessor(0, byteLength, MemoryMappedFileAccess.ReadWrite);
        _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref _ptr);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T GetValue(params int[] indices)
    {
        int offset = ShapeHelper.GetFlatIndex(indices, _strides);
        return *(T*)(_ptr + (long)offset * sizeof(T));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SetValue(T value, params int[] indices)
    {
        int offset = ShapeHelper.GetFlatIndex(indices, _strides);
        *(T*)(_ptr + (long)offset * sizeof(T)) = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan()
    {
        return new Span<T>(_ptr, _totalLength);
    }

    /// <summary>
    /// Materializes the memory-mapped tensor into an in-memory NDArray.
    /// </summary>
    public NDArray<T> ToNDArray()
    {
        var array = new NDArray<T>(_shape);
        AsSpan().CopyTo(array.AsSpan());
        return array;
    }

    /// <summary>
    /// Flushes dirty pages to disk storage.
    /// </summary>
    public void Flush()
    {
        _accessor?.Flush();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_accessor != null && _ptr != null)
            {
                _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
                _accessor.Dispose();
                _accessor = null;
                _ptr = null;
            }
            _mmf?.Dispose();
            _mmf = null;
            _disposed = true;
        }
    }
}

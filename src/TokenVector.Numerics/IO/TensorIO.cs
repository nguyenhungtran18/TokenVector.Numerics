using System;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.IO;

/// <summary>
/// High-performance Tensor Serialization and I/O engine supporting NumPy .npy Binary Format, Raw Binary, and CSV.
/// </summary>
public static unsafe class TensorIO
{
    private static readonly byte[] NpyMagic = { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' };

    #region NumPy .npy Format I/O

    /// <summary>
    /// Saves tensor in standard NumPy .npy (Version 1.0) binary format to a file.
    /// </summary>
    public static void SaveNpy<T>(string filePath, NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(filePath);
        using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        SaveNpyStream(tensor, fs);
    }

    /// <summary>
    /// Saves tensor in standard NumPy .npy (Version 1.0) binary format to any writable stream.
    /// </summary>
    public static void SaveNpyStream<T>(NDArray<T> tensor, Stream stream) where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentNullException.ThrowIfNull(stream);

        var contig = tensor.Contiguous();
        string descr = GetNpyTypeDescriptor<T>();
        string shapeStr = (tensor.Rank == 1) ? $"({tensor.Shape[0]},)" : $"({string.Join(", ", tensor.Shape)})";

        // Build header dictionary string
        string dict = $"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shapeStr}, }}";
        int dictLen = Encoding.ASCII.GetByteCount(dict);

        // Header must be padded with spaces to align total (10 + headerLen) to 64 bytes
        int totalPreHeader = 10; // 6 (magic) + 1 (major) + 1 (minor) + 2 (header_len)
        int headerLen = dictLen + 1; // + 1 for '\n'
        int pad = (64 - ((totalPreHeader + headerLen) % 64)) % 64;
        headerLen += pad;

        string paddedDict = dict + new string(' ', pad) + "\n";
        byte[] headerBytes = Encoding.ASCII.GetBytes(paddedDict);

        using var bw = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);

        // 1. Magic string
        bw.Write(NpyMagic);
        // 2. Version 1.0
        bw.Write((byte)1);
        bw.Write((byte)0);
        // 3. Header len (ushort little-endian)
        bw.Write((ushort)headerBytes.Length);
        // 4. Header bytes
        bw.Write(headerBytes);

        // 5. Raw tensor data bytes
        var span = contig.AsReadOnlySpan();
        fixed (T* pData = &MemoryMarshal.GetReference(span))
        {
            var byteSpan = new ReadOnlySpan<byte>((byte*)pData, contig.TotalLength * sizeof(T));
            bw.Write(byteSpan);
        }
    }

    /// <summary>
    /// Loads tensor from standard NumPy .npy binary file.
    /// </summary>
    public static NDArray<T> LoadNpy<T>(string filePath) where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(filePath);
        if (!File.Exists(filePath)) throw new FileNotFoundException($"File not found: {filePath}");

        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        return LoadNpyStream<T>(fs);
    }

    /// <summary>
    /// Loads tensor from standard NumPy .npy binary stream.
    /// </summary>
    public static NDArray<T> LoadNpyStream<T>(Stream stream) where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var br = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);

        // 1. Check Magic
        byte[] magic = br.ReadBytes(6);
        if (!magic.AsSpan().SequenceEqual(NpyMagic))
        {
            throw new InvalidDataException("Invalid .npy file: magic header mismatch.");
        }

        // 2. Version
        byte major = br.ReadByte();
        byte minor = br.ReadByte();

        // 3. Header length
        int headerLen = (major == 1) ? br.ReadUInt16() : (int)br.ReadUInt32();
        byte[] headerBytes = br.ReadBytes(headerLen);
        string headerStr = Encoding.ASCII.GetString(headerBytes);

        // 4. Parse shape from header dictionary
        int[] shape = ParseNpyShape(headerStr);
        var result = new NDArray<T>(shape);

        // 5. Read raw data
        var span = result.AsSpan();
        fixed (T* pDest = &MemoryMarshal.GetReference(span))
        {
            var byteSpan = new Span<byte>((byte*)pDest, result.TotalLength * sizeof(T));
            int read = stream.Read(byteSpan);
            if (read < byteSpan.Length)
            {
                throw new EndOfStreamException("Unexpected end of file while reading tensor data.");
            }
        }

        return result;
    }

    private static string GetNpyTypeDescriptor<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(float)) return "<f4";
        if (typeof(T) == typeof(double)) return "<f8";
        if (typeof(T) == typeof(int)) return "<i4";
        if (typeof(T) == typeof(long)) return "<i8";
        if (typeof(T) == typeof(short)) return "<i2";
        if (typeof(T) == typeof(byte)) return "|u1";
        throw new NotSupportedException($"Type {typeof(T).Name} does not have a standard .npy type descriptor.");
    }

    private static int[] ParseNpyShape(string header)
    {
        // Example: 'shape': (2, 3),
        int shapeIdx = header.IndexOf("'shape':", StringComparison.Ordinal);
        if (shapeIdx < 0) throw new InvalidDataException("Shape descriptor not found in .npy header.");

        int openParen = header.IndexOf('(', shapeIdx);
        int closeParen = header.IndexOf(')', openParen);
        string shapeContent = header.Substring(openParen + 1, closeParen - openParen - 1).Trim();

        if (string.IsNullOrEmpty(shapeContent)) return Array.Empty<int>(); // 0-dim scalar

        string[] tokens = shapeContent.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        int[] shape = new int[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            shape[i] = int.Parse(tokens[i]);
        }
        return shape;
    }

    #endregion

    #region Raw Binary I/O

    public static void SaveBinary<T>(string filePath, NDArray<T> tensor) where T : unmanaged, INumber<T>
    {
        using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs);

        bw.Write(tensor.Rank);
        for (int d = 0; d < tensor.Rank; d++) bw.Write(tensor.Shape[d]);

        var contig = tensor.Contiguous();
        var span = contig.AsReadOnlySpan();
        fixed (T* pData = &MemoryMarshal.GetReference(span))
        {
            bw.Write(new ReadOnlySpan<byte>((byte*)pData, contig.TotalLength * sizeof(T)));
        }
    }

    public static NDArray<T> LoadBinary<T>(string filePath) where T : unmanaged, INumber<T>
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);

        int rank = br.ReadInt32();
        int[] shape = new int[rank];
        for (int d = 0; d < rank; d++) shape[d] = br.ReadInt32();

        var result = new NDArray<T>(shape);
        var span = result.AsSpan();
        fixed (T* pDest = &MemoryMarshal.GetReference(span))
        {
            fs.ReadExactly(new Span<byte>((byte*)pDest, result.TotalLength * sizeof(T)));
        }

        return result;
    }

    #endregion

    #region CSV I/O

    public static void SaveCsv<T>(string filePath, NDArray<T> tensor, string delimiter = ",") where T : unmanaged, INumber<T>
    {
        if (tensor.Rank != 2) throw new ArgumentException("SaveCsv requires a 2D matrix.");
        int rows = tensor.Shape[0];
        int cols = tensor.Shape[1];

        using var sw = new StreamWriter(filePath, false, Encoding.UTF8);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (c > 0) sw.Write(delimiter);
                sw.Write(tensor[r, c]);
            }
            sw.WriteLine();
        }
    }

    #endregion
}

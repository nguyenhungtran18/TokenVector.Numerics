using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TokenVector.Numerics.LinAlg;

/// <summary>
/// Lightweight generic complex number struct for high-performance DSP, FFT, and tensor mathematics.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Complex<T> : IEquatable<Complex<T>> where T : unmanaged, INumber<T>
{
    public readonly T Real;
    public readonly T Imaginary;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Complex(T real, T imaginary)
    {
        Real = real;
        Imaginary = imaginary;
    }

    public static Complex<T> Zero => new(T.Zero, T.Zero);
    public static Complex<T> One => new(T.One, T.Zero);
    public static Complex<T> ImaginaryOne => new(T.Zero, T.One);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex<T> operator +(Complex<T> a, Complex<T> b) => new(a.Real + b.Real, a.Imaginary + b.Imaginary);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex<T> operator -(Complex<T> a, Complex<T> b) => new(a.Real - b.Real, a.Imaginary - b.Imaginary);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex<T> operator *(Complex<T> a, Complex<T> b) =>
        new(a.Real * b.Real - a.Imaginary * b.Imaginary, a.Real * b.Imaginary + a.Imaginary * b.Real);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex<T> operator *(Complex<T> a, T scalar) => new(a.Real * scalar, a.Imaginary * scalar);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex<T> operator /(Complex<T> a, T scalar) => new(a.Real / scalar, a.Imaginary / scalar);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Complex<T> operator /(Complex<T> a, Complex<T> b)
    {
        T denom = b.Real * b.Real + b.Imaginary * b.Imaginary;
        return new((a.Real * b.Real + a.Imaginary * b.Imaginary) / denom, (a.Imaginary * b.Real - a.Real * b.Imaginary) / denom);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Complex<T> Conjugate() => new(Real, -Imaginary);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T MagnitudeSquared() => Real * Real + Imaginary * Imaginary;

    public bool Equals(Complex<T> other) => Real.Equals(other.Real) && Imaginary.Equals(other.Imaginary);
    public override bool Equals(object? obj) => obj is Complex<T> other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(Real, Imaginary);

    public static bool operator ==(Complex<T> left, Complex<T> right) => left.Equals(right);
    public static bool operator !=(Complex<T> left, Complex<T> right) => !left.Equals(right);

    public override string ToString() => $"({Real} + {Imaginary}j)";
}

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Spatial;

/// <summary>
/// 4D Quaternion struct for smooth 3D rotations, spatial algebra, and Spherical Linear Interpolation (Slerp).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Quaternion<T> : IEquatable<Quaternion<T>>
    where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>, IRootFunctions<T>
{
    public readonly T X;
    public readonly T Y;
    public readonly T Z;
    public readonly T W;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Quaternion(T x, T y, T z, T w)
    {
        X = x;
        Y = y;
        Z = z;
        W = w;
    }

    public static Quaternion<T> Identity => new(T.Zero, T.Zero, T.Zero, T.One);

    /// <summary>
    /// Creates a quaternion from an axis vector and rotation angle in radians.
    /// </summary>
    public static Quaternion<T> FromAxisAngle(T ax, T ay, T az, T radians)
    {
        T halfAngle = radians / T.CreateChecked(2.0);
        T sin = T.Sin(halfAngle);
        T cos = T.Cos(halfAngle);

        T norm = T.Sqrt(ax * ax + ay * ay + az * az);
        if (norm > T.Zero)
        {
            ax /= norm;
            ay /= norm;
            az /= norm;
        }

        return new Quaternion<T>(ax * sin, ay * sin, az * sin, cos);
    }

    /// <summary>
    /// Creates a quaternion from Euler angles (Yaw: Y, Pitch: X, Roll: Z in radians).
    /// </summary>
    public static Quaternion<T> FromEuler(T pitch, T yaw, T roll)
    {
        T halfPitch = pitch / T.CreateChecked(2.0);
        T halfYaw = yaw / T.CreateChecked(2.0);
        T halfRoll = roll / T.CreateChecked(2.0);

        T cX = T.Cos(halfPitch);
        T sX = T.Sin(halfPitch);
        T cY = T.Cos(halfYaw);
        T sY = T.Sin(halfYaw);
        T cZ = T.Cos(halfRoll);
        T sZ = T.Sin(halfRoll);

        return new Quaternion<T>(
            sX * cY * cZ + cX * sY * sZ,
            cX * sY * cZ - sX * cY * sZ,
            cX * cY * sZ - sX * sY * cZ,
            cX * cY * cZ + sX * sY * sZ
        );
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Quaternion<T> operator *(Quaternion<T> a, Quaternion<T> b)
    {
        return new Quaternion<T>(
            a.W * b.X + a.X * b.W + a.Y * b.Z - a.Z * b.Y,
            a.W * b.Y - a.X * b.Z + a.Y * b.W + a.Z * b.X,
            a.W * b.Z + a.X * b.Y - a.Y * b.X + a.Z * b.W,
            a.W * b.W - a.X * b.X - a.Y * b.Y - a.Z * b.Z
        );
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Quaternion<T> operator -(Quaternion<T> q) => new(-q.X, -q.Y, -q.Z, -q.W);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Quaternion<T> Conjugate() => new(-X, -Y, -Z, W);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T LengthSquared() => X * X + Y * Y + Z * Z + W * W;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Length() => T.Sqrt(LengthSquared());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Quaternion<T> Normalize()
    {
        T len = Length();
        if (len < T.CreateChecked(1e-15)) return Identity;
        return new Quaternion<T>(X / len, Y / len, Z / len, W / len);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Quaternion<T> Inverse()
    {
        T lenSq = LengthSquared();
        if (lenSq < T.CreateChecked(1e-15)) return Identity;
        return new Quaternion<T>(-X / lenSq, -Y / lenSq, -Z / lenSq, W / lenSq);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Dot(Quaternion<T> a, Quaternion<T> b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z + a.W * b.W;

    /// <summary>
    /// Performs Spherical Linear Interpolation (Slerp) between two quaternions.
    /// </summary>
    public static Quaternion<T> Slerp(Quaternion<T> q1, Quaternion<T> q2, T t)
    {
        T cosTheta = Dot(q1, q2);

        // Take shortest path along the 4D hypersphere
        if (cosTheta < T.Zero)
        {
            q2 = -q2;
            cosTheta = -cosTheta;
        }

        // If quaternions are very close, fallback to linear interpolation to prevent division by zero
        if (cosTheta > T.CreateChecked(0.9995))
        {
            T x = q1.X + t * (q2.X - q1.X);
            T y = q1.Y + t * (q2.Y - q1.Y);
            T z = q1.Z + t * (q2.Z - q1.Z);
            T w = q1.W + t * (q2.W - q1.W);
            return new Quaternion<T>(x, y, z, w).Normalize();
        }

        // Standard Slerp
        T theta = T.Acos(Math.Clamp(double.CreateChecked(cosTheta), -1.0, 1.0) is var cl ? T.CreateChecked(cl) : cosTheta);
        T sinTheta = T.Sin(theta);

        T scale1 = T.Sin((T.One - t) * theta) / sinTheta;
        T scale2 = T.Sin(t * theta) / sinTheta;

        return new Quaternion<T>(
            scale1 * q1.X + scale2 * q2.X,
            scale1 * q1.Y + scale2 * q2.Y,
            scale1 * q1.Z + scale2 * q2.Z,
            scale1 * q1.W + scale2 * q2.W
        ).Normalize();
    }

    /// <summary>
    /// Converts quaternion to 4x4 rotation matrix.
    /// </summary>
    public NDArray<T> ToRotationMatrix4x4()
    {
        var q = Normalize();
        T xx = q.X * q.X, yy = q.Y * q.Y, zz = q.Z * q.Z;
        T xy = q.X * q.Y, xz = q.X * q.Z, yz = q.Y * q.Z;
        T wx = q.W * q.X, wy = q.W * q.Y, wz = q.W * q.Z;

        T two = T.CreateChecked(2.0);
        var m = NDArray<T>.Eye(4);

        m[0, 0] = T.One - two * (yy + zz);
        m[0, 1] = two * (xy - wz);
        m[0, 2] = two * (xz + wy);

        m[1, 0] = two * (xy + wz);
        m[1, 1] = T.One - two * (xx + zz);
        m[1, 2] = two * (yz - wx);

        m[2, 0] = two * (xz - wy);
        m[2, 1] = two * (yz + wx);
        m[2, 2] = T.One - two * (xx + yy);

        return m;
    }

    public bool Equals(Quaternion<T> other) => X.Equals(other.X) && Y.Equals(other.Y) && Z.Equals(other.Z) && W.Equals(other.W);
    public override bool Equals(object? obj) => obj is Quaternion<T> other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(X, Y, Z, W);

    public static bool operator ==(Quaternion<T> left, Quaternion<T> right) => left.Equals(right);
    public static bool operator !=(Quaternion<T> left, Quaternion<T> right) => !left.Equals(right);

    public override string ToString() => $"Quaternion<{typeof(T).Name}>(x={X}, y={Y}, z={Z}, w={W})";
}

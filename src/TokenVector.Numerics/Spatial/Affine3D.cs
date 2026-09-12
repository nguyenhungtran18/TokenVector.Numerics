using System;
using System.Numerics;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Spatial;

/// <summary>
/// Computer graphics and 3D spatial transformations: 4x4 Affine matrices, Camera projections, Vector Dot and Cross products.
/// </summary>
public static class Affine3D
{
    /// <summary>
    /// Creates a 4x4 3D translation matrix.
    /// </summary>
    public static NDArray<T> Translation<T>(T tx, T ty, T tz) where T : unmanaged, IFloatingPoint<T>
    {
        var m = NDArray<T>.Eye(4);
        m[0, 3] = tx;
        m[1, 3] = ty;
        m[2, 3] = tz;
        return m;
    }

    /// <summary>
    /// Creates a 4x4 3D scaling matrix.
    /// </summary>
    public static NDArray<T> Scaling<T>(T sx, T sy, T sz) where T : unmanaged, IFloatingPoint<T>
    {
        var m = NDArray<T>.Eye(4);
        m[0, 0] = sx;
        m[1, 1] = sy;
        m[2, 2] = sz;
        return m;
    }

    /// <summary>
    /// Creates a 4x4 rotation matrix around X-axis (angle in radians).
    /// </summary>
    public static NDArray<T> RotationX<T>(T radians) where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>
    {
        var m = NDArray<T>.Eye(4);
        T cos = T.Cos(radians);
        T sin = T.Sin(radians);
        m[1, 1] = cos;
        m[1, 2] = -sin;
        m[2, 1] = sin;
        m[2, 2] = cos;
        return m;
    }

    /// <summary>
    /// Creates a 4x4 rotation matrix around Y-axis (angle in radians).
    /// </summary>
    public static NDArray<T> RotationY<T>(T radians) where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>
    {
        var m = NDArray<T>.Eye(4);
        T cos = T.Cos(radians);
        T sin = T.Sin(radians);
        m[0, 0] = cos;
        m[0, 2] = sin;
        m[2, 0] = -sin;
        m[2, 2] = cos;
        return m;
    }

    /// <summary>
    /// Creates a 4x4 rotation matrix around Z-axis (angle in radians).
    /// </summary>
    public static NDArray<T> RotationZ<T>(T radians) where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>
    {
        var m = NDArray<T>.Eye(4);
        T cos = T.Cos(radians);
        T sin = T.Sin(radians);
        m[0, 0] = cos;
        m[0, 1] = -sin;
        m[1, 0] = sin;
        m[1, 1] = cos;
        return m;
    }

    /// <summary>
    /// Creates a 4x4 rotation matrix around an arbitrary axis vector.
    /// </summary>
    public static NDArray<T> RotationAxisAngle<T>(T ax, T ay, T az, T radians) where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>, IRootFunctions<T>
    {
        T norm = T.Sqrt(ax * ax + ay * ay + az * az);
        if (norm > T.Zero)
        {
            ax /= norm;
            ay /= norm;
            az /= norm;
        }

        T c = T.Cos(radians);
        T s = T.Sin(radians);
        T t = T.One - c;

        var m = NDArray<T>.Eye(4);
        m[0, 0] = t * ax * ax + c;
        m[0, 1] = t * ax * ay - s * az;
        m[0, 2] = t * ax * az + s * ay;

        m[1, 0] = t * ax * ay + s * az;
        m[1, 1] = t * ay * ay + c;
        m[1, 2] = t * ay * az - s * ax;

        m[2, 0] = t * ax * az - s * ay;
        m[2, 1] = t * ay * az + s * ax;
        m[2, 2] = t * az * az + c;

        return m;
    }

    /// <summary>
    /// Creates a View Matrix (LookAt camera).
    /// </summary>
    public static NDArray<T> LookAt<T>(
        (T X, T Y, T Z) eye,
        (T X, T Y, T Z) target,
        (T X, T Y, T Z) up) where T : unmanaged, IFloatingPoint<T>, IRootFunctions<T>
    {
        // Forward vector zAxis = normalize(eye - target)
        T fX = eye.X - target.X;
        T fY = eye.Y - target.Y;
        T fZ = eye.Z - target.Z;
        T fNorm = T.Sqrt(fX * fX + fY * fY + fZ * fZ);
        if (fNorm > T.Zero) { fX /= fNorm; fY /= fNorm; fZ /= fNorm; }

        // Right vector xAxis = normalize(cross(up, zAxis))
        T rX = up.Y * fZ - up.Z * fY;
        T rY = up.Z * fX - up.X * fZ;
        T rZ = up.X * fY - up.Y * fX;
        T rNorm = T.Sqrt(rX * rX + rY * rY + rZ * rZ);
        if (rNorm > T.Zero) { rX /= rNorm; rY /= rNorm; rZ /= rNorm; }

        // Up vector yAxis = cross(zAxis, xAxis)
        T uX = fY * rZ - fZ * rY;
        T uY = fZ * rX - fX * rZ;
        T uZ = fX * rY - fY * rX;

        var m = NDArray<T>.Eye(4);
        m[0, 0] = rX; m[0, 1] = rY; m[0, 2] = rZ; m[0, 3] = -(rX * eye.X + rY * eye.Y + rZ * eye.Z);
        m[1, 0] = uX; m[1, 1] = uY; m[1, 2] = uZ; m[1, 3] = -(uX * eye.X + uY * eye.Y + uZ * eye.Z);
        m[2, 0] = fX; m[2, 1] = fY; m[2, 2] = fZ; m[2, 3] = -(fX * eye.X + fY * eye.Y + fZ * eye.Z);

        return m;
    }

    /// <summary>
    /// Creates a Perspective projection matrix.
    /// </summary>
    public static NDArray<T> PerspectiveFov<T>(T fovYRadians, T aspectRatio, T nearZ, T farZ) where T : unmanaged, IFloatingPoint<T>, ITrigonometricFunctions<T>
    {
        T halfFov = fovYRadians / T.CreateChecked(2.0);
        T tanHalfFov = T.Tan(halfFov);
        T yScale = T.One / tanHalfFov;
        T xScale = yScale / aspectRatio;

        var m = NDArray<T>.Zeros(4, 4);
        m[0, 0] = xScale;
        m[1, 1] = yScale;
        m[2, 2] = (farZ + nearZ) / (nearZ - farZ);
        m[2, 3] = (T.CreateChecked(2.0) * farZ * nearZ) / (nearZ - farZ);
        m[3, 2] = -T.One;

        return m;
    }

    /// <summary>
    /// Creates an Orthographic projection matrix.
    /// </summary>
    public static NDArray<T> Orthographic<T>(T width, T height, T nearZ, T farZ) where T : unmanaged, IFloatingPoint<T>
    {
        var m = NDArray<T>.Eye(4);
        m[0, 0] = T.CreateChecked(2.0) / width;
        m[1, 1] = T.CreateChecked(2.0) / height;
        m[2, 2] = -T.CreateChecked(2.0) / (farZ - nearZ);
        m[2, 3] = -(farZ + nearZ) / (farZ - nearZ);

        return m;
    }

    /// <summary>
    /// Computes 3D cross product of two 3-element vectors: a x b.
    /// </summary>
    public static NDArray<T> Cross<T>(NDArray<T> a, NDArray<T> b) where T : unmanaged, INumber<T>
    {
        if (a.TotalLength != 3 || b.TotalLength != 3)
        {
            throw new ArgumentException("Cross product requires 3-element vectors.");
        }

        var result = new NDArray<T>(3);
        result[0] = a[1] * b[2] - a[2] * b[1];
        result[1] = a[2] * b[0] - a[0] * b[2];
        result[2] = a[0] * b[1] - a[1] * b[0];

        return result;
    }

    /// <summary>
    /// Computes dot product of two 1D vectors.
    /// </summary>
    public static T Dot<T>(NDArray<T> a, NDArray<T> b) where T : unmanaged, INumber<T>
    {
        if (a.TotalLength != b.TotalLength)
        {
            throw new ArgumentException("Vectors must have equal length for Dot product.");
        }
        return (a * b).Sum();
    }
}

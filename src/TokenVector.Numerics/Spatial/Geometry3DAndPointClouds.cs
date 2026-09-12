// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Spatial;

/// <summary>
/// Provides advanced 3D computational geometry, point cloud registration (Iterative Closest Point - ICP), ray tracing intersections, and coordinate frames.
/// </summary>
public static class Geometry3DAndPointClouds
{
    #region Distance & Intersections

    /// <summary>
    /// Computes signed and absolute distance from a 3D point to a plane defined by a point and unit normal vector.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double PointToPlaneDistance(NDArray<double> point, NDArray<double> planePoint, NDArray<double> planeNormal)
    {
        double dot = 0.0;
        for (int i = 0; i < 3; i++)
        {
            dot += (point[i] - planePoint[i]) * planeNormal[i];
        }
        return dot;
    }

    /// <summary>
    /// Computes minimum Euclidean distance from a point P to a line segment AB.
    /// </summary>
    public static double PointToSegmentDistance(NDArray<double> p, NDArray<double> a, NDArray<double> b)
    {
        double abX = b[0] - a[0], abY = b[1] - a[1], abZ = b[2] - a[2];
        double apX = p[0] - a[0], apY = p[1] - a[1], apZ = p[2] - a[2];

        double abLenSq = abX * abX + abY * abY + abZ * abZ;
        if (abLenSq < 1e-15)
        {
            return Math.Sqrt(apX * apX + apY * apY + apZ * apZ);
        }

        double t = Math.Clamp((apX * abX + apY * abY + apZ * abZ) / abLenSq, 0.0, 1.0);
        double projX = a[0] + t * abX;
        double projY = a[1] + t * abY;
        double projZ = a[2] + t * abZ;

        double dx = p[0] - projX;
        double dy = p[1] - projY;
        double dz = p[2] - projZ;

        return Math.Sqrt(dx * dx + dy * dy + dz * dz);
    }

    /// <summary>
    /// Möller–Trumbore Ray-Triangle Intersection algorithm.
    /// Returns (hasHit, distance t, barycentric u, barycentric v).
    /// </summary>
    public static (bool HasHit, double T, double U, double V) RayTriangleIntersect(
        NDArray<double> rayOrigin, NDArray<double> rayDir,
        NDArray<double> v0, NDArray<double> v1, NDArray<double> v2)
    {
        const double eps = 1e-7;

        double edge1X = v1[0] - v0[0], edge1Y = v1[1] - v0[0], edge1Z = v1[2] - v0[2];
        // corrected edge indices:
        edge1Y = v1[1] - v0[1];
        edge1Z = v1[2] - v0[2];

        double edge2X = v2[0] - v0[0], edge2Y = v2[1] - v0[1], edge2Z = v2[2] - v0[2];

        // pvec = rayDir x edge2
        double pvecX = rayDir[1] * edge2Z - rayDir[2] * edge2Y;
        double pvecY = rayDir[2] * edge2X - rayDir[0] * edge2Z;
        double pvecZ = rayDir[0] * edge2Y - rayDir[1] * edge2X;

        double det = edge1X * pvecX + edge1Y * pvecY + edge1Z * pvecZ;
        if (Math.Abs(det) < eps) return (false, 0.0, 0.0, 0.0);

        double invDet = 1.0 / det;

        // tvec = rayOrigin - v0
        double tvecX = rayOrigin[0] - v0[0], tvecY = rayOrigin[1] - v0[1], tvecZ = rayOrigin[2] - v0[2];
        double u = (tvecX * pvecX + tvecY * pvecY + tvecZ * pvecZ) * invDet;
        if (u < 0.0 || u > 1.0) return (false, 0.0, 0.0, 0.0);

        // qvec = tvec x edge1
        double qvecX = tvecY * edge1Z - tvecZ * edge1Y;
        double qvecY = tvecZ * edge1X - tvecX * edge1Z;
        double qvecZ = tvecX * edge1Y - tvecY * edge1X;

        double v = (rayDir[0] * qvecX + rayDir[1] * qvecY + rayDir[2] * qvecZ) * invDet;
        if (v < 0.0 || u + v > 1.0) return (false, 0.0, 0.0, 0.0);

        double t = (edge2X * qvecX + edge2Y * qvecY + edge2Z * qvecZ) * invDet;
        if (t < eps) return (false, 0.0, 0.0, 0.0);

        return (true, t, u, v);
    }

    /// <summary>
    /// Checks if two 3D Axis-Aligned Bounding Boxes (AABB) intersect.
    /// </summary>
    public static bool AABBIntersect(NDArray<double> minA, NDArray<double> maxA, NDArray<double> minB, NDArray<double> maxB)
    {
        return (minA[0] <= maxB[0] && maxA[0] >= minB[0]) &&
               (minA[1] <= maxB[1] && maxA[1] >= minB[1]) &&
               (minA[2] <= maxB[2] && maxA[2] >= minB[2]);
    }

    #endregion

    #region Point Cloud Registration (Iterative Closest Point - ICP)

    /// <summary>
    /// Aligns source point cloud to target point cloud using Kabsch/SVD rigid registration step (Rotation R [3, 3] and Translation t [3]).
    /// </summary>
    public static (NDArray<double> R, NDArray<double> T) AlignPointCloudsKabsch(NDArray<double> source, NDArray<double> target)
    {
        int n = source.Shape[0];
        if (target.Shape[0] != n || source.Shape[1] != 3 || target.Shape[1] != 3)
            throw new ArgumentException("Source and target point clouds must have matching shape [N, 3].");

        // 1. Compute Centroids
        var centroidSrc = new NDArray<double>(3);
        var centroidTgt = new NDArray<double>(3);

        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < 3; d++)
            {
                centroidSrc[d] += source[i, d];
                centroidTgt[d] += target[i, d];
            }
        }
        for (int d = 0; d < 3; d++)
        {
            centroidSrc[d] /= n;
            centroidTgt[d] /= n;
        }

        // 2. Center the clouds & compute cross-covariance matrix H = P^T * Q
        var h = new NDArray<double>(3, 3);
        for (int i = 0; i < n; i++)
        {
            double px = source[i, 0] - centroidSrc[0];
            double py = source[i, 1] - centroidSrc[1];
            double pz = source[i, 2] - centroidSrc[2];

            double qx = target[i, 0] - centroidTgt[0];
            double qy = target[i, 1] - centroidTgt[1];
            double qz = target[i, 2] - centroidTgt[2];

            h[0, 0] += px * qx; h[0, 1] += px * qy; h[0, 2] += px * qz;
            h[1, 0] += py * qx; h[1, 1] += py * qy; h[1, 2] += py * qz;
            h[2, 0] += pz * qx; h[2, 1] += pz * qy; h[2, 2] += pz * qz;
        }

        // 3. SVD of H: H = U * S * V^T
        var (u, _, vt) = SVD.Decompose(h);
        var v = vt.MatrixTranspose();
        var uT = u.MatrixTranspose();

        // R = V * U^T
        var r = MatrixMultiplication.MatMul(v, uT);

        // Ensure right-handed coordinate system (det(R) == 1)
        double detR = Decomposition.Det(r);
        if (detR < 0)
        {
            // Negate third column of V
            for (int i = 0; i < 3; i++) v[i, 2] = -v[i, 2];
            r = MatrixMultiplication.MatMul(v, uT);
        }

        // Translation t = centroidTgt - R * centroidSrc
        var rCentroidSrc = MatrixMultiplication.MatMul(r, centroidSrc.Reshape(3, 1)).Reshape(3);
        var tVec = centroidTgt - rCentroidSrc;

        return (r, tVec);
    }

    #endregion

    #region Coordinate System Conversions

    /// <summary>
    /// Converts Cartesian (x, y, z) to Spherical (r, theta [azimuth], phi [inclination from +z]).
    /// </summary>
    public static (double R, double Theta, double Phi) CartesianToSpherical(double x, double y, double z)
    {
        double r = Math.Sqrt(x * x + y * y + z * z);
        if (r < 1e-15) return (0.0, 0.0, 0.0);
        double theta = Math.Atan2(y, x);
        double phi = Math.Acos(Math.Clamp(z / r, -1.0, 1.0));
        return (r, theta, phi);
    }

    /// <summary>
    /// Converts Spherical (r, theta, phi) to Cartesian (x, y, z).
    /// </summary>
    public static (double X, double Y, double Z) SphericalToCartesian(double r, double theta, double phi)
    {
        double x = r * Math.Sin(phi) * Math.Cos(theta);
        double y = r * Math.Sin(phi) * Math.Sin(theta);
        double z = r * Math.Cos(phi);
        return (x, y, z);
    }

    #endregion
}

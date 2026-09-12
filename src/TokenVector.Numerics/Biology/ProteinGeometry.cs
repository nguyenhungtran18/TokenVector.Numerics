// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;
using TokenVector.Numerics.Spatial;

namespace TokenVector.Numerics.Biology;

/// <summary>
/// Provides computational structural biology and protein mathematics (AlphaFold SE(3) geometry, Dihedral angles, Kabsch RMSD, TM-score).
/// </summary>
public static class ProteinGeometry
{
    /// <summary>
    /// Computes the dihedral (torsion) angle in radians [-pi, pi] defined by 4 consecutive 3D atom coordinates (p1, p2, p3, p4).
    /// Used for calculating protein backbone angles (phi, psi, omega).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static double ComputeDihedralAngle(NDArray<double> p1, NDArray<double> p2, NDArray<double> p3, NDArray<double> p4)
    {
        // Vectors between points: b1 = p2 - p1, b2 = p3 - p2, b3 = p4 - p3
        double b1x = p2[0] - p1[0], b1y = p2[1] - p1[1], b1z = p2[2] - p1[2];
        double b2x = p3[0] - p2[0], b2y = p3[1] - p2[1], b2z = p3[2] - p2[2];
        double b3x = p4[0] - p3[0], b3y = p4[1] - p3[1], b3z = p4[2] - p3[2];

        // Normal vectors: n1 = b1 x b2, n2 = b2 x b3
        double n1x = b1y * b2z - b1z * b2y;
        double n1y = b1z * b2x - b1x * b2z;
        double n1z = b1x * b2y - b1y * b2x;

        double n2x = b2y * b3z - b2z * b3y;
        double n2y = b2z * b3x - b2x * b3z;
        double n2z = b2x * b3y - b2y * b3x;

        // Unit vector along b2
        double b2Len = Math.Sqrt(b2x * b2x + b2y * b2y + b2z * b2z);
        if (b2Len < 1e-15) return 0.0;
        double m1x = b2x / b2Len, m1y = b2y / b2Len, m1z = b2z / b2Len;

        // x = n1 . n2, y = (n1 x m1) . n2
        double x = n1x * n2x + n1y * n2y + n1z * n2z;

        double n1CrossMx = n1y * m1z - n1z * m1y;
        double n1CrossMy = n1z * m1x - n1x * m1z;
        double n1CrossMz = n1x * m1y - n1y * m1x;

        double y = n1CrossMx * n2x + n1CrossMy * n2y + n1CrossMz * n2z;

        return Math.Atan2(y, x);
    }

    /// <summary>
    /// Computes optimal Root-Mean-Square Deviation (RMSD) between predicted and true 3D coordinate structures [N, 3] after Kabsch superposition.
    /// </summary>
    public static double KabschRMSD(NDArray<double> predicted, NDArray<double> reference)
    {
        int n = predicted.Shape[0];
        var (r, t) = Geometry3DAndPointClouds.AlignPointCloudsKabsch(predicted, reference);

        double totalDistSq = 0.0;
        for (int i = 0; i < n; i++)
        {
            // Transformed: p_align = R * p + t
            double px = predicted[i, 0], py = predicted[i, 1], pz = predicted[i, 2];
            double rx = r[0, 0] * px + r[0, 1] * py + r[0, 2] * pz + t[0];
            double ry = r[1, 0] * px + r[1, 1] * py + r[1, 2] * pz + t[1];
            double rz = r[2, 0] * px + r[2, 1] * py + r[2, 2] * pz + t[2];

            double dx = rx - reference[i, 0];
            double dy = ry - reference[i, 1];
            double dz = rz - reference[i, 2];

            totalDistSq += dx * dx + dy * dy + dz * dz;
        }

        return Math.Sqrt(totalDistSq / n);
    }

    /// <summary>
    /// Computes TM-score (0 to 1) for assessing structural similarity between two protein structures.
    /// TM-score > 0.5 indicates generally the same protein fold.
    /// </summary>
    public static double TMScore(NDArray<double> predicted, NDArray<double> reference)
    {
        int lRef = reference.Shape[0];
        if (lRef == 0) return 0.0;

        // Scaling factor d0 = 1.24 * (L_ref - 15)^(1/3) - 1.8
        double d0 = (lRef > 15) ? (1.24 * Math.Cbrt(lRef - 15) - 1.8) : 0.5;
        if (d0 < 0.5) d0 = 0.5;
        double d0Sq = d0 * d0;

        var (r, t) = Geometry3DAndPointClouds.AlignPointCloudsKabsch(predicted, reference);

        double sum = 0.0;
        for (int i = 0; i < lRef; i++)
        {
            double px = predicted[i, 0], py = predicted[i, 1], pz = predicted[i, 2];
            double rx = r[0, 0] * px + r[0, 1] * py + r[0, 2] * pz + t[0];
            double ry = r[1, 0] * px + r[1, 1] * py + r[1, 2] * pz + t[1];
            double rz = r[2, 0] * px + r[2, 1] * py + r[2, 2] * pz + t[2];

            double dx = rx - reference[i, 0];
            double dy = ry - reference[i, 1];
            double dz = rz - reference[i, 2];

            double diSq = dx * dx + dy * dy + dz * dz;
            sum += 1.0 / (1.0 + diSq / d0Sq);
        }

        return sum / lRef;
    }
}

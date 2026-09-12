// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Science;

/// <summary>
/// Provides orbital mechanics, Keplerian orbit propagation, Hohmann transfers, and geodetic coordinate transformations for aerospace engineering.
/// </summary>
public static class Astrodynamics
{
    // Standard Earth Gravitational Parameter mu = GM (km^3 / s^2)
    public const double EarthMu = 398600.4418;
    // WGS84 Ellipsoid Constants
    public const double Wgs84A = 6378.137; // Semi-major axis in km
    public const double Wgs84F = 1.0 / 298.257223563; // Flattening
    public const double Wgs84B = Wgs84A * (1.0 - Wgs84F); // Semi-minor axis
    public const double Wgs84E2 = 2.0 * Wgs84F - Wgs84F * Wgs84F; // First eccentricity squared

    /// <summary>
    /// Solves Kepler's Equation M = E - e*sin(E) for Eccentric Anomaly E using Newton-Raphson with cubic Halley starter.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static double SolveKepler(double meanAnomaly, double eccentricity, double tol = 1e-12, int maxIter = 100)
    {
        // Normalize M to [-pi, pi]
        double m = meanAnomaly % (2.0 * Math.PI);
        if (m < -Math.PI) m += 2.0 * Math.PI;
        if (m > Math.PI) m -= 2.0 * Math.PI;

        // Initial guess
        double e = eccentricity;
        double en = (e < 0.8) ? m : Math.PI;

        for (int iter = 0; iter < maxIter; iter++)
        {
            double sinE = Math.Sin(en);
            double cosE = Math.Cos(en);
            double f = en - e * sinE - m;
            double fPrime = 1.0 - e * cosE;

            if (Math.Abs(f) < tol) return en;

            // Halley step
            double fDoublePrime = e * sinE;
            double delta = f / (fPrime - 0.5 * f * fDoublePrime / fPrime);
            en -= delta;

            if (Math.Abs(delta) < tol) break;
        }

        return en;
    }

    /// <summary>
    /// Converts 6 Keplerian orbital elements (a, e, i, raan, omega, nu) to Cartesian position (r) and velocity (v) vectors in km and km/s.
    /// Angles must be in radians.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static (NDArray<double> Position, NDArray<double> Velocity) KeplerianToCartesian(
        double a, double e, double i, double raan, double omega, double nu, double mu = EarthMu)
    {
        // Semilatus rectum
        double p = a * (1.0 - e * e);
        double rMag = p / (1.0 + e * Math.Cos(nu));

        // Position and velocity in Perifocal coordinate system (PQW)
        double rP = rMag * Math.Cos(nu);
        double rQ = rMag * Math.Sin(nu);

        double sqrtMuOverP = Math.Sqrt(mu / p);
        double vP = -sqrtMuOverP * Math.Sin(nu);
        double vQ = sqrtMuOverP * (e + Math.Cos(nu));

        // Rotation matrix from Perifocal to ECI frame: R = Rz(-raan) * Rx(-i) * Rz(-omega)
        double cosO = Math.Cos(raan), sinO = Math.Sin(raan);
        double cosI = Math.Cos(i), sinI = Math.Sin(i);
        double cosW = Math.Cos(omega), sinW = Math.Sin(omega);

        double p11 = cosO * cosW - sinO * sinW * cosI;
        double p12 = -cosO * sinW - sinO * cosW * cosI;
        double p21 = sinO * cosW + cosO * sinW * cosI;
        double p22 = -sinO * sinW + cosO * cosW * cosI;
        double p31 = sinW * sinI;
        double p32 = cosW * sinI;

        double rx = p11 * rP + p12 * rQ;
        double ry = p21 * rP + p22 * rQ;
        double rz = p31 * rP + p32 * rQ;

        double vx = p11 * vP + p12 * vQ;
        double vy = p21 * vP + p22 * vQ;
        double vz = p31 * vP + p32 * vQ;

        var pos = NDArray<double>.FromArray([rx, ry, rz], 3);
        var vel = NDArray<double>.FromArray([vx, vy, vz], 3);
        return (pos, vel);
    }

    /// <summary>
    /// Computes Hohmann transfer Delta-V and time-of-flight between two coplanar circular orbits with radii r1 and r2.
    /// </summary>
    public static (double DeltaV1, double DeltaV2, double TotalDeltaV, double TimeOfFlight) HohmannTransfer(
        double r1, double r2, double mu = EarthMu)
    {
        double v1 = Math.Sqrt(mu / r1);
        double v2 = Math.Sqrt(mu / r2);

        // Semi-major axis of elliptical transfer orbit
        double aTrans = (r1 + r2) / 2.0;

        double vTrans1 = Math.Sqrt(mu * (2.0 / r1 - 1.0 / aTrans));
        double vTrans2 = Math.Sqrt(mu * (2.0 / r2 - 1.0 / aTrans));

        double dv1 = Math.Abs(vTrans1 - v1);
        double dv2 = Math.Abs(v2 - vTrans2);
        double totalDv = dv1 + dv2;
        double tof = Math.PI * Math.Sqrt(Math.Pow(aTrans, 3) / mu);

        return (dv1, dv2, totalDv, tof);
    }

    /// <summary>
    /// Converts WGS84 Geodetic Coordinates (lat, lon in degrees, altitude in km) to ECEF Cartesian Coordinates (X, Y, Z in km).
    /// </summary>
    public static NDArray<double> GeodeticToEcef(double latDeg, double lonDeg, double altKm)
    {
        double latRad = latDeg * Math.PI / 180.0;
        double lonRad = lonDeg * Math.PI / 180.0;

        double sinLat = Math.Sin(latRad);
        double cosLat = Math.Cos(latRad);
        double sinLon = Math.Sin(lonRad);
        double cosLon = Math.Cos(lonRad);

        double n = Wgs84A / Math.Sqrt(1.0 - Wgs84E2 * sinLat * sinLat);

        double x = (n + altKm) * cosLat * cosLon;
        double y = (n + altKm) * cosLat * sinLon;
        double z = (n * (1.0 - Wgs84E2) + altKm) * sinLat;

        return NDArray<double>.FromArray([x, y, z], 3);
    }

    /// <summary>
    /// Converts ECEF Cartesian Coordinates (X, Y, Z in km) to WGS84 Geodetic Coordinates (lat, lon in degrees, altitude in km) via Bowring's method.
    /// </summary>
    public static (double LatDeg, double LonDeg, double AltKm) EcefToGeodetic(double x, double y, double z)
    {
        double p = Math.Sqrt(x * x + y * y);
        double theta = Math.Atan2(z * Wgs84A, p * Wgs84B);
        double ePrime2 = (Wgs84A * Wgs84A - Wgs84B * Wgs84B) / (Wgs84B * Wgs84B);

        double lat = Math.Atan2(
            z + ePrime2 * Wgs84B * Math.Pow(Math.Sin(theta), 3),
            p - Wgs84E2 * Wgs84A * Math.Pow(Math.Cos(theta), 3)
        );

        double lon = Math.Atan2(y, x);

        double sinLat = Math.Sin(lat);
        double cosLat = Math.Cos(lat);
        double n = Wgs84A / Math.Sqrt(1.0 - Wgs84E2 * sinLat * sinLat);
        double alt = p / cosLat - n;

        return (lat * 180.0 / Math.PI, lon * 180.0 / Math.PI, alt);
    }

    #region Advanced Astrodynamics (Lambert, Gibbs, Bi-Elliptic, J2)

    /// <summary>
    /// J2 Perturbation Earth gravitational potential acceleration step.
    /// J2 = 1.08263e-3.
    /// </summary>
    public static NDArray<double> J2Perturbation(NDArray<double> r, double mu = EarthMu, double re = Wgs84A)
    {
        double j2 = 1.08263e-3;
        double x = r[0], y = r[1], z = r[2];
        double rMag = Math.Sqrt(x * x + y * y + z * z);
        double rSq = rMag * rMag;
        double zSq = z * z;

        double factor = (1.5 * j2 * mu * re * re) / Math.Pow(rMag, 5);
        double ax = factor * x * (5.0 * zSq / rSq - 1.0);
        double ay = factor * y * (5.0 * zSq / rSq - 1.0);
        double az = factor * z * (5.0 * zSq / rSq - 3.0);

        return NDArray<double>.FromArray([ax, ay, az], 3);
    }

    /// <summary>
    /// Computes Bi-Elliptic transfer delta-v between initial circular orbit r1 and final circular orbit r2 via intermediate apogee rB.
    /// </summary>
    public static (double Dv1, double Dv2, double Dv3, double TotalDv) BiEllipticTransfer(
        double r1, double rB, double r2, double mu = EarthMu)
    {
        double v1 = Math.Sqrt(mu / r1);
        double v2 = Math.Sqrt(mu / r2);

        double a1 = (r1 + rB) / 2.0;
        double a2 = (rB + r2) / 2.0;

        double vA1 = Math.Sqrt(mu * (2.0 / r1 - 1.0 / a1));
        double vB1 = Math.Sqrt(mu * (2.0 / rB - 1.0 / a1));
        double vB2 = Math.Sqrt(mu * (2.0 / rB - 1.0 / a2));
        double vA2 = Math.Sqrt(mu * (2.0 / r2 - 1.0 / a2));

        double dv1 = Math.Abs(vA1 - v1);
        double dv2 = Math.Abs(vB2 - vB1);
        double dv3 = Math.Abs(v2 - vA2);

        return (dv1, dv2, dv3, dv1 + dv2 + dv3);
    }

    /// <summary>
    /// Gibbs Orbit Determination: computes velocity vector v2 at position r2 from 3 coplanar position vectors r1, r2, r3.
    /// </summary>
    public static NDArray<double> GibbsOrbitDetermination(
        NDArray<double> r1, NDArray<double> r2, NDArray<double> r3, double mu = EarthMu)
    {
        double r1Mag = Math.Sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]);
        double r2Mag = Math.Sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]);
        double r3Mag = Math.Sqrt(r3[0] * r3[0] + r3[1] * r3[1] + r3[2] * r3[2]);

        // Cross products: c12 = r1 x r2, c23 = r2 x r3, c31 = r3 x r1
        double[] Cross(NDArray<double> a, NDArray<double> b) =>
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

        double[] c12 = Cross(r1, r2);
        double[] c23 = Cross(r2, r3);
        double[] c31 = Cross(r3, r1);

        double[] N = [
            r1Mag * c23[0] + r2Mag * c31[0] + r3Mag * c12[0],
            r1Mag * c23[1] + r2Mag * c31[1] + r3Mag * c12[1],
            r1Mag * c23[2] + r2Mag * c31[2] + r3Mag * c12[2]
        ];

        double[] D = [c12[0] + c23[0] + c31[0], c12[1] + c23[1] + c31[1], c12[2] + c23[2] + c31[2]];

        double nMag = Math.Sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
        double dMag = Math.Sqrt(D[0] * D[0] + D[1] * D[1] + D[2] * D[2]);

        double[] dCrossR2 = [
            D[1] * r2[2] - D[2] * r2[1],
            D[2] * r2[0] - D[0] * r2[2],
            D[0] * r2[1] - D[1] * r2[0]
        ];

        double factor = Math.Sqrt(mu / (nMag * dMag));
        double[] v2 = [
            factor * ((dCrossR2[0] / r2Mag) + (r2[0] * (N[0] / nMag))),
            factor * ((dCrossR2[1] / r2Mag) + (r2[1] * (N[1] / nMag))),
            factor * ((dCrossR2[2] / r2Mag) + (r2[2] * (N[2] / nMag)))
        ];

        return NDArray<double>.FromArray(v2, 3);
    }

    /// <summary>
    /// Converts ECI (J2000) position vector to ECEF (ITRF) given Greenwich Mean Sidereal Time (GMST in radians).
    /// </summary>
    public static NDArray<double> ECI_To_ECEF(NDArray<double> rECI, double gmstRad)
    {
        double cosG = Math.Cos(gmstRad);
        double sinG = Math.Sin(gmstRad);

        double x = cosG * rECI[0] + sinG * rECI[1];
        double y = -sinG * rECI[0] + cosG * rECI[1];
        double z = rECI[2];

        return NDArray<double>.FromArray([x, y, z], 3);
    }

    #endregion
}

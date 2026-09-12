// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Physics;

/// <summary>
/// Provides 2D Lattice Boltzmann Method (LBM D2Q9) incompressible fluid dynamics simulation.
/// </summary>
public static class LatticeBoltzmann2D
{
    // D2Q9 Discrete Velocity Directions: (cx, cy)
    private static readonly int[] Cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
    private static readonly int[] Cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
    // D2Q9 Lattice Weights
    private static readonly double[] W = [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0];
    // Opposite directions for bounce-back boundary
    private static readonly int[] Opp = [0, 3, 4, 1, 2, 7, 8, 5, 6];

    /// <summary>
    /// Initializes a D2Q9 distribution function tensor [Ny, Nx, 9] with equilibrium values.
    /// </summary>
    public static NDArray<double> InitializeEquilibrium(int ny, int nx, double rho0 = 1.0, double u0 = 0.0, double v0 = 0.0)
    {
        var f = new NDArray<double>(ny, nx, 9);
        double uSq = u0 * u0 + v0 * v0;
        for (int y = 0; y < ny; y++)
        {
            for (int x = 0; x < nx; x++)
            {
                for (int i = 0; i < 9; i++)
                {
                    double cu = Cx[i] * u0 + Cy[i] * v0;
                    f[y, x, i] = rho0 * W[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uSq);
                }
            }
        }
        return f;
    }

    /// <summary>
    /// Executes one single time-step of D2Q9 Lattice Boltzmann simulation with BGK collision and streaming.
    /// F: distribution functions shape [Ny, Nx, 9], Tau: relaxation time.
    /// Returns macroscopic Density [Ny, Nx], Velocity U [Ny, Nx], and Velocity V [Ny, Nx].
    /// </summary>
    public static (NDArray<double> Rho, NDArray<double> U, NDArray<double> V, NDArray<double> FNext) Step(
        NDArray<double> f, double tau = 0.6, BoolNDArray? obstacleMask = null)
    {
        int ny = f.Shape[0];
        int nx = f.Shape[1];

        var rho = new NDArray<double>(ny, nx);
        var u = new NDArray<double>(ny, nx);
        var v = new NDArray<double>(ny, nx);
        var fColl = new NDArray<double>(ny, nx, 9);
        var fNext = new NDArray<double>(ny, nx, 9);

        double invTau = 1.0 / tau;

        // 1. Compute macroscopic variables & Collision step
        Parallel.For(0, ny, y =>
        {
            for (int x = 0; x < nx; x++)
            {
                if (obstacleMask != null && obstacleMask[y, x])
                {
                    continue;
                }

                double density = 0.0;
                double ux = 0.0, vy = 0.0;

                for (int i = 0; i < 9; i++)
                {
                    double fi = f[y, x, i];
                    density += fi;
                    ux += fi * Cx[i];
                    vy += fi * Cy[i];
                }

                if (density > 1e-12)
                {
                    ux /= density;
                    vy /= density;
                }

                rho[y, x] = density;
                u[y, x] = ux;
                v[y, x] = vy;

                double uSq = ux * ux + vy * vy;

                // BGK Equilibrium distribution f_eq
                for (int i = 0; i < 9; i++)
                {
                    double cu = Cx[i] * ux + Cy[i] * vy;
                    double feq = density * W[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uSq);
                    fColl[y, x, i] = f[y, x, i] - invTau * (f[y, x, i] - feq);
                }
            }
        });

        // 2. Streaming & Bounce-Back boundary condition
        Parallel.For(0, ny, y =>
        {
            for (int x = 0; x < nx; x++)
            {
                for (int i = 0; i < 9; i++)
                {
                    int prevX = (x - Cx[i] + nx) % nx;
                    int prevY = (y - Cy[i] + ny) % ny;

                    if (obstacleMask != null && obstacleMask[y, x])
                    {
                        // Bounce-back on obstacle
                        fNext[y, x, i] = fColl[y, x, Opp[i]];
                    }
                    else
                    {
                        fNext[y, x, i] = fColl[prevY, prevX, i];
                    }
                }
            }
        });

        return (rho, u, v, fNext);
    }
}

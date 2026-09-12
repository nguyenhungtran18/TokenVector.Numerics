// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Physics;

/// <summary>
/// Provides computational physics engines: Runge-Kutta 4th Order ODE integrator, N-Body Gravitational Symplectic Verlet Integrator, and 3D Vector Calculus (Gradient, Divergence, Curl, Laplacian).
/// </summary>
public static class PhysicsODEAndFields
{
    #region ODE Solvers (Runge-Kutta 4th Order)

    /// <summary>
    /// Integrates a system of ordinary differential equations dy/dt = f(t, y) from t0 to tEnd using classical Runge-Kutta 4th order (RK4).
    /// Returns an array of state vectors at each time step.
    /// </summary>
    public static (double[] Timestamps, NDArray<double>[] Trajectory) SolveRK4(
        Func<double, NDArray<double>, NDArray<double>> f,
        double t0, double tEnd, NDArray<double> y0, int numSteps)
    {
        if (numSteps <= 0) throw new ArgumentOutOfRangeException(nameof(numSteps));

        double dt = (tEnd - t0) / numSteps;
        var timestamps = new double[numSteps + 1];
        var trajectory = new NDArray<double>[numSteps + 1];

        timestamps[0] = t0;
        trajectory[0] = y0.Clone();

        double t = t0;
        var y = y0.Clone();

        for (int i = 0; i < numSteps; i++)
        {
            var k1 = f(t, y);
            var k2 = f(t + 0.5 * dt, y + k1 * (0.5 * dt));
            var k3 = f(t + 0.5 * dt, y + k2 * (0.5 * dt));
            var k4 = f(t + dt, y + k3 * dt);

            // y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            var delta = (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
            y = y + delta;
            t += dt;

            timestamps[i + 1] = t;
            trajectory[i + 1] = y.Clone();
        }

        return (timestamps, trajectory);
    }

    #endregion

    #region N-Body Gravitational Dynamics (Symplectic Verlet)

    /// <summary>
    /// Advances an N-body gravitational system by time step dt using Symplectic Velocity-Verlet integration.
    /// Positions: [N, 3], Velocities: [N, 3], Masses: [N].
    /// </summary>
    public static (NDArray<double> NewPositions, NDArray<double> NewVelocities) NBodyVerletStep(
        NDArray<double> positions, NDArray<double> velocities, NDArray<double> masses,
        double dt, double g = 1.0, double softening = 1e-4)
    {
        int n = masses.TotalLength;
        var newPos = positions.Clone();
        var newVel = velocities.Clone();

        var acc0 = ComputeGravitationalAccelerations(newPos, masses, g, softening);

        // 1. Position update: r_{n+1} = r_n + v_n * dt + 0.5 * a_n * dt^2
        Parallel.For(0, n, i =>
        {
            for (int d = 0; d < 3; d++)
            {
                newPos[i, d] += newVel[i, d] * dt + 0.5 * acc0[i, d] * dt * dt;
            }
        });

        // 2. Compute new accelerations a_{n+1}
        var acc1 = ComputeGravitationalAccelerations(newPos, masses, g, softening);

        // 3. Velocity update: v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
        Parallel.For(0, n, i =>
        {
            for (int d = 0; d < 3; d++)
            {
                newVel[i, d] += 0.5 * (acc0[i, d] + acc1[i, d]) * dt;
            }
        });

        return (newPos, newVel);
    }

    private static NDArray<double> ComputeGravitationalAccelerations(
        NDArray<double> pos, NDArray<double> masses, double g, double softening)
    {
        int n = masses.TotalLength;
        var acc = new NDArray<double>(n, 3);
        double epsSq = softening * softening;

        Parallel.For(0, n, i =>
        {
            double ax = 0.0, ay = 0.0, az = 0.0;
            double xi = pos[i, 0], yi = pos[i, 1], zi = pos[i, 2];

            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                double dx = pos[j, 0] - xi;
                double dy = pos[j, 1] - yi;
                double dz = pos[j, 2] - zi;

                double rSq = dx * dx + dy * dy + dz * dz + epsSq;
                double invR3 = 1.0 / (rSq * Math.Sqrt(rSq));
                double forceFactor = g * masses[j] * invR3;

                ax += forceFactor * dx;
                ay += forceFactor * dy;
                az += forceFactor * dz;
            }

            acc[i, 0] = ax;
            acc[i, 1] = ay;
            acc[i, 2] = az;
        });

        return acc;
    }

    #endregion

    #region Vector Calculus Fields (3D Gradient, Divergence, Curl, Laplacian)

    /// <summary>
    /// Computes 3D Gradient (\nabla f) of a scalar field of shape [Nx, Ny, Nz].
    /// Returns 3 scalar fields (dF/dx, dF/dy, dF/dz).
    /// </summary>
    public static (NDArray<double> GradX, NDArray<double> GradY, NDArray<double> GradZ) Gradient3D(
        NDArray<double> f, double dx = 1.0, double dy = 1.0, double dz = 1.0)
    {
        int nx = f.Shape[0], ny = f.Shape[1], nz = f.Shape[2];
        var gx = new NDArray<double>(nx, ny, nz);
        var gy = new NDArray<double>(nx, ny, nz);
        var gz = new NDArray<double>(nx, ny, nz);

        Parallel.For(0, nx, i =>
        {
            int iPrev = Math.Max(0, i - 1), iNext = Math.Min(nx - 1, i + 1);
            double hx = (iNext - iPrev) * dx;

            for (int j = 0; j < ny; j++)
            {
                int jPrev = Math.Max(0, j - 1), jNext = Math.Min(ny - 1, j + 1);
                double hy = (jNext - jPrev) * dy;

                for (int k = 0; k < nz; k++)
                {
                    int kPrev = Math.Max(0, k - 1), kNext = Math.Min(nz - 1, k + 1);
                    double hz = (kNext - kPrev) * dz;

                    gx[i, j, k] = (f[iNext, j, k] - f[iPrev, j, k]) / hx;
                    gy[i, j, k] = (f[i, jNext, k] - f[i, jPrev, k]) / hy;
                    gz[i, j, k] = (f[i, j, kNext] - f[i, j, kPrev]) / hz;
                }
            }
        });

        return (gx, gy, gz);
    }

    /// <summary>
    /// Computes 3D Divergence (\nabla \cdot F) of a vector field F = (fx, fy, fz).
    /// </summary>
    public static NDArray<double> Divergence3D(
        NDArray<double> fx, NDArray<double> fy, NDArray<double> fz,
        double dx = 1.0, double dy = 1.0, double dz = 1.0)
    {
        var (dfx_dx, _, _) = Gradient3D(fx, dx, dy, dz);
        var (_, dfy_dy, _) = Gradient3D(fy, dx, dy, dz);
        var (_, _, dfz_dz) = Gradient3D(fz, dx, dy, dz);

        return dfx_dx + dfy_dy + dfz_dz;
    }

    /// <summary>
    /// Computes 3D Curl (\nabla \times F) of a vector field F = (fx, fy, fz).
    /// </summary>
    public static (NDArray<double> CurlX, NDArray<double> CurlY, NDArray<double> CurlZ) Curl3D(
        NDArray<double> fx, NDArray<double> fy, NDArray<double> fz,
        double dx = 1.0, double dy = 1.0, double dz = 1.0)
    {
        var (_, dfx_dy, dfx_dz) = Gradient3D(fx, dx, dy, dz);
        var (dfy_dx, _, dfy_dz) = Gradient3D(fy, dx, dy, dz);
        var (dfz_dx, dfz_dy, _) = Gradient3D(fz, dx, dy, dz);

        var curlX = dfz_dy - dfy_dz;
        var curlY = dfx_dz - dfz_dx;
        var curlZ = dfy_dx - dfx_dy;

        return (curlX, curlY, curlZ);
    }

    /// <summary>
    /// Computes 3D Laplacian (\nabla^2 f = \nabla \cdot \nabla f) of a scalar field.
    /// </summary>
    public static NDArray<double> Laplacian3D(NDArray<double> f, double dx = 1.0, double dy = 1.0, double dz = 1.0)
    {
        var (gx, gy, gz) = Gradient3D(f, dx, dy, dz);
        return Divergence3D(gx, gy, gz, dx, dy, dz);
    }

    #endregion
}

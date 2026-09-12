using System;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Robotics;

/// <summary>
/// Frontier robotics, optimal control, and kinematic dynamics algorithms.
/// Includes Discrete and Continuous Algebraic Riccati Equations (LQR) and Damped Least Squares (DLS) Inverse Kinematics.
/// </summary>
public static class OptimalControlAndRobotics
{
    /// <summary>
    /// Solves the Discrete-time Linear Quadratic Regulator (LQR) problem.
    /// Minimizes sum of (x' Q x + u' R u) subject to x_{k+1} = A x_k + B u_k.
    /// Returns the optimal feedback gain matrix K such that u = -K x, and the Riccati matrix P.
    /// </summary>
    public static (NDArray<double> K, NDArray<double> P) SolveDiscreteLQR(
        NDArray<double> A,
        NDArray<double> B,
        NDArray<double> Q,
        NDArray<double> R,
        int maxIter = 500,
        double tolerance = 1e-9)
    {
        if (A.Rank != 2 || B.Rank != 2 || Q.Rank != 2 || R.Rank != 2)
            throw new ArgumentException("Matrices A, B, Q, R must be 2D.");

        int n = A.Shape[0];
        int m = B.Shape[1];

        NDArray<double> P = Q.Clone();
        NDArray<double> At = A.Transpose();
        NDArray<double> Bt = B.Transpose();

        for (int iter = 0; iter < maxIter; iter++)
        {
            // Term: R + B' P B
            NDArray<double> BtP = MatrixMultiplication.MatMul(Bt, P);
            NDArray<double> BtPB = MatrixMultiplication.MatMul(BtP, B);
            NDArray<double> R_BtPB = BtPB + R;

            // Invert (R + B' P B)
            NDArray<double> invR_BtPB = Decomposition.Inverse(R_BtPB);

            // Term: A' P B
            NDArray<double> AtP = MatrixMultiplication.MatMul(At, P);
            NDArray<double> AtPB = MatrixMultiplication.MatMul(AtP, B);

            // Term: A' P A
            NDArray<double> AtPA = MatrixMultiplication.MatMul(AtP, A);

            // Term: AtPB * inv(R + BtPB) * (Bt P A)
            NDArray<double> BtPA = MatrixMultiplication.MatMul(BtP, A);
            NDArray<double> step = MatrixMultiplication.MatMul(AtPB, MatrixMultiplication.MatMul(invR_BtPB, BtPA));

            NDArray<double> PNext = Q + (AtPA - step);

            // Check convergence ||PNext - P||_F
            double diff = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double d = PNext[i, j] - P[i, j];
                    diff += d * d;
                }
            }

            P = PNext;
            if (Math.Sqrt(diff) < tolerance)
                break;
        }

        // K = (R + B' P B)^-1 * (B' P A)
        NDArray<double> BtP_final = MatrixMultiplication.MatMul(Bt, P);
        NDArray<double> R_BtPB_final = MatrixMultiplication.MatMul(BtP_final, B) + R;
        NDArray<double> inv_final = Decomposition.Inverse(R_BtPB_final);
        NDArray<double> BtPA_final = MatrixMultiplication.MatMul(BtP_final, A);
        NDArray<double> K = MatrixMultiplication.MatMul(inv_final, BtPA_final);

        return (K, P);
    }

    /// <summary>
    /// Inverse Kinematics step via Damped Least Squares (DLS / Levenberg-Marquardt IK):
    /// \Delta \theta = J^T (J J^T + \lambda^2 I)^{-1} \Delta x
    /// Prevents singularities and numerical instability in robotic arm manipulability.
    /// </summary>
    /// <param name="jacobian">Robotic Jacobian matrix (m x n).</param>
    /// <param name="deltaX">End-effector task space Cartesian error / velocity (m x 1 or 1D array length m).</param>
    /// <param name="lambda">Damping factor (lambda > 0).</param>
    /// <returns>Joint angle velocity / step Delta theta (n x 1 or 1D).</returns>
    public static NDArray<double> JacobianDLS(NDArray<double> jacobian, NDArray<double> deltaX, double lambda = 0.01)
    {
        if (jacobian.Rank != 2)
            throw new ArgumentException("Jacobian must be a 2D matrix.");

        int m = jacobian.Shape[0];
        int n = jacobian.Shape[1];

        NDArray<double> Jt = jacobian.Transpose();
        NDArray<double> JJt = MatrixMultiplication.MatMul(jacobian, Jt);

        // Add damping lambda^2 * I
        double lambdaSq = lambda * lambda;
        for (int i = 0; i < m; i++)
        {
            JJt[i, i] += lambdaSq;
        }

        NDArray<double> dx2D = deltaX.Rank == 1 ? deltaX.Reshape(m, 1) : deltaX;
        
        // Solve (JJ^T + \lambda^2 I) y = \Delta x
        NDArray<double> y = Decomposition.Solve(JJt, dx2D);

        // \Delta \theta = J^T * y
        NDArray<double> dTheta = MatrixMultiplication.MatMul(Jt, y);

        return deltaX.Rank == 1 ? dTheta.Reshape(n) : dTheta;
    }

    /// <summary>
    /// 2D Unicycle / Differential Drive mobile robot forward kinematics step.
    /// </summary>
    public static (double x, double y, double theta) UnicycleKinematics(
        double x, double y, double theta, double v, double omega, double dt)
    {
        if (Math.Abs(omega) < 1e-9)
        {
            return (x + v * Math.Cos(theta) * dt, y + v * Math.Sin(theta) * dt, theta);
        }
        else
        {
            double thetaNext = theta + omega * dt;
            double xNext = x + (v / omega) * (Math.Sin(thetaNext) - Math.Sin(theta));
            double yNext = y - (v / omega) * (Math.Cos(thetaNext) - Math.Cos(theta));
            return (xNext, yNext, thetaNext);
        }
    }
}

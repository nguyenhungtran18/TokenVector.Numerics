// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Optimize;

/// <summary>
/// Numerical Optimization and Root Finding algorithms (Brent 1D, Nelder-Mead Simplex, BFGS Quasi-Newton, Brentq Roots, Levenberg-Marquardt CurveFit, Simplex LP).
/// </summary>
public static class OptimizationAndRootFinding
{
    #region 1D Root Finding (Brentq & Bisection)

    /// <summary>
    /// Finds a zero of a continuous function f in the interval [a, b] using Brent's method (brentq).
    /// </summary>
    public static double RootBrentq(Func<double, double> f, double a, double b, double tol = 1e-9, int maxIter = 100)
    {
        double fa = f(a);
        double fb = f(b);

        if (fa * fb > 0)
            throw new ArgumentException("Root must be bracketed: f(a) and f(b) must have opposite signs.");

        if (Math.Abs(fa) < tol) return a;
        if (Math.Abs(fb) < tol) return b;

        double c = a, fc = fa;
        double d = b - a, e = d;

        for (int iter = 0; iter < maxIter; iter++)
        {
            if (fb * fc > 0)
            {
                c = a; fc = fa;
                d = b - a; e = d;
            }

            if (Math.Abs(fc) < Math.Abs(fb))
            {
                a = b; b = c; c = a;
                fa = fb; fb = fc; fc = fa;
            }

            double tol1 = 2.0 * 1e-15 * Math.Abs(b) + 0.5 * tol;
            double xm = 0.5 * (c - b);

            if (Math.Abs(xm) <= tol1 || Math.Abs(fb) < tol)
                return b;

            if (Math.Abs(e) >= tol1 && Math.Abs(fa) > Math.Abs(fb))
            {
                double s = fb / fa;
                double p, q;

                if (Math.Abs(a - c) < 1e-15)
                {
                    // Linear interpolation
                    p = 2.0 * xm * s;
                    q = 1.0 - s;
                }
                else
                {
                    // Inverse quadratic interpolation
                    q = fa / fc;
                    double r = fb / fc;
                    p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                }

                if (p > 0) q = -q;
                p = Math.Abs(p);

                if (2.0 * p < Math.Min(3.0 * xm * q - Math.Abs(tol1 * q), Math.Abs(e * q)))
                {
                    e = d;
                    d = p / q;
                }
                else
                {
                    d = xm;
                    e = d;
                }
            }
            else
            {
                d = xm;
                e = d;
            }

            a = b; fa = fb;
            if (Math.Abs(d) > tol1) b += d;
            else b += (xm >= 0 ? tol1 : -tol1);
            fb = f(b);
        }

        return b;
    }

    /// <summary>
    /// Bisection method to find root in [a, b].
    /// </summary>
    public static double Bisection(Func<double, double> f, double a, double b, double tol = 1e-9, int maxIter = 100)
    {
        double fa = f(a);
        double fb = f(b);
        if (fa * fb > 0) throw new ArgumentException("Root must be bracketed.");

        for (int i = 0; i < maxIter; i++)
        {
            double mid = 0.5 * (a + b);
            double fMid = f(mid);

            if (Math.Abs(fMid) < tol || (b - a) * 0.5 < tol)
                return mid;

            if (fa * fMid < 0)
            {
                b = mid;
                fb = fMid;
            }
            else
            {
                a = mid;
                fa = fMid;
            }
        }
        return 0.5 * (a + b);
    }

    #endregion

    #region 1D Minimization (Brent)

    /// <summary>
    /// Finds local minimum of 1D scalar function in [a, b] using Brent's golden section + parabolic interpolation.
    /// </summary>
    public static (double XMin, double FMin) MinimizeBrent(Func<double, double> f, double a, double b, double tol = 1e-6, int maxIter = 100)
    {
        double gold = 0.3819660112501051; // (3 - sqrt(5)) / 2
        double x = a + gold * (b - a);
        double w = x, v = x;
        double fx = f(x), fw = fx, fv = fx;
        double d = 0, e = 0;

        for (int iter = 0; iter < maxIter; iter++)
        {
            double mid = 0.5 * (a + b);
            double tol1 = 1e-10 * Math.Abs(x) + tol;
            double tol2 = 2.0 * tol1;

            if (Math.Abs(x - mid) <= (tol2 - 0.5 * (b - a)))
                break;

            double p = 0, q = 0, r = 0;
            if (Math.Abs(e) > tol1)
            {
                r = (x - w) * (fx - fv);
                q = (x - v) * (fx - fw);
                p = (x - v) * q - (x - w) * r;
                q = 2.0 * (q - r);
                if (q > 0) p = -p;
                q = Math.Abs(q);
                double etemp = e;
                e = d;

                if (Math.Abs(p) < Math.Abs(0.5 * q * etemp) && p > q * (a - x) && p < q * (b - x))
                {
                    d = p / q;
                    double u = x + d;
                    if (u - a < tol2 || b - u < tol2)
                        d = (mid - x >= 0) ? tol1 : -tol1;
                }
                else
                {
                    e = (x >= mid) ? a - x : b - x;
                    d = gold * e;
                }
            }
            else
            {
                e = (x >= mid) ? a - x : b - x;
                d = gold * e;
            }

            double uNew = Math.Abs(d) >= tol1 ? x + d : x + (d >= 0 ? tol1 : -tol1);
            double fu = f(uNew);

            if (fu <= fx)
            {
                if (uNew >= x) a = x; else b = x;
                v = w; fv = fw;
                w = x; fw = fx;
                x = uNew; fx = fu;
            }
            else
            {
                if (uNew < x) a = uNew; else b = uNew;
                if (fu <= fw || Math.Abs(w - x) < 1e-15)
                {
                    v = w; fv = fw;
                    w = uNew; fw = fu;
                }
                else if (fu <= fv || Math.Abs(v - x) < 1e-15 || Math.Abs(v - w) < 1e-15)
                {
                    v = uNew; fv = fu;
                }
            }
        }

        return (x, fx);
    }

    #endregion

    #region Multidimensional Unconstrained Optimization (Nelder-Mead & BFGS)

    /// <summary>
    /// Nelder-Mead Downhill Simplex algorithm for multidimensional unconstrained optimization without derivatives.
    /// </summary>
    public static (NDArray<double> XMin, double FMin) MinimizeNelderMead(
        Func<NDArray<double>, double> f, NDArray<double> x0, double tol = 1e-6, int maxIter = 1000)
    {
        int n = x0.TotalLength;
        // Simplex with n+1 vertices
        var simplex = new NDArray<double>[n + 1];
        var fVals = new double[n + 1];

        simplex[0] = x0.Clone();
        fVals[0] = f(simplex[0]);

        for (int i = 0; i < n; i++)
        {
            var p = x0.Clone();
            p[i] += (Math.Abs(p[i]) > 1e-4) ? 0.05 * p[i] : 0.00025;
            simplex[i + 1] = p;
            fVals[i + 1] = f(p);
        }

        double alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5;

        for (int iter = 0; iter < maxIter; iter++)
        {
            // Sort vertices by objective value
            int[] order = new int[n + 1];
            for (int i = 0; i <= n; i++) order[i] = i;
            Array.Sort(order, (i1, i2) => fVals[i1].CompareTo(fVals[i2]));

            int best = order[0];
            int worst = order[n];
            int secondWorst = order[n - 1];

            // Check tolerance: range of function values in simplex
            double fRange = Math.Abs(fVals[worst] - fVals[best]);
            if (fRange < tol) break;

            // Centroid of all vertices except worst
            var centroid = new NDArray<double>(n);
            for (int i = 0; i < n; i++)
            {
                int idx = order[i];
                for (int d = 0; d < n; d++) centroid[d] += simplex[idx][d];
            }
            for (int d = 0; d < n; d++) centroid[d] /= n;

            // 1. Reflection
            var xr = new NDArray<double>(n);
            for (int d = 0; d < n; d++) xr[d] = centroid[d] + alpha * (centroid[d] - simplex[worst][d]);
            double fxr = f(xr);

            if (fxr >= fVals[best] && fxr < fVals[secondWorst])
            {
                simplex[worst] = xr;
                fVals[worst] = fxr;
                continue;
            }

            // 2. Expansion
            if (fxr < fVals[best])
            {
                var xe = new NDArray<double>(n);
                for (int d = 0; d < n; d++) xe[d] = centroid[d] + gamma * (xr[d] - centroid[d]);
                double fxe = f(xe);
                if (fxe < fxr)
                {
                    simplex[worst] = xe;
                    fVals[worst] = fxe;
                }
                else
                {
                    simplex[worst] = xr;
                    fVals[worst] = fxr;
                }
                continue;
            }

            // 3. Contraction
            var xc = new NDArray<double>(n);
            for (int d = 0; d < n; d++) xc[d] = centroid[d] + rho * (simplex[worst][d] - centroid[d]);
            double fxc = f(xc);

            if (fxc < fVals[worst])
            {
                simplex[worst] = xc;
                fVals[worst] = fxc;
                continue;
            }

            // 4. Shrink
            for (int i = 1; i <= n; i++)
            {
                int idx = order[i];
                for (int d = 0; d < n; d++)
                {
                    simplex[idx][d] = simplex[best][d] + sigma * (simplex[idx][d] - simplex[best][d]);
                }
                fVals[idx] = f(simplex[idx]);
            }
        }

        int minIdx = 0;
        for (int i = 1; i <= n; i++) if (fVals[i] < fVals[minIdx]) minIdx = i;

        return (simplex[minIdx], fVals[minIdx]);
    }

    /// <summary>
    /// BFGS Quasi-Newton optimization algorithm.
    /// </summary>
    public static (NDArray<double> XMin, double FMin) MinimizeBFGS(
        Func<NDArray<double>, double> f, Func<NDArray<double>, NDArray<double>>? grad, NDArray<double> x0, double tol = 1e-6, int maxIter = 200)
    {
        int n = x0.TotalLength;
        var x = x0.Clone();
        var H = NDArray<double>.Eye(n); // Approximate inverse Hessian

        grad ??= (pt) => ApproximateGradient(f, pt);

        var g = grad(x);

        for (int iter = 0; iter < maxIter; iter++)
        {
            if (MatrixOps.Norm(g, "2") < tol) break;

            // Search direction p = -H * g
            var p = MatrixMultiplication.MatMul(H, g.Reshape(n, 1)).Reshape(n) * (-1.0);

            // Backtracking line search (Armijo rule)
            double alpha = 1.0;
            double c1 = 1e-4;
            double fx = f(x);
            double slope = 0;
            for (int i = 0; i < n; i++) slope += g[i] * p[i];

            while (f(x + p * alpha) > fx + c1 * alpha * slope && alpha > 1e-8)
            {
                alpha *= 0.5;
            }

            var s = p * alpha;
            var xNext = x + s;
            var gNext = grad(xNext);
            var y = gNext - g;

            // Update H using BFGS formula
            double ys = 0;
            for (int i = 0; i < n; i++) ys += y[i] * s[i];

            if (Math.Abs(ys) > 1e-14)
            {
                double rho = 1.0 / ys;
                var I = NDArray<double>.Eye(n);
                var syT = MatrixOps.Outer(s, y);
                var yTs = MatrixOps.Outer(y, s);

                var term1 = I - syT * rho;
                var term2 = I - yTs * rho;
                var ssT = MatrixOps.Outer(s, s) * rho;

                H = MatrixMultiplication.MatMul(MatrixMultiplication.MatMul(term1, H), term2) + ssT;
            }

            x = xNext;
            g = gNext;
        }

        return (x, f(x));
    }

    private static NDArray<double> ApproximateGradient(Func<NDArray<double>, double> f, NDArray<double> x, double eps = 1e-7)
    {
        int n = x.TotalLength;
        var g = new NDArray<double>(n);
        var xp = x.Clone();

        for (int i = 0; i < n; i++)
        {
            double orig = xp[i];
            xp[i] = orig + eps;
            double fPlus = f(xp);
            xp[i] = orig - eps;
            double fMinus = f(xp);
            xp[i] = orig;

            g[i] = (fPlus - fMinus) / (2.0 * eps);
        }
        return g;
    }

    #endregion

    #region Linear Programming (Simplex)

    /// <summary>
    /// Solves linear program: Minimize c^T x subject to A x &lt;= b, x &gt;= 0.
    /// Returns optimal x and optimal objective value.
    /// </summary>
    public static (NDArray<double> XOpt, double MinValue) LinearProgramSimplex(
        NDArray<double> c, NDArray<double> A, NDArray<double> b, int maxIter = 500)
    {
        int m = A.Shape[0]; // Number of constraints
        int n = A.Shape[1]; // Number of variables

        // Tableau: (m + 1) x (n + m + 1)
        var tab = new NDArray<double>(m + 1, n + m + 1);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++) tab[i, j] = A[i, j];
            tab[i, n + i] = 1.0; // Slack variable
            tab[i, n + m] = b[i]; // RHS
        }

        // Objective row: -c (for minimization)
        for (int j = 0; j < n; j++) tab[m, j] = -c[j];

        int[] basis = new int[m];
        for (int i = 0; i < m; i++) basis[i] = n + i;

        for (int iter = 0; iter < maxIter; iter++)
        {
            // Find entering column (most negative in bottom row)
            int enterCol = -1;
            double minCoeff = -1e-9;
            for (int j = 0; j < n + m; j++)
            {
                if (tab[m, j] < minCoeff)
                {
                    minCoeff = tab[m, j];
                    enterCol = j;
                }
            }

            if (enterCol == -1) break; // Optimal!

            // Find leaving row (minimum positive ratio test)
            int leaveRow = -1;
            double minRatio = double.MaxValue;
            for (int i = 0; i < m; i++)
            {
                if (tab[i, enterCol] > 1e-12)
                {
                    double ratio = tab[i, n + m] / tab[i, enterCol];
                    if (ratio < minRatio)
                    {
                        minRatio = ratio;
                        leaveRow = i;
                    }
                }
            }

            if (leaveRow == -1) throw new InvalidOperationException("Linear program is unbounded.");

            // Pivot on (leaveRow, enterCol)
            basis[leaveRow] = enterCol;
            double pivot = tab[leaveRow, enterCol];
            for (int j = 0; j <= n + m; j++) tab[leaveRow, j] /= pivot;

            for (int i = 0; i <= m; i++)
            {
                if (i != leaveRow)
                {
                    double factor = tab[i, enterCol];
                    for (int j = 0; j <= n + m; j++) tab[i, j] -= factor * tab[leaveRow, j];
                }
            }
        }

        var xOpt = new NDArray<double>(n);
        for (int i = 0; i < m; i++)
        {
            if (basis[i] < n) xOpt[basis[i]] = tab[i, n + m];
        }

        double optVal = tab[m, n + m];
        return (xOpt, optVal);
    }

    #endregion

    #region Quadratic Programming (ADMM / Operator Splitting QP Solver)

    /// <summary>
    /// Solves Convex Quadratic Program:
    /// Minimize (1/2) * x^T * P * x + q^T * x
    /// Subject to: G * x &lt;= h (inequality), A * x == b (equality), lb &lt;= x &lt;= ub (box bounds).
    /// Uses Operator-Splitting ADMM (Alternating Direction Method of Multipliers) matching OSQP standard.
    /// </summary>
    public static (NDArray<double> XOpt, double MinValue) QPSolve(
        NDArray<double> P,
        NDArray<double> q,
        NDArray<double>? G = null,
        NDArray<double>? h = null,
        NDArray<double>? A = null,
        NDArray<double>? b = null,
        NDArray<double>? lb = null,
        NDArray<double>? ub = null,
        double rho = 1.0,
        double sigma = 1e-6,
        double tol = 1e-5,
        int maxIter = 1000)
    {
        if (P.Rank != 2 || P.Shape[0] != P.Shape[1])
            throw new ArgumentException("P matrix must be square 2D.", nameof(P));

        int n = P.Shape[0];
        if (q.TotalLength != n)
            throw new ArgumentException("q vector must have length matching P dimension.", nameof(q));

        // Stack all constraints into C_mat * x in [l_vec, u_vec]
        var constraintRows = new List<double[]>();
        var lList = new List<double>();
        var uList = new List<double>();

        // 1. Equality constraints A * x == b => b <= A * x <= b
        if (A != null && b != null)
        {
            int mA = A.Shape[0];
            for (int i = 0; i < mA; i++)
            {
                double[] row = new double[n];
                for (int j = 0; j < n; j++) row[j] = A[i, j];
                constraintRows.Add(row);
                lList.Add(b[i]);
                uList.Add(b[i]);
            }
        }

        // 2. Inequality constraints G * x <= h => -inf <= G * x <= h
        if (G != null && h != null)
        {
            int mG = G.Shape[0];
            for (int i = 0; i < mG; i++)
            {
                double[] row = new double[n];
                for (int j = 0; j < n; j++) row[j] = G[i, j];
                constraintRows.Add(row);
                lList.Add(double.NegativeInfinity);
                uList.Add(h[i]);
            }
        }

        // 3. Box bounds lb <= I * x <= ub
        if (lb != null || ub != null)
        {
            for (int i = 0; i < n; i++)
            {
                double[] row = new double[n];
                row[i] = 1.0;
                constraintRows.Add(row);
                lList.Add(lb != null ? lb[i] : double.NegativeInfinity);
                uList.Add(ub != null ? ub[i] : double.PositiveInfinity);
            }
        }

        // If no constraints, add identity with [-inf, inf]
        if (constraintRows.Count == 0)
        {
            for (int i = 0; i < n; i++)
            {
                double[] row = new double[n];
                row[i] = 1.0;
                constraintRows.Add(row);
                lList.Add(double.NegativeInfinity);
                uList.Add(double.PositiveInfinity);
            }
        }

        int mTotal = constraintRows.Count;
        var CMat = new NDArray<double>(mTotal, n);
        var lVec = new NDArray<double>(mTotal);
        var uVec = new NDArray<double>(mTotal);

        for (int i = 0; i < mTotal; i++)
        {
            for (int j = 0; j < n; j++) CMat[i, j] = constraintRows[i][j];
            lVec[i] = lList[i];
            uVec[i] = uList[i];
        }

        var CT = CMat.MatrixTranspose(); // [n, mTotal]

        // System matrix K = P + sigma * I + rho * C^T * C
        var K = P.Clone();
        for (int i = 0; i < n; i++) K[i, i] += sigma;

        var CTC = MatrixMultiplication.MatMul(CT, CMat);
        K = K + CTC * rho;

        var (pK, lK, uK) = Decomposition.LU(K);

        // State variables
        var x = new NDArray<double>(n);
        var z = new NDArray<double>(mTotal);
        var y = new NDArray<double>(mTotal);

        for (int iter = 0; iter < maxIter; iter++)
        {
            // RHS = sigma * x - q + C^T * (rho * z - y)
            var rhs = new NDArray<double>(n, 1);
            var rhoZMinusY = new NDArray<double>(mTotal, 1);
            for (int i = 0; i < mTotal; i++)
            {
                rhoZMinusY[i, 0] = rho * z[i] - y[i];
            }

            var ctTerm = MatrixMultiplication.MatMul(CT, rhoZMinusY);

            for (int i = 0; i < n; i++)
            {
                rhs[i, 0] = sigma * x[i] - q[i] + ctTerm[i, 0];
            }

            // Solve K * xNext = rhs
            var pb = MatrixMultiplication.MatMul(pK, rhs);
            var yTemp = new NDArray<double>(n, 1);
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < i; j++) sum += lK[i, j] * yTemp[j, 0];
                yTemp[i, 0] = pb[i, 0] - sum;
            }

            var xNext = new NDArray<double>(n);
            for (int i = n - 1; i >= 0; i--)
            {
                double sum = 0.0;
                for (int j = i + 1; j < n; j++) sum += uK[i, j] * xNext[j];
                xNext[i] = (yTemp[i, 0] - sum) / uK[i, i];
            }

            // zNext = clamp(C * xNext + y / rho, l, u)
            var Cx = MatrixMultiplication.MatMul(CMat, xNext.Reshape(n, 1));
            var zNext = new NDArray<double>(mTotal);
            double primRes = 0.0;

            for (int i = 0; i < mTotal; i++)
            {
                double val = Cx[i, 0] + y[i] / rho;
                double clamped = Math.Clamp(val, lVec[i], uVec[i]);
                zNext[i] = clamped;

                double diff = Cx[i, 0] - clamped;
                primRes = Math.Max(primRes, Math.Abs(diff));
            }

            // yNext = y + rho * (C * xNext - zNext)
            double dualRes = 0.0;
            for (int i = 0; i < mTotal; i++)
            {
                double diffZ = zNext[i] - z[i];
                dualRes = Math.Max(dualRes, Math.Abs(rho * diffZ));
                y[i] += rho * (Cx[i, 0] - zNext[i]);
            }

            x = xNext;
            z = zNext;

            if (primRes < tol && dualRes < tol)
                break;
        }

        // Objective value = 0.5 * x^T * P * x + q^T * x
        var Px = MatrixMultiplication.MatMul(P, x.Reshape(n, 1));
        double obj = 0.0;
        for (int i = 0; i < n; i++)
        {
            obj += 0.5 * x[i] * Px[i, 0] + q[i] * x[i];
        }

        return (x, obj);
    }

    #endregion
}

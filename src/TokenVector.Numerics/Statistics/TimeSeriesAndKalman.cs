// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Statistics;

/// <summary>
/// Provides advanced time-series forecasting (Holt-Winters, ACF, PACF) and optimal state estimation (Kalman Filter 1D &amp; Multi-Dimensional).
/// </summary>
public static class TimeSeriesAndKalman
{
    #region 1D & Multi-Dimensional Kalman Filters

    /// <summary>
    /// 1D Kalman Filter implementation for optimal state estimation with scalar Gaussian noise.
    /// </summary>
    public sealed class KalmanFilter1D
    {
        public double State { get; private set; }
        public double Variance { get; private set; }
        public double ProcessNoise { get; set; }
        public double MeasurementNoise { get; set; }

        public KalmanFilter1D(double initialState, double initialVariance, double processNoise, double measurementNoise)
        {
            State = initialState;
            Variance = initialVariance;
            ProcessNoise = processNoise;
            MeasurementNoise = measurementNoise;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Predict(double u = 0.0)
        {
            State += u;
            Variance += ProcessNoise;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Update(double measurement)
        {
            double k = Variance / (Variance + MeasurementNoise);
            State += k * (measurement - State);
            Variance = (1.0 - k) * Variance;
            return State;
        }
    }

    /// <summary>
    /// Multi-Dimensional Linear Kalman Filter: x_k = F * x_{k-1} + w, z_k = H * x_k + v.
    /// </summary>
    public sealed class KalmanFilterND
    {
        public NDArray<double> State { get; private set; } // [dim_x, 1]
        public NDArray<double> Covariance { get; private set; } // [dim_x, dim_x]
        public NDArray<double> F { get; set; } // State transition [dim_x, dim_x]
        public NDArray<double> H { get; set; } // Measurement matrix [dim_z, dim_x]
        public NDArray<double> Q { get; set; } // Process noise [dim_x, dim_x]
        public NDArray<double> R { get; set; } // Measurement noise [dim_z, dim_z]

        public KalmanFilterND(NDArray<double> initialState, NDArray<double> initialCov, NDArray<double> f, NDArray<double> h, NDArray<double> q, NDArray<double> r)
        {
            State = initialState.Clone();
            Covariance = initialCov.Clone();
            F = f.Clone();
            H = h.Clone();
            Q = q.Clone();
            R = r.Clone();
        }

        public void Predict()
        {
            // x = F * x
            State = MatrixMultiplication.MatMul(F, State);
            // P = F * P * F^T + Q
            var fp = MatrixMultiplication.MatMul(F, Covariance);
            var fpfT = MatrixMultiplication.MatMul(fp, F.MatrixTranspose());
            Covariance = fpfT + Q;
        }

        public NDArray<double> Update(NDArray<double> measurement)
        {
            // y = z - H * x (Innovation)
            var hx = MatrixMultiplication.MatMul(H, State);
            var y = measurement - hx;

            // S = H * P * H^T + R (Innovation covariance)
            var hp = MatrixMultiplication.MatMul(H, Covariance);
            var s = MatrixMultiplication.MatMul(hp, H.MatrixTranspose()) + R;

            // K = P * H^T * S^-1 (Kalman Gain)
            var sInv = Decomposition.Inverse(s);
            var pht = MatrixMultiplication.MatMul(Covariance, H.MatrixTranspose());
            var k = MatrixMultiplication.MatMul(pht, sInv);

            // x = x + K * y
            State = State + MatrixMultiplication.MatMul(k, y);

            // P = (I - K * H) * P
            int dimX = State.Shape[0];
            var iMat = NDArray<double>.Eye(dimX);
            var kh = MatrixMultiplication.MatMul(k, H);
            var iMinusKh = iMat - kh;
            Covariance = MatrixMultiplication.MatMul(iMinusKh, Covariance);

            return State;
        }
    }

    #endregion

    #region Exponential Smoothing (Simple, Holt, Holt-Winters)

    /// <summary>
    /// Simple Exponential Smoothing: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}.
    /// </summary>
    public static NDArray<double> SimpleExponentialSmoothing(NDArray<double> series, double alpha)
    {
        var s = series.Contiguous();
        int n = s.TotalLength;
        var res = new NDArray<double>(n);
        if (n == 0) return res;

        res[0] = s[0];
        for (int i = 1; i < n; i++)
        {
            res[i] = alpha * s[i] + (1.0 - alpha) * res[i - 1];
        }

        return res;
    }

    /// <summary>
    /// Holt's Linear Trend Forecasting with alpha (level) and beta (trend).
    /// </summary>
    public static (NDArray<double> Fitted, NDArray<double> Forecast) HoltLinearTrend(
        NDArray<double> series, double alpha, double beta, int forecastSteps = 5)
    {
        var y = series.Contiguous();
        int n = y.TotalLength;
        var fitted = new NDArray<double>(n);

        if (n < 2) throw new ArgumentException("Holt linear trend requires at least 2 points.");

        double level = y[0];
        double trend = y[1] - y[0];
        fitted[0] = level;

        for (int i = 1; i < n; i++)
        {
            double prevLevel = level;
            level = alpha * y[i] + (1.0 - alpha) * (prevLevel + trend);
            trend = beta * (level - prevLevel) + (1.0 - beta) * trend;
            fitted[i] = level + trend;
        }

        var forecast = new NDArray<double>(forecastSteps);
        for (int h = 1; h <= forecastSteps; h++)
        {
            forecast[h - 1] = level + h * trend;
        }

        return (fitted, forecast);
    }

    #endregion

    #region Autocorrelation & Partial Autocorrelation (ACF & PACF)

    /// <summary>
    /// Computes sample Autocorrelation Function (ACF) up to maxLag.
    /// </summary>
    public static NDArray<double> Autocorrelation(NDArray<double> series, int maxLag)
    {
        var y = series.Contiguous();
        int n = y.TotalLength;
        if (maxLag >= n) maxLag = n - 1;

        double mean = 0.0;
        for (int i = 0; i < n; i++) mean += y[i];
        mean /= n;

        double variance = 0.0;
        for (int i = 0; i < n; i++)
        {
            double diff = y[i] - mean;
            variance += diff * diff;
        }

        var acf = new NDArray<double>(maxLag + 1);
        acf[0] = 1.0;

        for (int k = 1; k <= maxLag; k++)
        {
            double autoCov = 0.0;
            for (int i = 0; i < n - k; i++)
            {
                autoCov += (y[i] - mean) * (y[i + k] - mean);
            }
            acf[k] = (variance > 1e-15) ? (autoCov / variance) : 0.0;
        }

        return acf;
    }

    /// <summary>
    /// Computes sample Partial Autocorrelation Function (PACF) using Durbin-Levinson recursion.
    /// </summary>
    public static NDArray<double> PartialAutocorrelation(NDArray<double> series, int maxLag)
    {
        var r = Autocorrelation(series, maxLag).Contiguous();
        int p = maxLag;

        var pacf = new NDArray<double>(p + 1);
        pacf[0] = 1.0;
        if (p == 0) return pacf;

        double[] phi = new double[p + 1];
        double[] prevPhi = new double[p + 1];

        phi[1] = r[1];
        pacf[1] = phi[1];

        for (int k = 2; k <= p; k++)
        {
            for (int j = 1; j < k; j++) prevPhi[j] = phi[j];

            double num = r[k];
            double denom = 1.0;
            for (int j = 1; j < k; j++)
            {
                num -= prevPhi[j] * r[k - j];
                denom -= prevPhi[j] * r[j];
            }

            phi[k] = (Math.Abs(denom) > 1e-15) ? (num / denom) : 0.0;
            pacf[k] = phi[k];

            for (int j = 1; j < k; j++)
            {
                phi[j] = prevPhi[j] - phi[k] * prevPhi[k - j];
            }
        }

        return pacf;
    }

    #endregion
}

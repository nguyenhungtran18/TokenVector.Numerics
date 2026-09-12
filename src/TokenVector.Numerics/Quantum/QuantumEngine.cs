// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Quantum;

/// <summary>
/// Provides high-performance N-Qubit quantum statevector simulation, universal quantum gates (H, X, Y, Z, CNOT, Toffoli), measurement collapse, and density matrix entropy.
/// </summary>
public sealed class QState
{
    private readonly int _numQubits;
    private readonly int _dim; // 2^N
    private Complex<double>[] _state; // Amplitudes

    public int NumQubits => _numQubits;
    public int Dimension => _dim;
    public ReadOnlySpan<Complex<double>> Amplitudes => _state;

    public QState(int numQubits)
    {
        if (numQubits < 1 || numQubits > 28)
            throw new ArgumentOutOfRangeException(nameof(numQubits), "Number of qubits must be between 1 and 28.");

        _numQubits = numQubits;
        _dim = 1 << numQubits;
        _state = new Complex<double>[_dim];
        _state[0] = Complex<double>.One; // Initialize to |00...0>
    }

    #region Quantum Gates

    /// <summary>
    /// Applies Hadamard gate H = (1/sqrt(2)) * [[1, 1], [1, -1]] to target qubit.
    /// </summary>
    public QState H(int targetQubit)
    {
        double invSqrt2 = 1.0 / Math.Sqrt(2.0);
        return Apply1QubitGate(targetQubit,
            new Complex<double>(invSqrt2, 0), new Complex<double>(invSqrt2, 0),
            new Complex<double>(invSqrt2, 0), new Complex<double>(-invSqrt2, 0));
    }

    /// <summary>
    /// Applies Pauli-X (NOT) gate: [[0, 1], [1, 0]] to target qubit.
    /// </summary>
    public QState X(int targetQubit)
    {
        return Apply1QubitGate(targetQubit,
            Complex<double>.Zero, Complex<double>.One,
            Complex<double>.One, Complex<double>.Zero);
    }

    /// <summary>
    /// Applies Pauli-Y gate: [[0, -i], [i, 0]] to target qubit.
    /// </summary>
    public QState Y(int targetQubit)
    {
        return Apply1QubitGate(targetQubit,
            Complex<double>.Zero, new Complex<double>(0, -1),
            new Complex<double>(0, 1), Complex<double>.Zero);
    }

    /// <summary>
    /// Applies Pauli-Z (Phase Flip) gate: [[1, 0], [0, -1]] to target qubit.
    /// </summary>
    public QState Z(int targetQubit)
    {
        return Apply1QubitGate(targetQubit,
            Complex<double>.One, Complex<double>.Zero,
            Complex<double>.Zero, new Complex<double>(-1, 0));
    }

    /// <summary>
    /// Applies Phase gate S: [[1, 0], [0, i]] to target qubit.
    /// </summary>
    public QState S(int targetQubit)
    {
        return Apply1QubitGate(targetQubit,
            Complex<double>.One, Complex<double>.Zero,
            Complex<double>.Zero, new Complex<double>(0, 1));
    }

    /// <summary>
    /// Applies T (pi/8) gate: [[1, 0], [0, e^{i*pi/4}]] to target qubit.
    /// </summary>
    public QState T(int targetQubit)
    {
        double invSqrt2 = 1.0 / Math.Sqrt(2.0);
        return Apply1QubitGate(targetQubit,
            Complex<double>.One, Complex<double>.Zero,
            Complex<double>.Zero, new Complex<double>(invSqrt2, invSqrt2));
    }

    /// <summary>
    /// Applies arbitrary rotation gate Rz(theta) around Z-axis: [[e^{-i*theta/2}, 0], [0, e^{i*theta/2}]].
    /// </summary>
    public QState Rz(int targetQubit, double theta)
    {
        double halfTheta = theta * 0.5;
        var u00 = new Complex<double>(Math.Cos(-halfTheta), Math.Sin(-halfTheta));
        var u11 = new Complex<double>(Math.Cos(halfTheta), Math.Sin(halfTheta));
        return Apply1QubitGate(targetQubit, u00, Complex<double>.Zero, Complex<double>.Zero, u11);
    }

    /// <summary>
    /// Applies Controlled-NOT (CNOT) gate between control and target qubits.
    /// </summary>
    public QState CNOT(int controlQubit, int targetQubit)
    {
        CheckQubitIndex(controlQubit);
        CheckQubitIndex(targetQubit);
        if (controlQubit == targetQubit)
            throw new ArgumentException("Control and target qubits must be distinct.");

        int cMask = 1 << controlQubit;
        int tMask = 1 << targetQubit;

        Parallel.For(0, _dim, i =>
        {
            if ((i & cMask) != 0 && (i & tMask) == 0)
            {
                int paired = i | tMask;
                var tmp = _state[i];
                _state[i] = _state[paired];
                _state[paired] = tmp;
            }
        });

        return this;
    }

    /// <summary>
    /// Applies Toffoli (CCNOT) gate with two control qubits and one target qubit.
    /// </summary>
    public QState Toffoli(int control1, int control2, int targetQubit)
    {
        CheckQubitIndex(control1);
        CheckQubitIndex(control2);
        CheckQubitIndex(targetQubit);

        int c1Mask = 1 << control1;
        int c2Mask = 1 << control2;
        int tMask = 1 << targetQubit;

        Parallel.For(0, _dim, i =>
        {
            if ((i & c1Mask) != 0 && (i & c2Mask) != 0 && (i & tMask) == 0)
            {
                int paired = i | tMask;
                var tmp = _state[i];
                _state[i] = _state[paired];
                _state[paired] = tmp;
            }
        });

        return this;
    }

    /// <summary>
    /// Applies Controlled-Z (CZ) gate.
    /// </summary>
    public QState CZ(int controlQubit, int targetQubit)
    {
        CheckQubitIndex(controlQubit);
        CheckQubitIndex(targetQubit);
        int cMask = 1 << controlQubit;
        int tMask = 1 << targetQubit;

        Parallel.For(0, _dim, i =>
        {
            if ((i & cMask) != 0 && (i & tMask) != 0)
            {
                _state[i] = _state[i] * (-1.0);
            }
        });
        return this;
    }

    /// <summary>
    /// Applies SWAP gate between qubit1 and qubit2.
    /// </summary>
    public QState SWAP(int qubit1, int qubit2)
    {
        CheckQubitIndex(qubit1);
        CheckQubitIndex(qubit2);
        if (qubit1 == qubit2) return this;

        int m1 = 1 << qubit1;
        int m2 = 1 << qubit2;

        Parallel.For(0, _dim, i =>
        {
            bool b1 = (i & m1) != 0;
            bool b2 = (i & m2) != 0;
            if (b1 && !b2)
            {
                int paired = (i & ~m1) | m2;
                var tmp = _state[i];
                _state[i] = _state[paired];
                _state[paired] = tmp;
            }
        });
        return this;
    }

    /// <summary>
    /// Applies Controlled-Rz gate.
    /// </summary>
    public QState CRz(int controlQubit, int targetQubit, double theta)
    {
        CheckQubitIndex(controlQubit);
        CheckQubitIndex(targetQubit);
        int cMask = 1 << controlQubit;
        int tMask = 1 << targetQubit;
        var expPos = new Complex<double>(Math.Cos(theta * 0.5), -Math.Sin(theta * 0.5));
        var expNeg = new Complex<double>(Math.Cos(theta * 0.5), Math.Sin(theta * 0.5));

        Parallel.For(0, _dim, i =>
        {
            if ((i & cMask) != 0)
            {
                if ((i & tMask) == 0) _state[i] = _state[i] * expPos;
                else _state[i] = _state[i] * expNeg;
            }
        });
        return this;
    }

    /// <summary>
    /// Executes full Quantum Fourier Transform (QFT) circuit on all N qubits.
    /// </summary>
    public QState QFT()
    {
        for (int i = 0; i < _numQubits; i++)
        {
            H(i);
            for (int j = i + 1; j < _numQubits; j++)
            {
                double theta = Math.PI / (1 << (j - i));
                CRz(j, i, theta);
            }
        }
        for (int i = 0; i < _numQubits / 2; i++)
        {
            SWAP(i, _numQubits - 1 - i);
        }
        return this;
    }

    private QState Apply1QubitGate(int targetQubit, Complex<double> u00, Complex<double> u01, Complex<double> u10, Complex<double> u11)
    {
        CheckQubitIndex(targetQubit);
        int bit = 1 << targetQubit;
        int halfDim = _dim >> 1;

        Parallel.For(0, halfDim, idx =>
        {
            // Compute index with target bit = 0 and target bit = 1
            int low = idx & (bit - 1);
            int high = (idx >> targetQubit) << (targetQubit + 1);
            int i0 = high | low;
            int i1 = i0 | bit;

            var a0 = _state[i0];
            var a1 = _state[i1];

            _state[i0] = u00 * a0 + u01 * a1;
            _state[i1] = u10 * a0 + u11 * a1;
        });

        return this;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void CheckQubitIndex(int q)
    {
        if (q < 0 || q >= _numQubits)
            throw new ArgumentOutOfRangeException(nameof(q), $"Qubit index {q} is out of range for {_numQubits} qubits.");
    }

    #endregion

    #region Measurement & Probabilities

    /// <summary>
    /// Computes probability of measuring target qubit in state |1>.
    /// </summary>
    public double ProbabilityOne(int targetQubit)
    {
        CheckQubitIndex(targetQubit);
        int bit = 1 << targetQubit;
        double prob = 0.0;

        for (int i = 0; i < _dim; i++)
        {
            if ((i & bit) != 0)
            {
                prob += _state[i].MagnitudeSquared();
            }
        }

        return prob;
    }

    /// <summary>
    /// Gets probability distribution across all 2^N basis states.
    /// </summary>
    public double[] GetProbabilities()
    {
        var probs = new double[_dim];
        for (int i = 0; i < _dim; i++)
        {
            probs[i] = _state[i].MagnitudeSquared();
        }
        return probs;
    }

    /// <summary>
    /// Computes pure state density matrix rho = |psi&gt;&lt;psi| (dim x dim).
    /// </summary>
    public NDArray<double> DensityMatrix()
    {
        var rho = new NDArray<double>(_dim, _dim);
        for (int i = 0; i < _dim; i++)
        {
            for (int j = 0; j < _dim; j++)
            {
                // Real part of psi_i * conj(psi_j)
                rho[i, j] = _state[i].Real * _state[j].Real + _state[i].Imaginary * _state[j].Imaginary;
            }
        }
        return rho;
    }

    /// <summary>
    /// Computes the Von Neumann Entropy of the current quantum state.
    /// </summary>
    public double VonNeumannEntropy()
    {
        return VonNeumannEntropy(DensityMatrix());
    }

    /// <summary>
    /// Measures target qubit, collapsing the statevector according to Born's Rule. Returns measured outcome (0 or 1).
    /// </summary>
    public int Measure(int targetQubit, System.Random? rng = null)
    {
        rng ??= System.Random.Shared;
        double p1 = ProbabilityOne(targetQubit);
        int outcome = rng.NextDouble() < p1 ? 1 : 0;

        double normFactor = 1.0 / Math.Sqrt(outcome == 1 ? p1 : (1.0 - p1));
        int bit = 1 << targetQubit;

        for (int i = 0; i < _dim; i++)
        {
            int bitVal = (i & bit) != 0 ? 1 : 0;
            if (bitVal == outcome)
            {
                _state[i] = _state[i] * normFactor;
            }
            else
            {
                _state[i] = Complex<double>.Zero;
            }
        }

        return outcome;
    }

    /// <summary>
    /// Computes the Von Neumann Entropy of an arbitrary density matrix rho: S(rho) = -Tr(rho * ln(rho)).
    /// </summary>
    public static double VonNeumannEntropy(NDArray<double> densityMatrix)
    {
        var (eigvals, _) = Eigen.Eigh(densityMatrix);
        double entropy = 0.0;

        for (int i = 0; i < eigvals.TotalLength; i++)
        {
            double p = eigvals[i];
            if (p > 1e-12)
            {
                entropy -= p * Math.Log(p);
            }
        }

        return entropy;
    }

    #endregion
}

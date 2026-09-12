using System;
using System.Numerics;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.LinAlg;

namespace TokenVector.Numerics.Autograd.Nodes;

/// <summary>
/// Backward nodes and factory methods for fundamental arithmetic, matrix, and tensor transformation operations.
/// </summary>
/// <typeparam name="T">Floating-point numeric type.</typeparam>
public static class MathAndMatrixNodes<T> where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
{
    #region Add

    internal sealed class AddNode : AutogradNode<T>
    {
        public AddNode(Tensor<T> a, Tensor<T> b)
        {
            Inputs = new[] { a, b };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad)
            {
                var gradA = Unbroadcast(gradOutput, Inputs[0].Shape);
                Inputs[0].AccumulateGrad(gradA);
            }

            if (Inputs[1].RequiresGrad)
            {
                var gradB = Unbroadcast(gradOutput, Inputs[1].Shape);
                Inputs[1].AccumulateGrad(gradB);
            }
        }
    }

    public static Tensor<T> Add(Tensor<T> a, Tensor<T> b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        var outData = a.Data + b.Data;
        bool requiresGrad = a.RequiresGrad || b.RequiresGrad;
        AddNode? node = requiresGrad ? new AddNode(a, b) : null;
        var result = new Tensor<T>(outData, requiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    public static Tensor<T> AddScalar(Tensor<T> a, T b)
    {
        var bTensor = Tensor<T>.Full(b, 1);
        return Add(a, bTensor);
    }

    #endregion

    #region Subtract

    internal sealed class SubNode : AutogradNode<T>
    {
        public SubNode(Tensor<T> a, Tensor<T> b)
        {
            Inputs = new[] { a, b };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad)
            {
                var gradA = Unbroadcast(gradOutput, Inputs[0].Shape);
                Inputs[0].AccumulateGrad(gradA);
            }

            if (Inputs[1].RequiresGrad)
            {
                var gradB = Unbroadcast(-gradOutput, Inputs[1].Shape);
                Inputs[1].AccumulateGrad(gradB);
            }
        }
    }

    public static Tensor<T> Subtract(Tensor<T> a, Tensor<T> b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        var outData = a.Data - b.Data;
        bool requiresGrad = a.RequiresGrad || b.RequiresGrad;
        SubNode? node = requiresGrad ? new SubNode(a, b) : null;
        var result = new Tensor<T>(outData, requiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    public static Tensor<T> SubtractScalar(Tensor<T> a, T b)
    {
        var bTensor = Tensor<T>.Full(b, 1);
        return Subtract(a, bTensor);
    }

    public static Tensor<T> SubtractFromScalar(Tensor<T> b, T a)
    {
        var aTensor = Tensor<T>.Full(a, 1);
        return Subtract(aTensor, b);
    }

    #endregion

    #region Multiply

    internal sealed class MulNode : AutogradNode<T>
    {
        public MulNode(Tensor<T> a, Tensor<T> b)
        {
            Inputs = new[] { a, b };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad)
            {
                var gradA = Unbroadcast(gradOutput * Inputs[1].Data, Inputs[0].Shape);
                Inputs[0].AccumulateGrad(gradA);
            }

            if (Inputs[1].RequiresGrad)
            {
                var gradB = Unbroadcast(gradOutput * Inputs[0].Data, Inputs[1].Shape);
                Inputs[1].AccumulateGrad(gradB);
            }
        }
    }

    public static Tensor<T> Multiply(Tensor<T> a, Tensor<T> b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        var outData = a.Data * b.Data;
        bool requiresGrad = a.RequiresGrad || b.RequiresGrad;
        MulNode? node = requiresGrad ? new MulNode(a, b) : null;
        var result = new Tensor<T>(outData, requiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    public static Tensor<T> MultiplyScalar(Tensor<T> a, T b)
    {
        var bTensor = Tensor<T>.Full(b, 1);
        return Multiply(a, bTensor);
    }

    #endregion

    #region Divide

    internal sealed class DivNode : AutogradNode<T>
    {
        public DivNode(Tensor<T> a, Tensor<T> b)
        {
            Inputs = new[] { a, b };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad)
            {
                var gradA = Unbroadcast(gradOutput / Inputs[1].Data, Inputs[0].Shape);
                Inputs[0].AccumulateGrad(gradA);
            }

            if (Inputs[1].RequiresGrad)
            {
                var bData = Inputs[1].Data;
                var gradB = Unbroadcast(-gradOutput * Inputs[0].Data / (bData * bData), Inputs[1].Shape);
                Inputs[1].AccumulateGrad(gradB);
            }
        }
    }

    public static Tensor<T> Divide(Tensor<T> a, Tensor<T> b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        var outData = a.Data / b.Data;
        bool requiresGrad = a.RequiresGrad || b.RequiresGrad;
        DivNode? node = requiresGrad ? new DivNode(a, b) : null;
        var result = new Tensor<T>(outData, requiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    public static Tensor<T> DivideScalar(Tensor<T> a, T b)
    {
        var bTensor = Tensor<T>.Full(b, 1);
        return Divide(a, bTensor);
    }

    public static Tensor<T> DivideFromScalar(Tensor<T> b, T a)
    {
        var aTensor = Tensor<T>.Full(a, 1);
        return Divide(aTensor, b);
    }

    #endregion

    #region Negate

    internal sealed class NegNode : AutogradNode<T>
    {
        public NegNode(Tensor<T> a)
        {
            Inputs = new[] { a };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad)
            {
                Inputs[0].AccumulateGrad(-gradOutput);
            }
        }
    }

    public static Tensor<T> Negate(Tensor<T> a)
    {
        ArgumentNullException.ThrowIfNull(a);
        var outData = -a.Data;
        NegNode? node = a.RequiresGrad ? new NegNode(a) : null;
        var result = new Tensor<T>(outData, a.RequiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    #endregion

    #region Power

    internal sealed class PowNode : AutogradNode<T>
    {
        private readonly T _power;

        public PowNode(Tensor<T> a, T power)
        {
            Inputs = new[] { a };
            _power = power;
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad)
            {
                // d/dx (x^p) = p * x^(p-1)
                var pMinus1 = _power - T.One;
                var inSpan = Inputs[0].Data.AsReadOnlySpan();
                var gSpan = gradOutput.AsReadOnlySpan();
                var gradND = new NDArray<T>(Inputs[0].Shape);
                var gradSpan = gradND.AsSpan();

                for (int i = 0; i < inSpan.Length; i++)
                {
                    gradSpan[i] = gSpan[i] * _power * T.Pow(inSpan[i], pMinus1);
                }

                Inputs[0].AccumulateGrad(gradND);
            }
        }
    }

    public static Tensor<T> Pow(Tensor<T> a, T power)
    {
        ArgumentNullException.ThrowIfNull(a);
        var inSpan = a.Data.AsReadOnlySpan();
        var outND = new NDArray<T>(a.Shape);
        var outSpan = outND.AsSpan();

        for (int i = 0; i < inSpan.Length; i++)
        {
            outSpan[i] = T.Pow(inSpan[i], power);
        }

        PowNode? node = a.RequiresGrad ? new PowNode(a, power) : null;
        var result = new Tensor<T>(outND, a.RequiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    #endregion

    #region Exp & Log

    internal sealed class ExpNode : AutogradNode<T>
    {
        public ExpNode(Tensor<T> a)
        {
            Inputs = new[] { a };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad && Output != null)
            {
                // d/dx (exp(x)) = exp(x) = Output
                var gradA = gradOutput * Output.Data;
                Inputs[0].AccumulateGrad(gradA);
            }
        }
    }

    public static Tensor<T> Exp(Tensor<T> a)
    {
        ArgumentNullException.ThrowIfNull(a);
        var inSpan = a.Data.AsReadOnlySpan();
        var outND = new NDArray<T>(a.Shape);
        var outSpan = outND.AsSpan();

        for (int i = 0; i < inSpan.Length; i++)
        {
            outSpan[i] = T.Exp(inSpan[i]);
        }

        ExpNode? node = a.RequiresGrad ? new ExpNode(a) : null;
        var result = new Tensor<T>(outND, a.RequiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    internal sealed class LogNode : AutogradNode<T>
    {
        public LogNode(Tensor<T> a)
        {
            Inputs = new[] { a };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad)
            {
                // d/dx (log(x)) = 1 / x
                var gradA = gradOutput / Inputs[0].Data;
                Inputs[0].AccumulateGrad(gradA);
            }
        }
    }

    public static Tensor<T> Log(Tensor<T> a)
    {
        ArgumentNullException.ThrowIfNull(a);
        var inSpan = a.Data.AsReadOnlySpan();
        var outND = new NDArray<T>(a.Shape);
        var outSpan = outND.AsSpan();

        for (int i = 0; i < inSpan.Length; i++)
        {
            outSpan[i] = T.Log(inSpan[i]);
        }

        LogNode? node = a.RequiresGrad ? new LogNode(a) : null;
        var result = new Tensor<T>(outND, a.RequiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    #endregion

    #region MatMul

    internal sealed class MatMulNode : AutogradNode<T>
    {
        public MatMulNode(Tensor<T> a, Tensor<T> b)
        {
            Inputs = new[] { a, b };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            var a = Inputs[0];
            var b = Inputs[1];

            if (a.RequiresGrad)
            {
                var bT = b.Data.Transpose(0, 1);
                var gradA = MatrixMultiplication.MatMul(gradOutput, bT);
                a.AccumulateGrad(Unbroadcast(gradA, a.Shape));
            }

            if (b.RequiresGrad)
            {
                var aT = a.Data.Transpose(0, 1);
                var gradB = MatrixMultiplication.MatMul(aT, gradOutput);
                b.AccumulateGrad(Unbroadcast(gradB, b.Shape));
            }
        }
    }

    public static Tensor<T> MatMul(Tensor<T> a, Tensor<T> b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        var outData = MatrixMultiplication.MatMul(a.Data, b.Data);
        bool requiresGrad = a.RequiresGrad || b.RequiresGrad;
        MatMulNode? node = requiresGrad ? new MatMulNode(a, b) : null;
        var result = new Tensor<T>(outData, requiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    #endregion

    #region Transpose & Reshape

    internal sealed class TransposeNode : AutogradNode<T>
    {
        private readonly int _axis0;
        private readonly int _axis1;

        public TransposeNode(Tensor<T> a, int axis0, int axis1)
        {
            Inputs = new[] { a };
            _axis0 = axis0;
            _axis1 = axis1;
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad)
            {
                var gradA = gradOutput.Transpose(_axis0, _axis1);
                Inputs[0].AccumulateGrad(gradA.IsContiguous ? gradA : gradA.Contiguous());
            }
        }
    }

    public static Tensor<T> Transpose(Tensor<T> a, int axis0 = 0, int axis1 = 1)
    {
        ArgumentNullException.ThrowIfNull(a);
        var outData = a.Data.Transpose(axis0, axis1);
        TransposeNode? node = a.RequiresGrad ? new TransposeNode(a, axis0, axis1) : null;
        var result = new Tensor<T>(outData, a.RequiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    internal sealed class ReshapeNode : AutogradNode<T>
    {
        public ReshapeNode(Tensor<T> a)
        {
            Inputs = new[] { a };
        }

        public override void Backward(NDArray<T> gradOutput)
        {
            if (Inputs[0].RequiresGrad)
            {
                var gradA = gradOutput.Reshape(Inputs[0].Shape);
                Inputs[0].AccumulateGrad(gradA.IsContiguous ? gradA : gradA.Contiguous());
            }
        }
    }

    public static Tensor<T> Reshape(Tensor<T> a, params int[] newShape)
    {
        ArgumentNullException.ThrowIfNull(a);
        var outData = a.Data.Reshape(newShape);
        ReshapeNode? node = a.RequiresGrad ? new ReshapeNode(a) : null;
        var result = new Tensor<T>(outData, a.RequiresGrad, node);
        if (node != null) node.Output = result;
        return result;
    }

    #endregion
}

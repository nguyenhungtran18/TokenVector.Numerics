using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.LinAlg;

/// <summary>
/// Einstein Summation (EinSum) engine for arbitrary tensor contractions and index notation.
/// </summary>
public static class EinSum
{
    /// <summary>
    /// Evaluates Einstein Summation convention on operands: e.g. "ij,jk->ik", "bij,bjk->bik", "ii->", "ij->ji".
    /// </summary>
    public static NDArray<T> Evaluate<T>(string subscripts, params NDArray<T>[] operands) where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(subscripts);
        ArgumentNullException.ThrowIfNull(operands);

        subscripts = subscripts.Replace(" ", "");
        string[] parts = subscripts.Split("->");

        string[] inputTerms = parts[0].Split(',');
        if (inputTerms.Length != operands.Length)
        {
            throw new ArgumentException($"Number of subscripts terms ({inputTerms.Length}) does not match number of operands ({operands.Length}).");
        }

        // Map label -> dimension size
        Dictionary<char, int> dimSizes = new();
        for (int opIdx = 0; opIdx < operands.Length; opIdx++)
        {
            string term = inputTerms[opIdx];
            var op = operands[opIdx];
            if (term.Length != op.Rank)
            {
                throw new ArgumentException($"Subscript term '{term}' length does not match operand {opIdx} rank {op.Rank}.");
            }

            for (int d = 0; d < term.Length; d++)
            {
                char label = term[d];
                int size = op.Shape[d];

                if (dimSizes.TryGetValue(label, out int existingSize))
                {
                    if (existingSize != size)
                    {
                        throw new InvalidOperationException($"Dimension mismatch for label '{label}': {existingSize} != {size}.");
                    }
                }
                else
                {
                    dimSizes[label] = size;
                }
            }
        }

        // Determine output labels
        string outputTerm;
        if (parts.Length > 1)
        {
            outputTerm = parts[1];
        }
        else
        {
            // Standard NumPy default: labels that appear exactly once in input, sorted alphabetically
            Dictionary<char, int> counts = new();
            foreach (var term in inputTerms)
                foreach (char c in term)
                    counts[c] = counts.GetValueOrDefault(c) + 1;

            outputTerm = new string(counts.Where(kv => kv.Value == 1).Select(kv => kv.Key).OrderBy(c => c).ToArray());
        }

        int[] outShape = outputTerm.Select(c => dimSizes[c]).ToArray();
        var result = new NDArray<T>(outShape);

        // Separate output labels and contracted (summed) labels
        HashSet<char> outSet = new(outputTerm);
        List<char> contractedLabels = dimSizes.Keys.Where(c => !outSet.Contains(c)).ToList();
        int[] contractedSizes = contractedLabels.Select(c => dimSizes[c]).ToArray();
        int numContracted = ShapeHelper.ComputeTotalLength(contractedSizes);

        int outTotal = result.TotalLength;

        Parallel.For(0, outTotal, outIdx =>
        {
            Span<int> outCoords = stackalloc int[outShape.Length];
            ShapeHelper.GetMultiIndex(outIdx, outShape, outCoords);

            // Create lookup map for this output element
            Dictionary<char, int> labelValues = new();
            for (int i = 0; i < outputTerm.Length; i++)
            {
                labelValues[outputTerm[i]] = outCoords[i];
            }

            T sum = T.Zero;

            Span<int> cCoords = stackalloc int[contractedLabels.Count];
            for (int cIdx = 0; cIdx < numContracted; cIdx++)
            {
                ShapeHelper.GetMultiIndex(cIdx, contractedSizes, cCoords);
                for (int i = 0; i < contractedLabels.Count; i++)
                {
                    labelValues[contractedLabels[i]] = cCoords[i];
                }

                // Compute product across all operands
                T prod = T.One;
                for (int opIdx = 0; opIdx < operands.Length; opIdx++)
                {
                    string term = inputTerms[opIdx];
                    var op = operands[opIdx];

                    int[] opCoords = new int[term.Length];
                    for (int d = 0; d < term.Length; d++)
                    {
                        opCoords[d] = labelValues[term[d]];
                    }

                    prod *= op[opCoords];
                }

                sum += prod;
            }

            result[outCoords.ToArray()] = sum;
        });

        return result;
    }
}

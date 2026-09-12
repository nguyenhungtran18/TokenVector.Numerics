// Copyright (c) TokenVector Project. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using TokenVector.Numerics.Core;

namespace TokenVector.Numerics.Spatial;

/// <summary>
/// Spatial KD-Tree, 2D Convex Hull, Pairwise Distance Matrices (CDist), and Computational Polygon Geometry.
/// </summary>
public static class SpatialTreesAndGeometry
{
    #region KD-Tree for Fast N-Dimensional Spatial Search

    public sealed class KDTree
    {
        private sealed class KDNode
        {
            public double[] Point;
            public int Index;
            public KDNode? Left;
            public KDNode? Right;
            public int Axis;

            public KDNode(double[] point, int index, int axis)
            {
                Point = point;
                Index = index;
                Axis = axis;
            }
        }

        private readonly KDNode? _root;
        private readonly int _dim;

        public KDTree(NDArray<double> points)
        {
            int n = points.Shape[0];
            _dim = points.Shape[1];

            var pointList = new List<(double[] pt, int idx)>(n);
            for (int i = 0; i < n; i++)
            {
                double[] pt = new double[_dim];
                for (int d = 0; d < _dim; d++) pt[d] = points[i, d];
                pointList.Add((pt, i));
            }

            _root = BuildTree(pointList, 0);
        }

        private KDNode? BuildTree(List<(double[] pt, int idx)> pts, int depth)
        {
            if (pts.Count == 0) return null;

            int axis = depth % _dim;
            pts.Sort((a, b) => a.pt[axis].CompareTo(b.pt[axis]));

            int mid = pts.Count / 2;
            var node = new KDNode(pts[mid].pt, pts[mid].idx, axis);

            var leftPts = pts.GetRange(0, mid);
            var rightPts = pts.GetRange(mid + 1, pts.Count - (mid + 1));

            node.Left = BuildTree(leftPts, depth + 1);
            node.Right = BuildTree(rightPts, depth + 1);

            return node;
        }

        public (double Distance, int Index) QueryNearest(NDArray<double> queryPoint)
        {
            double[] q = new double[_dim];
            for (int d = 0; d < _dim; d++) q[d] = queryPoint[d];

            double bestDistSq = double.MaxValue;
            int bestIdx = -1;

            SearchNearest(_root, q, ref bestDistSq, ref bestIdx);

            return (Math.Sqrt(bestDistSq), bestIdx);
        }

        private void SearchNearest(KDNode? node, double[] q, ref double bestDistSq, ref int bestIdx)
        {
            if (node == null) return;

            double distSq = 0.0;
            for (int d = 0; d < _dim; d++)
            {
                double diff = q[d] - node.Point[d];
                distSq += diff * diff;
            }

            if (distSq < bestDistSq)
            {
                bestDistSq = distSq;
                bestIdx = node.Index;
            }

            int axis = node.Axis;
            double planeDiff = q[axis] - node.Point[axis];

            KDNode? first = planeDiff < 0 ? node.Left : node.Right;
            KDNode? second = planeDiff < 0 ? node.Right : node.Left;

            SearchNearest(first, q, ref bestDistSq, ref bestIdx);

            if (planeDiff * planeDiff < bestDistSq)
            {
                SearchNearest(second, q, ref bestDistSq, ref bestIdx);
            }
        }
    }

    #endregion

    #region 2D Convex Hull (Andrew's Monotone Chain)

    /// <summary>
    /// Computes the 2D convex hull of a set of points [N, 2] in O(N log N) using Andrew's monotone chain algorithm.
    /// Returns the vertices of the convex hull in counter-clockwise order.
    /// </summary>
    public static NDArray<double> ConvexHull2D(NDArray<double> points)
    {
        int n = points.Shape[0];
        if (n <= 3) return points.Clone();

        var pts = new List<(double X, double Y)>(n);
        for (int i = 0; i < n; i++)
        {
            pts.Add((points[i, 0], points[i, 1]));
        }

        pts.Sort((a, b) => a.X != b.X ? a.X.CompareTo(b.X) : a.Y.CompareTo(b.Y));

        double Cross((double X, double Y) o, (double X, double Y) a, (double X, double Y) b) =>
            (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);

        var lower = new List<(double X, double Y)>();
        foreach (var p in pts)
        {
            while (lower.Count >= 2 && Cross(lower[^2], lower[^1], p) <= 0)
                lower.RemoveAt(lower.Count - 1);
            lower.Add(p);
        }

        var upper = new List<(double X, double Y)>();
        for (int i = pts.Count - 1; i >= 0; i--)
        {
            var p = pts[i];
            while (upper.Count >= 2 && Cross(upper[^2], upper[^1], p) <= 0)
                upper.RemoveAt(upper.Count - 1);
            upper.Add(p);
        }

        lower.RemoveAt(lower.Count - 1);
        upper.RemoveAt(upper.Count - 1);
        lower.AddRange(upper);

        var hull = new NDArray<double>(lower.Count, 2);
        for (int i = 0; i < lower.Count; i++)
        {
            hull[i, 0] = lower[i].X;
            hull[i, 1] = lower[i].Y;
        }

        return hull;
    }

    #endregion

    #region Pairwise Distance Matrices (CDist)

    /// <summary>
    /// Computes pairwise distance matrix between collections of points XA [mA, d] and XB [mB, d].
    /// Metrics: "euclidean", "manhattan", "chebyshev", "cosine".
    /// </summary>
    public static NDArray<double> CDist(NDArray<double> xa, NDArray<double> xb, string metric = "euclidean")
    {
        int ma = xa.Shape[0];
        int mb = xb.Shape[0];
        int d = xa.Shape[1];

        var dist = new NDArray<double>(ma, mb);

        for (int i = 0; i < ma; i++)
        {
            for (int j = 0; j < mb; j++)
            {
                if (metric == "manhattan")
                {
                    double sum = 0;
                    for (int k = 0; k < d; k++) sum += Math.Abs(xa[i, k] - xb[j, k]);
                    dist[i, j] = sum;
                }
                else if (metric == "chebyshev")
                {
                    double maxD = 0;
                    for (int k = 0; k < d; k++) maxD = Math.Max(maxD, Math.Abs(xa[i, k] - xb[j, k]));
                    dist[i, j] = maxD;
                }
                else if (metric == "cosine")
                {
                    double dot = 0, nA = 0, nB = 0;
                    for (int k = 0; k < d; k++)
                    {
                        dot += xa[i, k] * xb[j, k];
                        nA += xa[i, k] * xa[i, k];
                        nB += xb[j, k] * xb[j, k];
                    }
                    double denom = Math.Sqrt(nA * nB);
                    dist[i, j] = denom > 1e-15 ? 1.0 - (dot / denom) : 0.0;
                }
                else // euclidean
                {
                    double sumSq = 0;
                    for (int k = 0; k < d; k++)
                    {
                        double diff = xa[i, k] - xb[j, k];
                        sumSq += diff * diff;
                    }
                    dist[i, j] = Math.Sqrt(sumSq);
                }
            }
        }

        return dist;
    }

    #endregion

    #region Polygon Geometry

    /// <summary>
    /// Tests whether point (x, y) is inside 2D polygon using Ray Casting algorithm.
    /// </summary>
    public static bool PointInPolygon(NDArray<double> polygon, double x, double y)
    {
        int n = polygon.Shape[0];
        bool inside = false;

        for (int i = 0, j = n - 1; i < n; j = i++)
        {
            double xi = polygon[i, 0], yi = polygon[i, 1];
            double xj = polygon[j, 0], yj = polygon[j, 1];

            bool intersect = ((yi > y) != (yj > y)) &&
                             (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }

        return inside;
    }

    /// <summary>
    /// Computes area of 2D polygon using Shoelace formula.
    /// </summary>
    public static double PolygonArea(NDArray<double> polygon)
    {
        int n = polygon.Shape[0];
        double area = 0.0;
        for (int i = 0; i < n; i++)
        {
            int j = (i + 1) % n;
            area += polygon[i, 0] * polygon[j, 1];
            area -= polygon[j, 0] * polygon[i, 1];
        }
        return Math.Abs(area) * 0.5;
    }

    #endregion
}

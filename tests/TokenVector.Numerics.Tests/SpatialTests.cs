using System;
using TokenVector.Numerics.Core;
using TokenVector.Numerics.Spatial;
using Xunit;

namespace TokenVector.Numerics.Tests;

public class SpatialTests
{
    [Fact]
    public void TestAffine3DTranslationAndScale()
    {
        var trans = Affine3D.Translation(10.0f, 20.0f, 30.0f);
        Assert.Equal(10.0f, trans[0, 3]);
        Assert.Equal(20.0f, trans[1, 3]);
        Assert.Equal(30.0f, trans[2, 3]);

        var scale = Affine3D.Scaling(2.0f, 3.0f, 4.0f);
        Assert.Equal(2.0f, scale[0, 0]);
        Assert.Equal(3.0f, scale[1, 1]);
        Assert.Equal(4.0f, scale[2, 2]);
    }

    [Fact]
    public void TestQuaternionMultiplicationAndSlerp()
    {
        var q1 = Quaternion<float>.FromAxisAngle(0.0f, 1.0f, 0.0f, 0.0f); // 0 degrees rotation
        var q2 = Quaternion<float>.FromAxisAngle(0.0f, 1.0f, 0.0f, (float)Math.PI / 2.0f); // 90 degrees rotation

        // Slerp at t = 0.5 should be 45 degrees
        var qMid = Quaternion<float>.Slerp(q1, q2, 0.5f);
        var qExpected = Quaternion<float>.FromAxisAngle(0.0f, 1.0f, 0.0f, (float)Math.PI / 4.0f);

        Assert.Equal(qExpected.X, qMid.X, precision: 4);
        Assert.Equal(qExpected.Y, qMid.Y, precision: 4);
        Assert.Equal(qExpected.Z, qMid.Z, precision: 4);
        Assert.Equal(qExpected.W, qMid.W, precision: 4);
    }

    [Fact]
    public void TestVectorCrossProduct()
    {
        // X-axis (1, 0, 0) cross Y-axis (0, 1, 0) = Z-axis (0, 0, 1)
        var vX = NDArray<float>.FromArray(new[] { 1.0f, 0.0f, 0.0f }, 3);
        var vY = NDArray<float>.FromArray(new[] { 0.0f, 1.0f, 0.0f }, 3);

        var vZ = Affine3D.Cross(vX, vY);
        Assert.Equal(0.0f, vZ[0]);
        Assert.Equal(0.0f, vZ[1]);
        Assert.Equal(1.0f, vZ[2]);
    }
}

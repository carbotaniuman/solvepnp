import core.stdc.float_ : DBL_EPSILON;

import mir.blas : nrm2, gemm;
import mir.exception : enforce;
import mir.math.numeric : prod;
import mir.math.stat : mean;
import mir.ndslice;
import mir.optim.least_squares : LeastSquaresSettings, optimize;
import mir.rc.array : RCI;

import std.array;
import std.algorithm : canFind, copy;
import std.math : abs;
import std.traits : isFloatingPoint;
import std.range : repeat;

import kaleidic.lubeck2 : mtimes, svd;

import project;

@nogc:

private bool isProperDistCoeffs(Iterator, SliceKind kind)(Slice!(Iterator, 1, kind) slice)
{
    return slice.length == 0 || slice.length == 4 || slice.length == 5 || slice.length == 8;
}

auto undistortPoints(T, SliceKind kindA, SliceKind kindB, SliceKind kindC)(Slice!(
        const(T)*, 2, kindA) slice, Slice!(const(float)*, 1,
        kindB) distCoeffs, Slice!(const(float)*, 2, kindC) cameraMatrix,)
        if (isFloatingPoint!T)
in
{
    assert(isProperDistCoeffs(distCoeffs), "distCoeffs must have length of 0, 4, 5, 8");
    assert(cameraMatrix.shape == [3, 3], "camera matrix must be of [3, 3]");
}
do
{
    int iters = 1;
    auto cameraMatrixD = cameraMatrix.as!double;

    auto n = slice.length;
    double[8] k = staticArray!(0.0.repeat(8));
    copy(distCoeffs.as!double, k[]);

    double[3][3] RR = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    double fx = cameraMatrixD[0][0];
    double fy = cameraMatrixD[1][1];
    double ifx = 1.0 / fx;
    double ify = 1.0 / fy;
    double cx = cameraMatrixD[0][2];
    double cy = cameraMatrixD[1][2];

    auto ret = rcslice!T(slice.shape);

    foreach (i; 0 .. n)
    {
        double x = slice[i][0];
        double y = slice[i][1];

        double x0, y0;
        x0 = x = (x - cx) * ifx;
        y0 = y = (y - cy) * ify;

        foreach (j; 0 .. iters)
        {
            double r2 = x * x + y * y;
            double icdist = (1 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2) / (
                    1 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2);
            double deltaX = 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x);
            double deltaY = k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y;
            x = (x0 - deltaX) * icdist;
            y = (y0 - deltaY) * icdist;
        }

        double xx = RR[0][0] * x + RR[0][1] * y + RR[0][2];
        double yy = RR[1][0] * x + RR[1][1] * y + RR[1][2];
        double ww = 1. / (RR[2][0] * x + RR[2][1] * y + RR[2][2]);
        x = xx * ww;
        y = yy * ww;

        ret[i][0] = cast(T) x;
        ret[i][1] = cast(T) y;
    }

    return ret;
}

auto solvePnPIterative(SliceKind kindA, SliceKind kindB, SliceKind kindC, SliceKind kindD)(Slice!(const(float)*,
        2, kindA) objectPoints, Slice!(const(float)*, 2, kindB) imagePoints,
        Slice!(const(float)*, 2, kindC) cameraMatrix, Slice!(const(float)*,
            1, kindD) distCoeffs,)
{
    scope float[3] data = [0.0, 0.0, 0.0];
    return solvePnPIterative(objectPoints, imagePoints, cameraMatrix,
            distCoeffs, data.sliced, data.sliced, false);
}

struct PnPResult
{
    Slice!(RCI!double, 1, Contiguous) rvec;
    Slice!(RCI!double, 1, Contiguous) tvec;
}

PnPResult solvePnPIterative(SliceKind kindA, SliceKind kindB, SliceKind kindC,
        SliceKind kindD, SliceKind kindE)(Slice!(const(float)*, 2,
        kindA) objectPoints, Slice!(const(float)*, 2, kindB) imagePoints,
        Slice!(const(float)*, 2, kindC) cameraMatrix, Slice!(const(float)*,
            1, kindD) distCoeffs, Slice!(const(float)*, 1, kindD) rVec,
        Slice!(const(float)*, 1, kindE) tVec, bool useExtrinsicGuess = true,)
in
{
    assert(objectPoints.shape[0] == imagePoints.shape[0],
            "array shape must be objectPoints=[N, 3] imagePoints=[N, 2]");
    assert(objectPoints.shape[1] == 3, "array shape must be objectPoints=[N, 3] imagePoints=[N, 2]");
    assert(imagePoints.shape[1] == 2, "array shape must be objectPoints=[N, 3] imagePoints=[N, 2]");
    assert(isProperDistCoeffs(distCoeffs), "distCoeffs must have length of 0, 4, 5, 8, 12, or 14");
    assert(cameraMatrix.shape == [3, 3], "camera");
    assert(rVec.shape == [3], "rvec must be a vec3");
    assert(tVec.shape == [3], "tVec must be a vec3");
}
do
{
    auto count = objectPoints.length;
    double[9] a = staticArray!(0.0.repeat(9));
    double[6] params;
    auto _r = params[0 .. 3].sliced;
    auto _t = params[3 .. 6].sliced;

    auto objectPointsD = objectPoints.as!double.rcfuse;
    auto imagePointsD = imagePoints.as!double.rcfuse;
    auto cameraMatrixD = cameraMatrix.as!double.rcfuse;

    auto mn = undistortPoints(imagePointsD.lightScope, distCoeffs, cameraMatrix);

    if (useExtrinsicGuess)
    {
        copy(concatenation(rVec, tVec).as!double, params[]);
    }
    else
    {
        double[3] temp;
        copy(objectPointsD.lightScope
                .alongDim!0
                .map!((a) => a.mean), temp[]);

        auto subtracted = objectPointsD.rcslice;
        subtracted[] -= temp.sliced;
        auto transposed = subtracted.transposed.mtimes(subtracted);
        auto res = svd(transposed);

        auto v = res.vt;
        auto w = res.sigma;

        if (w[2] / w[1] < 1e-3)
        {
            // planar case
            assert(false);
        }
        else
        {
            assert(count >= 6, "DLT needs 6 points to work properly");

            auto mL = rcslice!double(2 * count, 12);
            auto L = mL.flattened;

            foreach (i; 0 .. count)
            {
                double x = -mn[i][0];
                double y = -mn[i][1];

                foreach (j; 0 .. 3)
                {
                    L[24 * i + j] = L[24 * i + 16 + j] = objectPointsD[i][j];
                }
                L[24 * i + 3] = L[24 * i + 19] = 1.0;
                L[24 * i + 4] = L[24 * i + 5] = L[24 * i + 6] = L[24 * i + 7] = 0.0;
                L[24 * i + 12] = L[24 * i + 13] = L[24 * i + 14] = L[24 * i + 15] = 0.0;

                foreach (j; 0 .. 3)
                {
                    L[24 * i + 8 + j] = x * objectPointsD[i][j];
                }

                L[24 * i + 11] = x;

                foreach (j; 0 .. 3)
                {
                    L[24 * i + 20 + j] = y * objectPointsD[i][j];
                }
                L[24 * i + 23] = y;
            }

            auto ll = mL.transposed.mtimes(mL);

            auto llsvd = svd(ll);
            auto lv = llsvd.vt;

            int err;
            auto rrt = lv.flattened[(11 * 12) .. $].reshape([3, 4], err);

            auto rr = rrt[0 .. $, 0 .. 3].rcslice;
            auto tt = rrt[0 .. $, 3];

            if (rr.lightScope.det < 0)
            {
                rrt[] *= -1.0;
            }

            double norm = nrm2(rr.lightScope.flattened);
            assert(norm.abs > DBL_EPSILON);

            auto rrsvd = svd(rr);
            auto mU = rrsvd.u;
            auto mW = rrsvd.sigma;
            auto mV = rrsvd.vt;

            auto mR = rcslice!double(3, 3);
            gemm(1.0, mU.lightScope, mV.lightScope, 0.0, mR.lightScope);

            if (mR.lightScope.det < 0)
            {
                mR[] *= -1.0;
            }

            auto scale = nrm2(mR.lightScope.flattened) / norm;
            _t[] = tt[] * scale;

            import inmath : Matrix, Quaternion;

            auto z = Matrix!(double, 3, 3)(mR.accessFlat(0), mR.accessFlat(1),
                    mR.accessFlat(2), mR.accessFlat(3), mR.accessFlat(4),
                    mR.accessFlat(5), mR.accessFlat(6), mR.accessFlat(7), mR.accessFlat(8),);

            alias quatd = Quaternion!(double);
            auto angle = quatd.fromMatrix(z).toAxisAngle();
            copy(angle.ptr[0 .. 3], _r);

            debug {
                import std.stdio;
                writeln(_r);
            }
        }
    }

    auto param = params.sliced;
    // These don't need to allocate but w/e
    auto l = param.shape.rcslice(-double.infinity);
    auto u = param.shape.rcslice(+double.infinity);

    LeastSquaresSettings!double settings;
    settings.maxIterations = 20;

    // TODO: this is quite inefficient, do something about it
    try
    {
        settings.optimize!((x, y) {
            assert(x.length == 6);
            
            auto rv = x[0..3];
            auto tv = x[3..6];

            int err;
            auto reshaped = y.reshape([-1, 2], err);
            assert(err == 0);

            projectPoints(objectPointsD.lightScope, rv, tv, cameraMatrixD.lightScope, distCoeffs, reshaped);
            reshaped[] = reshaped[] - imagePointsD;

            // debug {
            //     import std.stdio;
            //     writeln(reshaped.shape);
            // }
        }, (x, y) {
            auto rv = x[0..3];
            auto tv = x[3..6];

            int err;
            assert(err == 0);

            double[28] imagePoints;
            auto imageSliced = imagePoints[].sliced(14, 2);

            auto dpdr = y[0..$, 0..3];
            auto dpdt = y[0..$, 3..6];

            projectPoints(objectPointsD.lightScope, rv, tv, cameraMatrixD.lightScope, distCoeffs, imageSliced, dpdr, dpdt);

            // debug {
            //     import std.stdio;
            //     writeln(dpdr);
            //     writeln(dpdt);
            // }
        })(2 * count, param, l.lightScope, u.lightScope);
    } catch (Exception e) {
        if (e.msg.canFind("Maximum number of iterations reached")) {

        } else {
            throw e;
        }
    }

    auto rvec = param[0..3].rcslice;
    auto tvec = param[3..6].rcslice;

    return PnPResult(rvec, tvec);
}

auto det(T, SliceKind kind)(Slice!(const(T)*, 2, kind) a)
in
{
    assert(a.length!0 == a.length!1, "matrix must be square");
}
do
{
    import mir.ndslice.topology : diagonal, zip, iota;
    import mir.math.numeric : ProdAccumulator;
    import mir.lapack : lapackint;
    import mir.lapack : getrf;

    auto m = a.as!T.rcslice.canonical;
    auto ipiv = a.length.mininitRcslice!lapackint;

    // LU factorization
    auto info = m.lightScope.getrf(ipiv.lightScope);

    // If matrix is singular, determinant is zero.
    if (info > 0)
    {
        return cast(T) 0;
    }

    // The determinant is the product of the diagonal entries
    // of the upper triangular matrix. The array ipiv contains
    // the pivots.
    int sign;
    ProdAccumulator!T prod;
    foreach (tup; m.diagonal.zip(ipiv, [ipiv.length].iota(1)))
    {
        prod.put(tup.a);
        sign ^= tup.b != tup.c; // i.e. row interchanged with another
    }
    if (sign & 1)
        prod.x = -prod.x;
    return prod.prod;
}

import core.stdc.float_ : FLT_EPSILON;

import std.algorithm : copy;
import std.array : staticArray;
import std.traits : isFloatingPoint;

import mir.ndslice;

import inmath;
import rodrigues;
import solve : isProperDistCoeffs;

alias mat2d = Matrix!(double, 2, 2);
alias mat3d = Matrix!(double, 3, 3);

@nogc:

private Slice!(T, N, desired) morph(SliceKind desired, T, size_t N)(Slice!(T, N, Contiguous) slice)
{
    static if (desired == Contiguous)
    {
        return slice;
    }
    else static if (desired == Canonical)
    {
        return slice.canonical;
    }
    else
    {
        return slice.universal;
    }
}

// dfmt off
void projectPoints(ObjT, SliceKind objKind, SliceKind rVecKind, SliceKind tVecKind, MatT, SliceKind matKind,
        DistT, SliceKind distKind, SliceKind imgKind, SliceKind dpdrKind = Contiguous, SliceKind dpdtKind = Contiguous)(
        Slice!(const(ObjT)*, 2, objKind) objectPoints,
        Slice!(const(double)*, 1, rVecKind) rVec,
        Slice!(const(double)*, 1, tVecKind) tVec,
        Slice!(const(MatT)*, 2,matKind) cameraMatrix,
        Slice!(const(DistT)*, 1, distKind) distCoeffs,
        Slice!(double*, 2, imgKind) imagePoints,
        Slice!(double*, 2, dpdrKind) dpdr = slice!double(0, 0).morph!dpdrKind,
        Slice!(double*, 2, dpdtKind) dpdt = slice!double(0, 0).morph!dpdrKind,
        Slice!(double*, 2, Contiguous) dpdf = slice!double(0, 0),
        Slice!(double*, 2, Contiguous) dpdc = slice!double(0, 0),
        Slice!(double*, 2, Contiguous) dpdk = slice!double(0, 0),
        Slice!(double*, 2, Contiguous) dpdo = slice!double(0, 0),
        double aspectRatio = 0)
        if (isFloatingPoint!ObjT && isFloatingPoint!MatT && isFloatingPoint!DistT)
    // dfmt on
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

    auto count = objectPoints.shape[0];

    assert(dpdr.elementCount == 0 || dpdr.shape == [2 * count, 3],
            "dp/drot must be [2 * N, 3] or empty");
    assert(dpdt.elementCount == 0 || dpdt.shape == [2 * count, 3],
            "dp/dT must be [2 * N, 3] or empty");
    assert(dpdf.elementCount == 0 || dpdf.shape == [2 * count, 2],
            "dp/dT must be [2 * N, 2] or empty");
    assert(dpdc.elementCount == 0 || dpdc.shape == [2 * count, 2],
            "dp/dc must be [2 * N, 2] or empty");

    assert(dpdo.elementCount == 0 || dpdo.shape == [2 * count, 3 * count],
            "dp/do must be [2 * N, 3 * N] or empty");
}
do
{
    // auto dMatTiltdTauX = mat3d(0, 0, 0, 0, 0, 0, 0, -1, 0);
    // auto dMatTiltdTauY = mat3d(0, 0, 0, 0, 0, 0, 1, 0, 0);
    mat3d matTilt = mat3d.identity;

    auto count = objectPoints.shape[0];

    bool dpdr_p = dpdr.elementCount != 0;
    bool dpdt_p = dpdt.elementCount != 0;
    bool dpdf_p = dpdf.elementCount != 0;
    bool dpdc_p = dpdc.elementCount != 0;

    bool dpdk_p = false;
    if (dpdk.length != 0)
    {
        // if( !CV_IS_MAT(dpdk) ||
        //     (CV_MAT_TYPE(dpdk->type) != CV_32FC1 && CV_MAT_TYPE(dpdk->type) != CV_64FC1) ||
        //     dpdk->rows != count*2 || (dpdk->cols != 14 && dpdk->cols != 12 && dpdk->cols != 8 && dpdk->cols != 5 && dpdk->cols != 4 && dpdk->cols != 2) )
        //     CV_Error( CV_StsBadArg, "dp/df must be 2Nx14, 2Nx12, 2Nx8, 2Nx5, 2Nx4 or 2Nx2 floating-point matrix" );
        dpdk_p = true;
    }
    bool dpdo_p = dpdo.length != 0;

    bool calc_derivatives = dpdr_p || dpdt_p || dpdf_p || dpdc_p || dpdk_p || dpdo_p;

    // matM = objectPoints  
    // _m = imagePoints
    // r = rvec
    // t = tvec
    // _a = cameraMatrix

    auto M = objectPoints.as!double;
    auto m = imagePoints.as!double;
    auto a = cameraMatrix.flattened.as!double;

    double[12] k = staticArray!(0.0.repeat(12));
    copy(distCoeffs.as!double, k[]);

    double fx = a[0];
    double fy = a[4];
    double cx = a[2];
    double cy = a[5];

    auto t = tVec.as!double;

    // TODO:
    double[9] R;
    double[27] dRdr;
    auto res = axisAngleToRotationMatrix(rVec);
    copy(res.matrix.lightScope.flattened, R[]);
    copy(res.jacobian.lightScope.flattened, dRdr[]);

    bool fixedAspectRatio = aspectRatio > FLT_EPSILON;
    if (fixedAspectRatio)
    {
        fx = fy * aspectRatio;
    }

    foreach (i; 0 .. count)
    {
        double X = M[i][0], Y = M[i][1], Z = M[i][2];
        double x = R[0] * X + R[1] * Y + R[2] * Z + t[0];
        double y = R[3] * X + R[4] * Y + R[5] * Z + t[1];
        double z = R[6] * X + R[7] * Y + R[8] * Z + t[2];

        double r2, r4, r6, a1, a2, a3, cdist, icdist2;
        double xd, yd, xd0, yd0, invProj;
        vec3d vecTilt;
        vec3d dVecTilt;
        mat2d dMatTilt;

        double z0 = z;
        z = z ? 1. / z : 1;
        x *= z;
        y *= z;

        r2 = x * x + y * y;
        r4 = r2 * r2;
        r6 = r4 * r2;
        a1 = 2 * x * y;
        a2 = r2 + 2 * x * x;
        a3 = r2 + 2 * y * y;
        cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6;
        icdist2 = 1. / (1 + k[5] * r2 + k[6] * r4 + k[7] * r6);
        xd0 = x * cdist * icdist2 + k[2] * a1 + k[3] * a2 + k[8] * r2 + k[9] * r4;
        yd0 = y * cdist * icdist2 + k[2] * a3 + k[3] * a1 + k[10] * r2 + k[11] * r4;

        // additional distortion by projecting onto a tilt plane
        vecTilt = matTilt * vec3d(xd0, yd0, 1);
        invProj = vecTilt.vector[2] ? 1. / vecTilt.vector[2] : 1;
        xd = invProj * vecTilt.vector[0];
        yd = invProj * vecTilt.vector[1];

        m[i][0] = xd * fx + cx;
        m[i][1] = yd * fy + cy;

        if (calc_derivatives)
        {
            if (dpdc_p)
            {
                dpdc[i * 2][0] = 1;
                dpdc[i * 2][1] = 0; // dp_xdc_x; dp_xdc_y
                dpdc[i * 2 + 1][0] = 0;
                dpdc[i * 2 + 1][1] = 1;
            }

            if (dpdf_p)
            {
                if (fixedAspectRatio)
                {
                    dpdf[i * 2][0] = 0;
                    dpdf[i * 2][1] = xd * aspectRatio; // dp_xdf_x; dp_xdf_y
                    dpdf[i * 2 + 1][0] = 0;
                    dpdf[i * 2 + 1][1] = yd;
                }
                else
                {
                    dpdf[i * 2][0] = xd;
                    dpdf[i * 2][1] = 0;
                    dpdf[i * 2 + 1][0] = 0;
                    dpdf[i * 2 + 1][1] = yd;
                }
            }

            foreach (row; 0 .. 2)
            {
                foreach (col; 0 .. 2)
                {
                    dMatTilt.matrix[row][col] = matTilt.matrix[row][col]
                        * vecTilt.vector[2] - matTilt.matrix[2][col] * vecTilt.vector[row];
                }
            }

            double invProjSquare = (invProj * invProj);
            dMatTilt *= invProjSquare;

            // if (dpdk_p)
            // {
            //     vec2d dXdYd;
            //     int dpdkCols = ?;
            //     dXdYd = dMatTilt * vec2d(x * icdist2 * r2, y * icdist2 * r2);
            //     dpdk[i * 2][0] = fx * dXdYd.vector[0];
            //     dpdk[i * 2 + 1][0] = fy * dXdYd.vector[1];
            //     dXdYd = dMatTilt * vec2d(x * icdist2 * r4, y * icdist2 * r4);
            //     dpdk[i * 2][1] = fx * dXdYd.vector[0];
            //     dpdk[i * 2 + 1][1] = fy * dXdYd.vector[1];
            //     if (dpdkCols > 2)
            //     {
            //         dXdYd = dMatTilt * vec2d(a1, a3);
            //         dpdk_p[2] = fx * dXdYd.vector[0];
            //         dpdk_p[dpdk_step + 2] = fy * dXdYd.vector[1];
            //         dXdYd = dMatTilt * vec2d(a2, a1);
            //         dpdk_p[3] = fx * dXdYd.vector[0];
            //         dpdk_p[dpdk_step + 3] = fy * dXdYd.vector[1];
            //         if (dpdkCols > 4)
            //         {
            //             dXdYd = dMatTilt * vec2d(x * icdist2 * r6, y * icdist2 * r6);
            //             dpdk_p[4] = fx * dXdYd.vector[0];
            //             dpdk_p[dpdk_step + 4] = fy * dXdYd.vector[1];

            //             if (dpdkCols > 5)
            //             {
            //                 dXdYd = dMatTilt * vec2d(x * cdist * (-icdist2) * icdist2 * r2,
            //                         y * cdist * (-icdist2) * icdist2 * r2);
            //                 dpdk_p[5] = fx * dXdYd.vector[0];
            //                 dpdk_p[dpdk_step + 5] = fy * dXdYd.vector[1];
            //                 dXdYd = dMatTilt * vec2d(x * cdist * (-icdist2) * icdist2 * r4,
            //                         y * cdist * (-icdist2) * icdist2 * r4);
            //                 dpdk_p[6] = fx * dXdYd.vector[0];
            //                 dpdk_p[dpdk_step + 6] = fy * dXdYd.vector[1];
            //                 dXdYd = dMatTilt * vec2d(x * cdist * (-icdist2) * icdist2 * r6,
            //                         y * cdist * (-icdist2) * icdist2 * r6);
            //                 dpdk_p[7] = fx * dXdYd.vector[0];
            //                 dpdk_p[dpdk_step + 7] = fy * dXdYd.vector[1];
            //                 if (dpdkCols > 8)
            //                 {
            //                     dXdYd = dMatTilt * vec2d(r2, 0);
            //                     dpdk_p[8] = fx * dXdYd.vector[0]; //s1
            //                     dpdk_p[dpdk_step + 8] = fy * dXdYd.vector[1]; //s1
            //                     dXdYd = dMatTilt * vec2d(r4, 0);
            //                     dpdk_p[9] = fx * dXdYd.vector[0]; //s2
            //                     dpdk_p[dpdk_step + 9] = fy * dXdYd.vector[1]; //s2
            //                     dXdYd = dMatTilt * vec2d(0, r2);
            //                     dpdk_p[10] = fx * dXdYd.vector[0]; //s3
            //                     dpdk_p[dpdk_step + 10] = fy * dXdYd.vector[1]; //s3
            //                     dXdYd = dMatTilt * vec2d(0, r4);
            //                     dpdk_p[11] = fx * dXdYd.vector[0]; //s4
            //                     dpdk_p[dpdk_step + 11] = fy * dXdYd.vector[1]; //s4
            //                     if (dpdkCols > 12)
            //                     {
            //                         dVecTilt = dMatTiltdTauX * vec3d(xd0, yd0, 1);
            //                         dpdk_p[12] = fx * invProjSquare * (
            //                                 dVecTilt.vector[0] * vecTilt.vector[2]
            //                                 - dVecTilt.vector[2] * vecTilt.vector[0]);
            //                         dpdk_p[dpdk_step + 12] = fy * invProjSquare * (
            //                                 dVecTilt.vector[1] * vecTilt.vector[2]
            //                                 - dVecTilt.vector[2] * vecTilt.vector[1]);
            //                         dVecTilt = dMatTiltdTauY * vec3d(xd0, yd0, 1);
            //                         dpdk_p[13] = fx * invProjSquare * (
            //                                 dVecTilt.vector[0] * vecTilt.vector[2]
            //                                 - dVecTilt.vector[2] * vecTilt.vector[0]);
            //                         dpdk_p[dpdk_step + 13] = fy * invProjSquare * (
            //                                 dVecTilt.vector[1] * vecTilt.vector[2]
            //                                 - dVecTilt.vector[2] * vecTilt.vector[1]);
            //                     }
            //                 }
            //             }
            //         }
            //     }
            //     dpdk_p += dpdk_step * 2;
            // }

            if (dpdt_p)
            {
                auto dxdt = [z, 0, -x * z].staticArray!double;
                auto dydt = [0, z, -y * z].staticArray!double;
                foreach (j; 0 .. 3)
                {
                    double dr2dt = 2 * x * dxdt[j] + 2 * y * dydt[j];
                    double dcdist_dt = k[0] * dr2dt + 2 * k[1] * r2 * dr2dt + 3 * k[4] * r4 * dr2dt;
                    double dicdist2_dt = -icdist2 * icdist2 * (
                            k[5] * dr2dt + 2 * k[6] * r2 * dr2dt + 3 * k[7] * r4 * dr2dt);
                    double da1dt = 2 * (x * dydt[j] + y * dxdt[j]);
                    double dmxdt = (
                            dxdt[j] * cdist * icdist2 + x * dcdist_dt * icdist2 + x * cdist * dicdist2_dt + k[2] * da1dt + k[3] * (
                            dr2dt + 4 * x * dxdt[j]) + k[8] * dr2dt + 2 * r2 * k[9] * dr2dt);
                    double dmydt = (
                            dydt[j] * cdist * icdist2 + y * dcdist_dt * icdist2 + y * cdist * dicdist2_dt + k[2] * (
                            dr2dt + 4 * y * dydt[j]) + k[3] * da1dt + k[10]
                            * dr2dt + 2 * r2 * k[11] * dr2dt);
                    vec2d dXdYd = dMatTilt * vec2d(dmxdt, dmydt);
                    dpdt[i * 2][j] = fx * dXdYd.vector[0];
                    dpdt[i * 2 + 1][j] = fy * dXdYd.vector[1];
                }
            }

            if (dpdr_p)
            {
                auto dx0dr = [
                    X * dRdr[0] + Y * dRdr[1] + Z * dRdr[2],
                    X * dRdr[9] + Y * dRdr[10] + Z * dRdr[11],
                    X * dRdr[18] + Y * dRdr[19] + Z * dRdr[20]
                ].staticArray!double;
                auto dy0dr = [
                    X * dRdr[3] + Y * dRdr[4] + Z * dRdr[5],
                    X * dRdr[12] + Y * dRdr[13] + Z * dRdr[14],
                    X * dRdr[21] + Y * dRdr[22] + Z * dRdr[23]
                ].staticArray!double;
                auto dz0dr = [
                    X * dRdr[6] + Y * dRdr[7] + Z * dRdr[8],
                    X * dRdr[15] + Y * dRdr[16] + Z * dRdr[17],
                    X * dRdr[24] + Y * dRdr[25] + Z * dRdr[26]
                ].staticArray!double;

                foreach (j; 0 .. 3)
                {
                    double dxdr = z * (dx0dr[j] - x * dz0dr[j]);
                    double dydr = z * (dy0dr[j] - y * dz0dr[j]);
                    double dr2dr = 2 * x * dxdr + 2 * y * dydr;
                    double dcdist_dr = (k[0] + 2 * k[1] * r2 + 3 * k[4] * r4) * dr2dr;
                    double dicdist2_dr = -icdist2 * icdist2 * (
                            k[5] + 2 * k[6] * r2 + 3 * k[7] * r4) * dr2dr;
                    double da1dr = 2 * (x * dydr + y * dxdr);
                    double dmxdr = (
                            dxdr * cdist * icdist2 + x * dcdist_dr * icdist2 + x * cdist * dicdist2_dr + k[2] * da1dr + k[3] * (
                            dr2dr + 4 * x * dxdr) + (k[8] + 2 * r2 * k[9]) * dr2dr);
                    double dmydr = (dydr * cdist * icdist2 + y * dcdist_dr * icdist2 + y * cdist * dicdist2_dr + k[2] * (
                            dr2dr + 4 * y * dydr) + k[3] * da1dr + (k[10] + 2 * r2 * k[11]) * dr2dr);
                    vec2d dXdYd = dMatTilt * vec2d(dmxdr, dmydr);
                    dpdr[i * 2][j] = fx * dXdYd.vector[0];
                    dpdr[i * 2 + 1][j] = fy * dXdYd.vector[1];
                }
            }

            if (dpdo_p)
            {
                auto dxdo = [
                    z * (R[0] - x * z * z0 * R[6]), z * (R[1] - x * z * z0 * R[7]),
                    z * (R[2] - x * z * z0 * R[8])
                ].staticArray!double;
                auto dydo = [
                    z * (R[3] - y * z * z0 * R[6]), z * (R[4] - y * z * z0 * R[7]),
                    z * (R[5] - y * z * z0 * R[8])
                ].staticArray!double;

                foreach (j; 0 .. 3)
                {
                    double dr2do = 2 * x * dxdo[j] + 2 * y * dydo[j];
                    double dr4do = 2 * r2 * dr2do;
                    double dr6do = 3 * r4 * dr2do;
                    double da1do = 2 * y * dxdo[j] + 2 * x * dydo[j];
                    double da2do = dr2do + 4 * x * dxdo[j];
                    double da3do = dr2do + 4 * y * dydo[j];
                    double dcdist_do = k[0] * dr2do + k[1] * dr4do + k[4] * dr6do;
                    double dicdist2_do = -icdist2 * icdist2 * (
                            k[5] * dr2do + k[6] * dr4do + k[7] * dr6do);
                    double dxd0_do = cdist * icdist2 * dxdo[j] + x * icdist2
                        * dcdist_do + x * cdist * dicdist2_do + k[2] * da1do + k[3]
                        * da2do + k[8] * dr2do + k[9] * dr4do;
                    double dyd0_do = cdist * icdist2 * dydo[j] + y * icdist2
                        * dcdist_do + y * cdist * dicdist2_do + k[2] * da3do + k[3]
                        * da1do + k[10] * dr2do + k[11] * dr4do;
                    vec2d dXdYd = dMatTilt * vec2d(dxd0_do, dyd0_do);
                    dpdo[i * 2][i * 3 + j] = fx * dXdYd.vector[0];
                    dpdo[i * 2 + 1][i * 3 + j] = fy * dXdYd.vector[1];
                }
            }
        }
    }
}

import core.stdc.float_ : DBL_EPSILON;

import std.algorithm : copy, swap;
import std.array : staticArray;
import std.math : fabs;
import std.range : repeat;
import std.traits : isFloatingPoint;

import mir.blas : nrm2;
import mir.ndslice;
import mir.rc.array : RCI;
import kaleidic.lubeck2 : eye, mtimes, svd;

import inmath;

@nogc:

alias mat3d = Matrix!(double, 3, 3);

struct AxisAngleResults(T)
{
    Slice!(RCI!T, 1) axisAngle;
    Slice!(RCI!double, 2) jacobian;
}

AxisAngleResults!T rotationMatrixToAxisAngle(T)(Slice!(const(T)*, 2, Contiguous) matrix)
        if (isFloatingPoint!T)
{
    alias vec3t = Vector!(T, 3);

    auto res = svd(matrix);

    auto U = res.u;
    auto Vt = res.vt;

    auto R = U * Vt;

    vec3t r = vec3t(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]);
    double s = sqrt((r.x * r.x + r.y * r.y + r.z * r.z) * 0.25);

    double c = (R[0, 0] + R[1, 1] + R[2, 2] - 1) * 0.5;
    c = c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c;

    double theta = acos(c);

    if (s < 1e-5)
    {
        double t;

        if (c > 0)
        {
            r = vec3t(0, 0, 0);
        }
        else
        {
            t = (R[0, 0] + 1) * 0.5;
            r.x = sqrt(max(t, 0.0));
            t = (R[1, 1] + 1) * 0.5;
            r.y = sqrt(max(t, 0.0)) * (R[0, 1] < 0 ? -1.0 : 1.0);
            t = (R[2, 2] + 1) * 0.5;
            r.z = sqrt(max(t, 0.0)) * (R[0, 2] < 0 ? -1.0 : 1.0);
            if (fabs(r.x) < fabs(r.y) && fabs(r.x) < fabs(r.z) && (R[1, 2] > 0) != (r.y * r.z > 0))
                r.z = -r.z;
            theta /= nrm2(r.vector.sliced);
            r *= theta;
        }

        double[27] J = staticArray!(0.0.repeat(27));
        if (c > 0)
        {
            J[5] = J[15] = J[19] = -0.5;
            J[7] = J[11] = J[21] = 0.5;
        }

        int err;
        auto jacobian = J.rcslice.reshape([3, 9], err);
        assert(err == 0);
        auto ret = AxisAngleResults!T(r.vector.sliced.rcslice, jacobian);

        return ret;
    }
    else
    {
        double vth = 1 / (2 * s);

        double t, dtheta_dtr = -1.0 / s;
        // var1 = [vth;theta]
        // var = [om1;var1] = [om1;vth;theta]
        double dvth_dtheta = -vth * c / s;
        double d1 = 0.5 * dvth_dtheta * dtheta_dtr;
        double d2 = 0.5 * dtheta_dtr;

        // dfmt off

        // dvar1/dR = dvar1/dtheta*dtheta/dR = [dvth/dtheta; 1] * dtheta/dtr * dtr/dR
        double[5 * 9] dvardR = [
            0, 0, 0, 0, 0, 1, 0, -1, 0,
            0, 0, -1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, -1, 0, 0, 0, 0, 0,
            d1, 0, 0, 0, d1, 0, 0, 0, d1,
            d2, 0, 0, 0, d2, 0, 0, 0, d2
        ];

        // var2 = [om;theta]
        double[4 * 5] dvar2dvar = [
            vth, 0, 0, r.x, 0,
            0, vth, 0, r.y, 0,
            0, 0, vth, r.z, 0,
            0, 0, 0, 0, 1
        ];

        double[3 * 4] domegadvar2 = [
            theta, 0, 0, r.x*vth,
            0, theta, 0, r.y*vth,
            0, 0, theta, r.z*vth
        ];

        // dfmt on

        int err;
        auto mdvardR = dvardR.sliced.reshape([5, 9], err);
        assert(err == 0);

        auto mdvar2dvar = dvar2dvar.sliced.reshape([4, 5], err);
        assert(err == 0);

        auto mdomegadvar2 = domegadvar2.sliced.reshape([3, 4], err);
        assert(err == 0);

        auto t0 = mtimes(mdomegadvar2, mdvar2dvar);
        auto matJ = mtimes(t0, mdvardR);
        // copy(matJ.lightScope, J);

        // transpose every row of matJ (treat the rows as 3x3 matrices)
        auto J = matJ.lightScope.flattened;
        swap(J[1], J[3]);
        swap(J[2], J[6]);
        swap(J[5], J[7]);
        swap(J[10], J[12]);
        swap(J[11], J[15]);
        swap(J[14], J[16]);
        swap(J[19], J[21]);
        swap(J[20], J[24]);
        swap(J[23], J[25]);

        vth *= theta;
        r *= vth;

        return AxisAngleResults!T(r.vector.sliced.rcslice, matJ);
    }
}

struct RotationMatrixResults(T)
{
    Slice!(RCI!T, 2) matrix;
    Slice!(RCI!double, 2) jacobian;
}

RotationMatrixResults!T axisAngleToRotationMatrix(T)(Slice!(const(T)*, 1, Contiguous) vector)
        if (isFloatingPoint!T)
{
    alias vec3t = Vector!(T, 3);
    double[27] J = staticArray!(0.0.repeat(27));

    vec3t r = vec3t(vector[0], vector[1], vector[2]);

    auto mat = eye(3);
    double theta = nrm2(r.vector.sliced);

    if (theta < DBL_EPSILON)
    {
        J[5] = J[15] = J[19] = -1;
        J[7] = J[11] = J[21] = 1;
    }
    else
    {
        double c = cos(theta);
        double s = sin(theta);
        double c1 = 1. - c;
        double itheta = theta ? 1. / theta : 0.;

        r *= itheta;

        mat3d rrt = mat3d(r.x * r.x, r.x * r.y, r.x * r.z, r.x * r.y, r.y * r.y,
                r.y * r.z, r.x * r.z, r.y * r.z, r.z * r.z);
        mat3d r_x = mat3d(0, -r.z, r.y, r.z, 0, -r.x, -r.y, r.x, 0);

        // R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
        mat3d R = c * mat3d.identity + c1 * rrt + s * r_x;
        foreach (i; 0 .. 3)
        {
            foreach (j; 0 .. 3)
            {
                mat[i][j] = R.matrix[i][j];
            }
        }

        auto I = [1, 0, 0, 0, 1, 0, 0, 0, 1].staticArray!double;
        auto drrt = [
            r.x + r.x, r.y, r.z, r.y, 0, 0, r.z, 0, 0, 0, r.x, 0, r.x,
            r.y + r.y, r.z, 0, r.z, 0, 0, 0, r.x, 0, 0, r.y, r.x, r.y, r.z + r.z
        ].staticArray!double;
        auto d_r_x_ = [
            0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 1, 0,
            0, 0, 0, 0
        ].staticArray!double;

        foreach (i; 0 .. 3)
        {
            double ri = i == 0 ? r.x : i == 1 ? r.y : r.z;
            double a0 = -s * ri, a1 = (s - 2 * c1 * itheta) * ri, a2 = c1 * itheta;
            double a3 = (c - s * itheta) * ri, a4 = s * itheta;
            for (int k = 0; k < 9; k++)
                J[i * 9 + k] = a0 * I[k] + a1 * rrt[k / 3][k % 3] + a2
                    * drrt[i * 9 + k] + a3 * r_x[k / 3][k % 3] + a4 * d_r_x_[i * 9 + k];
        }
    }

    int err;
    auto jacobian = J.rcslice.reshape([3, 9], err);
    assert(err == 0);
    auto ret = RotationMatrixResults!T(mat, jacobian);

    return ret;
}

// if( src->cols == 1 || src->rows == 1 )
//     {
//         int step = src->rows > 1 ? src->step / elem_size : 1;

//         if( src->rows + src->cols*CV_MAT_CN(src->type) - 1 != 3 )
//             CV_Error( CV_StsBadSize, "Input matrix must be 1x3, 3x1 or 3x3" );

//         if( dst->rows != 3 || dst->cols != 3 || CV_MAT_CN(dst->type) != 1 )
//             CV_Error( CV_StsBadSize, "Output matrix must be 3x3, single-channel floating point matrix" );

//         Point3d r;
//         if( depth == CV_32F )
//         {
//             r.x = src->data.fl[0];
//             r.y = src->data.fl[step];
//             r.z = src->data.fl[step*2];
//         }
//         else
//         {
//             r.x = src->data.db[0];
//             r.y = src->data.db[step];
//             r.z = src->data.db[step*2];
//         }

//         double theta = norm(r);

//         if( theta < DBL_EPSILON )
//         {
//             cvSetIdentity( dst );

//             if( jacobian )
//             {
//                 memset( J, 0, sizeof(J) );
//                 J[5] = J[15] = J[19] = -1;
//                 J[7] = J[11] = J[21] = 1;
//             }
//         }
//         else
//         {
//             double c = cos(theta);
//             double s = sin(theta);
//             double c1 = 1. - c;
//             double itheta = theta ? 1./theta : 0.;

//             r *= itheta;

//             Matx33d rrt( r.x*r.x, r.x*r.y, r.x*r.z, r.x*r.y, r.y*r.y, r.y*r.z, r.x*r.z, r.y*r.z, r.z*r.z );
//             Matx33d r_x(    0, -r.z,  r.y,
//                           r.z,    0, -r.x,
//                          -r.y,  r.x,    0 );

//             // R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
//             Matx33d R = c*Matx33d::eye() + c1*rrt + s*r_x;

//             Mat(R).convertTo(cvarrToMat(dst), dst->type);

//             if( jacobian )
//             {
//                 const double I[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
//                 double drrt[] = { r.x+r.x, r.y, r.z, r.y, 0, 0, r.z, 0, 0,
//                                   0, r.x, 0, r.x, r.y+r.y, r.z, 0, r.z, 0,
//                                   0, 0, r.x, 0, 0, r.y, r.x, r.y, r.z+r.z };
//                 double d_r_x_[] = { 0, 0, 0, 0, 0, -1, 0, 1, 0,
//                                     0, 0, 1, 0, 0, 0, -1, 0, 0,
//                                     0, -1, 0, 1, 0, 0, 0, 0, 0 };
//                 for( int i = 0; i < 3; i++ )
//                 {
//                     double ri = i == 0 ? r.x : i == 1 ? r.y : r.z;
//                     double a0 = -s*ri, a1 = (s - 2*c1*itheta)*ri, a2 = c1*itheta;
//                     double a3 = (c - s*itheta)*ri, a4 = s*itheta;
//                     for( int k = 0; k < 9; k++ )
//                         J[i*9+k] = a0*I[k] + a1*rrt.val[k] + a2*drrt[i*9+k] +
//                                    a3*r_x.val[k] + a4*d_r_x_[i*9+k];
//                 }
//             }
//         }
//     }

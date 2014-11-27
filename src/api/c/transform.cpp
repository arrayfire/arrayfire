/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include <af/defines.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <ArrayInfo.hpp>
#include <backend.hpp>
#include <transform.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array transform(const af_array in, const af_array tf, const af::dim4 &odims,
                                 const af_interp_type method, const bool inverse)
{
    return getHandle(*transform<T>(getArray<T>(in), getArray<float>(tf), odims, method, inverse));
}

af_err af_transform(af_array *out, const af_array in, const af_array tf,
                    const dim_type odim0, const dim_type odim1,
                    const af_interp_type method, const bool inverse)
{
    try {
        ArrayInfo t_info = getInfo(tf);
        ArrayInfo i_info = getInfo(in);

        af::dim4 idims = i_info.dims();
        af::dim4 tdims = t_info.dims();
        af_dtype itype = i_info.getType();

        ARG_ASSERT(2, t_info.getType() == f32);
        ARG_ASSERT(5, method == AF_INTERP_NEAREST || method == AF_INTERP_BILINEAR);
        DIM_ASSERT(2, (tdims[0] == 3 && tdims[1] == 2));
        DIM_ASSERT(1, idims.elements() > 0);
        DIM_ASSERT(1, (idims.ndims() == 2 || idims.ndims() == 3));

        dim_type o0 = odim0, o1 = odim1;
        dim_type o2 = idims[2] * tdims[2];
        if (odim0 * odim1 == 0) {
            o0 = idims[0];
            o1 = idims[1];
        }
        af::dim4 odims(o0, o1, o2, 1);

        af_array output = 0;
        switch(itype) {
            case f32: output = transform<float  >(in, tf, odims, method, inverse);  break;
            case f64: output = transform<double >(in, tf, odims, method, inverse);  break;
            case s32: output = transform<int    >(in, tf, odims, method, inverse);  break;
            case u32: output = transform<uint   >(in, tf, odims, method, inverse);  break;
            case u8:  output = transform<uchar  >(in, tf, odims, method, inverse);  break;
            default:  TYPE_ERROR(1, itype);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_rotate(af_array *out, const af_array in, const float theta, const af_interp_type method,
                 const bool crop, const bool recenter)
{
    af_err ret = AF_SUCCESS;
    try {
        unsigned odims0 = 0, odims1 = 0;
        float c = std::cos(-theta), s = std::sin(-theta);
        float tx = 0, ty = 0;

        ArrayInfo info = getInfo(in);
        af::dim4 idims = info.dims();

        if(!crop) {
            odims0 = idims[0] * fabs(std::cos(theta)) + idims[1] * fabs(std::sin(theta));
            odims1 = idims[1] * fabs(std::cos(theta)) + idims[0] * fabs(std::sin(theta));
        } else {
            odims0 = idims[0];
            odims1 = idims[1];
        }

        if (recenter) { //Find new coordintates of center and translate it
            float nx = 0.5 * (idims[0] - 1);
            float ny = 0.5 * (idims[1] - 1);
            float mx = 0.5 * (odims0 - 1);
            float my = 0.5 * (odims1 - 1);
            float sx = (mx * c + my *-s);
            float sy = (mx * s + my * c);
            tx = -(sx - nx);
            ty = -(sy - ny);
        }

        //Correct transform matrix for forward rotation
        static float trans_mat[6] = {1, 0, 0,
                                     0, 1, 0};
        trans_mat[0] =  c;
        trans_mat[1] = -s;
        trans_mat[2] = tx;
        trans_mat[3] =  s;
        trans_mat[4] =  c;
        trans_mat[5] = ty;

        //If inverse, generated inverse matrix
        //if(inverse)
        //    calc_transform_inverse(trans_mat);

        static af::dim4 tdims(3, 2, 1, 1);
        af_array t = 0;
        ret = af_create_array(&t, trans_mat, tdims.ndims(), tdims.get(), f32);

        if (ret == AF_SUCCESS) {
            return af_transform(out, in, t, odims0, odims1, method, true);
        }
    }
    CATCHALL;

    return ret;
}

af_err af_translate(af_array *out, const af_array in, const float trans0, const float trans1,
                    const dim_type odim0, const dim_type odim1, const af_interp_type method)
{
    af_err ret = AF_SUCCESS;

    try {
        static float trans_mat[6] = {1, 0, 0,
                                     0, 1, 0};
        trans_mat[2] = trans0;
        trans_mat[5] = trans1;

        static af::dim4 tdims(3, 2, 1, 1);
        af_array t = 0;

        ret = af_create_array(&t, trans_mat, tdims.ndims(), tdims.get(), f32);

        if (ret == AF_SUCCESS) {
            ret = af_transform(out, in, t, odim0, odim1, method, true);
        }
    }
    CATCHALL;

    return ret;
}

af_err af_scale(af_array *out, const af_array in, const float scale0, const float scale1,
                    const dim_type odim0, const dim_type odim1, const af_interp_type method)
{
    af_err ret = AF_SUCCESS;
    try {
        ArrayInfo i_info = getInfo(in);
        af::dim4 idims = i_info.dims();

        dim_type _odim0 = odim0, _odim1 = odim1;
        float sx, sy;
        if(_odim0 == 0 && _odim1 == 0) {
            sx = 1.f / scale0, sy = 1.f / scale1;
            _odim0 = idims[0] / sx;
            _odim1 = idims[1] / sy;
        } else if ( _odim0 == 0 || _odim1 == 0) {
            return AF_ERR_SIZE;
        } else if (scale0 == 0 && scale1 == 0) {
            sx = idims[0] / (float)_odim0;
            sy = idims[1] / (float)_odim1;
        } else {
            sx = 1.f / scale0, sy = 1.f / scale1;
        }

        static float trans_mat[6] = {1, 0, 0,
                                     0, 1, 0};
        trans_mat[0] = sx;
        trans_mat[4] = sy;

        static af::dim4 tdims(3, 2, 1, 1);
        af_array t = 0;
        ret = af_create_array(&t, trans_mat, tdims.ndims(), tdims.get(), f32);

        if (ret == AF_SUCCESS) {
            return af_transform(out, in, t, _odim0, _odim1, method, true);
        }
    }
    CATCHALL;

    return ret;
}

af_err af_skew(af_array *out, const af_array in, const float skew0, const float skew1,
               const dim_type odim0, const dim_type odim1, const af_interp_type method,
               const bool inverse)
{
    af_err ret = AF_SUCCESS;
    try {
        float tx = std::tan(skew0);
        float ty = std::tan(skew1);

        static float trans_mat[6] = {1, 0, 0,
                                     0, 1, 0};
        trans_mat[1] = ty;
        trans_mat[3] = tx;

        if(inverse) {
            if(tx == 0 || ty == 0) {
                trans_mat[1] = tx;
                trans_mat[3] = ty;
            } else {
                //calc_tranform_inverse(trans_mat);
                //short cut of calc_transform_inverse
                float d = 1.0f / (1.0f - tx * ty);
                trans_mat[0] = d;
                trans_mat[1] = ty * d;
                trans_mat[3] = tx * d;
                trans_mat[4] = d;
            }
        }
        static af::dim4 tdims(3, 2, 1, 1);
        af_array t = 0;
        ret = af_create_array(&t, trans_mat, tdims.ndims(), tdims.get(), f32);

        if (ret == AF_SUCCESS) {
            return af_transform(out, in, t, odim0, odim1, method, true);
        }
    }
    CATCHALL;

    return ret;
}

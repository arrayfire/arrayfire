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
#include <common/err_common.hpp>
#include <handle.hpp>
#include <common/ArrayInfo.hpp>
#include <backend.hpp>
#include <transform.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array transform(const af_array in, const af_array tf, const af::dim4 &odims,
                                 const af_interp_type method, const bool inverse, const bool perspective)
{
    return getHandle(transform<T>(getArray<T>(in), getArray<float>(tf), odims, method, inverse, perspective));
}

AF_BATCH_KIND getTransformBatchKind(const dim4 &iDims, const dim4 &tDims)
{
    static const int baseDim = 2;

    dim_t iNd = iDims.ndims();
    dim_t tNd = tDims.ndims();

    if (iNd == baseDim && tNd == baseDim)
        return AF_BATCH_NONE;
    else if (iNd == baseDim && tNd <= 4)
        return AF_BATCH_RHS;
    else if (iNd <= 4 && tNd == baseDim)
        return AF_BATCH_LHS;
    else if (iNd <= 4 && tNd <= 4) {
        bool dimsMatch = true;
        bool isInterleaved = true;
        for (dim_t i = baseDim; i < 4; i++) {
            dimsMatch &= (iDims[i] == tDims[i]);
            isInterleaved &= (iDims[i] == 1 || tDims[i] == 1 || iDims[i] == tDims[i]);
        }
        if (dimsMatch) return AF_BATCH_SAME;
        return (isInterleaved ? AF_BATCH_DIFF : AF_BATCH_UNSUPPORTED);
    }
    else
        return AF_BATCH_UNSUPPORTED;
}

af_err af_transform(af_array *out, const af_array in, const af_array tf,
                    const dim_t odim0, const dim_t odim1,
                    const af_interp_type method, const bool inverse)
{
    try {
        const ArrayInfo& t_info = getInfo(tf);
        const ArrayInfo& i_info = getInfo(in);

        af::dim4 idims = i_info.dims();
        af::dim4 tdims = t_info.dims();
        af_dtype itype = i_info.getType();

        // Assert type and interpolation
        ARG_ASSERT(2, t_info.getType() == f32);
        ARG_ASSERT(5, method == AF_INTERP_NEAREST  ||
                      method == AF_INTERP_BILINEAR ||
                      method == AF_INTERP_BILINEAR_COSINE ||
                      method == AF_INTERP_BICUBIC ||
                      method == AF_INTERP_BICUBIC_SPLINE ||
                      method == AF_INTERP_LOWER);

        // Assert dimesions
        // Image can be 2D or higher
        DIM_ASSERT(1, idims.elements() > 0);
        DIM_ASSERT(1, idims.ndims() >= 2);

        // Transform can be 3x2 for affine transform or 3x3 for perspective transform
        DIM_ASSERT(2, (tdims[0] == 3 && (tdims[1] == 2 || tdims[1] == 3)));

        // If transform is batched, the output dimensions must be specified
        if(tdims[2] * tdims[3] > 1) {
            ARG_ASSERT(3, odim0 > 0);
            ARG_ASSERT(4, odim1 > 0);
        }

        // If idims[2] > 1 and tdims[2] > 1, then both must be equal
        // else at least one of them must be 1
        if(tdims[2] != 1 && idims[2] != 1)
            DIM_ASSERT(2, idims[2] == tdims[2]);
        else
            DIM_ASSERT(2, idims[2] == 1 || tdims[2] == 1);

        // If idims[3] > 1 and tdims[3] > 1, then both must be equal
        // else at least one of them must be 1
        if(tdims[3] != 1 && idims[3] != 1)
            DIM_ASSERT(2, idims[3] == tdims[3]);
        else
            DIM_ASSERT(2, idims[3] == 1 || tdims[3] == 1);

        const bool perspective = (tdims[1] == 3);
        dim_t o0 = odim0, o1 = odim1, o2 = 0, o3 = 0;
        if (odim0 * odim1 == 0) {
            o0 = idims[0];
            o1 = idims[1];
        }

        switch(getTransformBatchKind(idims, tdims)) {
            case AF_BATCH_NONE:     // Both are exactly 2D
            case AF_BATCH_LHS:   // Image is 3/4D, transform is 2D
            case AF_BATCH_SAME:     // Both are 3/4D and have the same dims
                o2 = idims[2];
                o3 = idims[3];
                break;
            case AF_BATCH_RHS:   // Image is 2D, transform is 3/4D
                o2 = tdims[2];
                o3 = tdims[3];
                break;
            case AF_BATCH_DIFF:     // Both are 3/4D, but have different dims
                o2 = idims[2] == 1 ? tdims[2] : idims[2];
                o3 = idims[3] == 1 ? tdims[3] : idims[3];
                break;
            case AF_BATCH_UNSUPPORTED:
            default:
                AF_ERROR("Unsupported combination of batching parameters in transform",
                          AF_ERR_NOT_SUPPORTED);
                break;
        }

        af::dim4 odims(o0, o1, o2, o3);

        af_array output = 0;
        switch(itype) {
            case f32: output = transform<float  >(in, tf, odims, method, inverse, perspective);  break;
            case f64: output = transform<double >(in, tf, odims, method, inverse, perspective);  break;
            case c32: output = transform<cfloat >(in, tf, odims, method, inverse, perspective);  break;
            case c64: output = transform<cdouble>(in, tf, odims, method, inverse, perspective);  break;
            case s32: output = transform<int    >(in, tf, odims, method, inverse, perspective);  break;
            case u32: output = transform<uint   >(in, tf, odims, method, inverse, perspective);  break;
            case s64: output = transform<intl   >(in, tf, odims, method, inverse, perspective);  break;
            case u64: output = transform<uintl  >(in, tf, odims, method, inverse, perspective);  break;
            case s16: output = transform<short  >(in, tf, odims, method, inverse, perspective);  break;
            case u16: output = transform<ushort >(in, tf, odims, method, inverse, perspective);  break;
            case u8:  output = transform<uchar  >(in, tf, odims, method, inverse, perspective);  break;
            case b8:  output = transform<char   >(in, tf, odims, method, inverse, perspective);  break;
            default:  TYPE_ERROR(1, itype);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_translate(af_array *out, const af_array in, const float trans0, const float trans1,
                    const dim_t odim0, const dim_t odim1, const af_interp_type method)
{

    try {
        float trans_mat[6] = {1, 0, 0,
                              0, 1, 0};
        trans_mat[2] = trans0;
        trans_mat[5] = trans1;

        const af::dim4 tdims(3, 2, 1, 1);
        af_array t = 0;

        AF_CHECK(af_create_array(&t, trans_mat, tdims.ndims(), tdims.get(), f32));
        AF_CHECK(af_transform(out, in, t, odim0, odim1, method, true));
        AF_CHECK(af_release_array(t));
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_scale(af_array *out, const af_array in, const float scale0, const float scale1,
                const dim_t odim0, const dim_t odim1, const af_interp_type method)
{
    try {
        const ArrayInfo& i_info = getInfo(in);
        af::dim4 idims = i_info.dims();

        dim_t _odim0 = odim0, _odim1 = odim1;
        float sx, sy;

        if(_odim0 == 0 || _odim1 == 0) {

            DIM_ASSERT(2, scale0 != 0);
            DIM_ASSERT(3, scale1 != 0);

            sx = 1.f / scale0, sy = 1.f / scale1;
            _odim0 = idims[0] / sx;
            _odim1 = idims[1] / sy;

        } else if (scale0 == 0 || scale1 == 0) {

            DIM_ASSERT(4, odim0 != 0);
            DIM_ASSERT(5, odim1 != 0);

            sx = idims[0] / (float)_odim0;
            sy = idims[1] / (float)_odim1;

        } else {

            sx = 1.f / scale0, sy = 1.f / scale1;
        }

        float trans_mat[6] = {1, 0, 0,
                              0, 1, 0};
        trans_mat[0] = sx;
        trans_mat[4] = sy;

        const af::dim4 tdims(3, 2, 1, 1);
        af_array t = 0;
        AF_CHECK(af_create_array(&t, trans_mat, tdims.ndims(), tdims.get(), f32));
        AF_CHECK(af_transform(out, in, t, _odim0, _odim1, method, true));
        AF_CHECK(af_release_array(t));
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_skew(af_array *out, const af_array in, const float skew0, const float skew1,
               const dim_t odim0, const dim_t odim1, const af_interp_type method,
               const bool inverse)
{
    try {
        float tx = std::tan(skew0);
        float ty = std::tan(skew1);

        float trans_mat[6] = {1, 0, 0,
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
        const af::dim4 tdims(3, 2, 1, 1);
        af_array t = 0;
        AF_CHECK(af_create_array(&t, trans_mat, tdims.ndims(), tdims.get(), f32));
        AF_CHECK(af_transform(out, in, t, odim0, odim1, method, true));
        AF_CHECK(af_release_array(t));
    }
    CATCHALL;

    return AF_SUCCESS;
}

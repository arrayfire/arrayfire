/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

#ifdef __cplusplus
namespace af
{
    class array;


    AFAPI array constant(double val, const dim4 &dims, af_dtype ty=f32);
    AFAPI array constant(af_cdouble val, const dim4 &dims);
    AFAPI array constant(af_cfloat val, const dim4 &dims);

    AFAPI array constant(double val, const dim_type d0, af_dtype ty=f32);
    AFAPI array constant(double val, const dim_type d0,
                         const dim_type d1, af_dtype ty=f32);
    AFAPI array constant(double val, const dim_type d0,
                         const dim_type d1, const dim_type d2, af_dtype ty=f32);
    AFAPI array constant(double val, const dim_type d0,
                         const dim_type d1, const dim_type d2,
                         const dim_type d3, af_dtype ty=f32);


    AFAPI array randu(const dim4 &dims, af_dtype ty=f32);
    AFAPI array randu(const dim_type d0, af_dtype ty=f32);
    AFAPI array randu(const dim_type d0,
                      const dim_type d1, af_dtype ty=f32);
    AFAPI array randu(const dim_type d0,
                      const dim_type d1, const dim_type d2, af_dtype ty=f32);
    AFAPI array randu(const dim_type d0,
                      const dim_type d1, const dim_type d2,
                      const dim_type d3, af_dtype ty=f32);


    AFAPI array randn(const dim4 &dims, af_dtype ty=f32);
    AFAPI array randn(const dim_type d0, af_dtype ty=f32);
    AFAPI array randn(const dim_type d0,
                      const dim_type d1, af_dtype ty=f32);
    AFAPI array randn(const dim_type d0,
                      const dim_type d1, const dim_type d2, af_dtype ty=f32);
    AFAPI array randn(const dim_type d0,
                      const dim_type d1, const dim_type d2,
                      const dim_type d3, af_dtype ty=f32);

    AFAPI array iota(const dim4 &dims, const unsigned rep = 3, af_dtype ty=f32);
    AFAPI array iota(const dim_type d0, const unsigned rep = 0, af_dtype ty=f32);
    AFAPI array iota(const dim_type d0, const dim_type d1,
                     const unsigned rep = 1, af_dtype ty=f32);
    AFAPI array iota(const dim_type d0, const dim_type d1, const dim_type d2,
                     const unsigned rep = 2, af_dtype ty=f32);
    AFAPI array iota(const dim_type d0, const dim_type d1, const dim_type d2,
                     const dim_type d3, const unsigned rep = 3, af_dtype ty=f32);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    // Create af_array from a constant value
    AFAPI af_err af_constant(af_array *arr, const double val, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    AFAPI af_err af_constant_c64(af_array *arr, const void* val,
                                 const unsigned ndims, const dim_type * const dims);

    AFAPI af_err af_constant_c32(af_array *arr, const void* val,
                                 const unsigned ndims, const dim_type * const dims);

    // Create sequence array
    AFAPI af_err af_iota(af_array *arr, const unsigned ndims, const dim_type * const dims,
                         const unsigned rep, const af_dtype type);

    // Generate Random Numbers using uniform distribution
    AFAPI af_err af_randu(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // Generate Random Numbers using normal distribution
    AFAPI af_err af_randn(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type);


#ifdef __cplusplus
}
#endif

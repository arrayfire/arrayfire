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


#define CONSTANT(TYPE, TY)                                          \
    AFAPI array constant(TYPE val, const dim4 &dims, dtype ty=TY);  \
    AFAPI array constant(TYPE val, const dim_type d0, dtype ty=TY); \
    AFAPI array constant(TYPE val, const dim_type d0,               \
                         const dim_type d1, dtype ty=TY);           \
    AFAPI array constant(TYPE val, const dim_type d0,               \
                         const dim_type d1, const dim_type d2,      \
                         dtype ty=TY);                              \
    AFAPI array constant(TYPE val, const dim_type d0,               \
                         const dim_type d1, const dim_type d2,      \
                         const dim_type d3, dtype ty=TY);           \

    CONSTANT(double             , f32)
    CONSTANT(float              , f32)
    CONSTANT(int                , f32)
    CONSTANT(unsigned           , f32)
    CONSTANT(char               , f32)
    CONSTANT(unsigned char      , f32)
    CONSTANT(cfloat             , c32)
    CONSTANT(cdouble            , c64)
    CONSTANT(long               , s64)
    CONSTANT(unsigned long      , u64)
    CONSTANT(long long          , s64)
    CONSTANT(unsigned long long , u64)
    CONSTANT(bool               ,  b8)

#undef CONSTANT

    AFAPI array randu(const dim4 &dims, dtype ty=f32);
    AFAPI array randu(const dim_type d0, dtype ty=f32);
    AFAPI array randu(const dim_type d0,
                      const dim_type d1, dtype ty=f32);
    AFAPI array randu(const dim_type d0,
                      const dim_type d1, const dim_type d2, dtype ty=f32);
    AFAPI array randu(const dim_type d0,
                      const dim_type d1, const dim_type d2,
                      const dim_type d3, dtype ty=f32);

    AFAPI array randn(const dim4 &dims, dtype ty=f32);
    AFAPI array randn(const dim_type d0, dtype ty=f32);
    AFAPI array randn(const dim_type d0,
                      const dim_type d1, dtype ty=f32);
    AFAPI array randn(const dim_type d0,
                      const dim_type d1, const dim_type d2, dtype ty=f32);
    AFAPI array randn(const dim_type d0,
                      const dim_type d1, const dim_type d2,
                      const dim_type d3, dtype ty=f32);

    AFAPI array identity(const dim4 &dims, dtype ty=f32);
    AFAPI array identity(const dim_type d0, dtype ty=f32);
    AFAPI array identity(const dim_type d0,
                         const dim_type d1, dtype ty=f32);
    AFAPI array identity(const dim_type d0,
                         const dim_type d1, const dim_type d2, dtype ty=f32);
    AFAPI array identity(const dim_type d0,
                         const dim_type d1, const dim_type d2,
                         const dim_type d3, dtype ty=f32);

    AFAPI array range(const dim4 &dims, const int seq_dim = -1, dtype ty=f32);
    AFAPI array range(const dim_type d0, const dim_type d1 = 1, const dim_type d2 = 1,
                      const dim_type d3 = 1, const int seq_dim = -1, dtype ty=f32);

    AFAPI array iota(const dim4 dims, const dim4 tile_dims = dim4(1), dtype ty=f32);

    AFAPI array diag(const array &in, const int num = 0, const bool extract = true);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    // Create af_array from a constant value
    AFAPI af_err af_constant(af_array *arr, const double val, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    AFAPI af_err af_constant_complex(af_array *arr, const double real, const double imag,
                                     const unsigned ndims, const dim_type * const dims, const af_dtype type);

    AFAPI af_err af_constant_long (af_array *arr, const  intl val, const unsigned ndims, const dim_type * const dims);

    AFAPI af_err af_constant_ulong(af_array *arr, const uintl val, const unsigned ndims, const dim_type * const dims);

    // Create sequence array
    AFAPI af_err af_range(af_array *arr, const unsigned ndims, const dim_type * const dims,
                          const int seq_dim, const af_dtype type);

    AFAPI af_err af_iota(af_array *result, const unsigned ndims, const dim_type * const dims,
                         const unsigned t_ndims, const dim_type * const tdims, const af_dtype type);

    // Generate Random Numbers using uniform distribution
    AFAPI af_err af_randu(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // Generate Random Numbers using normal distribution
    AFAPI af_err af_randn(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // Generate identity matrix
    AFAPI af_err af_identity(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // Create a diagonal matrix from input
    AFAPI af_err af_diag_create(af_array *out, const af_array in, const int num);

    // Extract a diagonal matrix from input
    AFAPI af_err af_diag_extract(af_array *out, const af_array in, const int num);

#ifdef __cplusplus
}
#endif

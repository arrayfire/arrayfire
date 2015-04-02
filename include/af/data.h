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

    /**
       \defgroup data_func_constant constant
       Create constant array from the specified dimensions
       @{

       \ingroup arrayfire_func
       \ingroup move_mat
    */
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

    /**
       @}
    */

    /**
       \defgroup data_func_randu randu
       Create a random array sampled from uniform distribution

       The data is uniformly distributed between [0, 1]

       @{

       \ingroup data_mat
       \ingroup arrayfire_func
    */

    AFAPI array randu(const dim4 &dims, dtype ty=f32);
    AFAPI array randu(const dim_type d0, dtype ty=f32);
    AFAPI array randu(const dim_type d0,
                      const dim_type d1, dtype ty=f32);
    AFAPI array randu(const dim_type d0,
                      const dim_type d1, const dim_type d2, dtype ty=f32);
    AFAPI array randu(const dim_type d0,
                      const dim_type d1, const dim_type d2,
                      const dim_type d3, dtype ty=f32);

    /**
       @}
    */

    /**
       \defgroup data_func_randn randn
       Create a random array sampled from a normal distribution

       The distribution is centered around 0

       @{

       \ingroup data_mat
       \ingroup arrayfire_func
    */
    AFAPI array randn(const dim4 &dims, dtype ty=f32);
    AFAPI array randn(const dim_type d0, dtype ty=f32);
    AFAPI array randn(const dim_type d0,
                      const dim_type d1, dtype ty=f32);
    AFAPI array randn(const dim_type d0,
                      const dim_type d1, const dim_type d2, dtype ty=f32);
    AFAPI array randn(const dim_type d0,
                      const dim_type d1, const dim_type d2,
                      const dim_type d3, dtype ty=f32);

    /**
       @}
    */

    /**
       \defgroup data_func_identity identity
       Create an identity array

       @{

       \ingroup data_mat
       \ingroup arrayfire_func
    */
    AFAPI array identity(const dim4 &dims, dtype ty=f32);
    AFAPI array identity(const dim_type d0, dtype ty=f32);
    AFAPI array identity(const dim_type d0,
                         const dim_type d1, dtype ty=f32);
    AFAPI array identity(const dim_type d0,
                         const dim_type d1, const dim_type d2, dtype ty=f32);
    AFAPI array identity(const dim_type d0,
                         const dim_type d1, const dim_type d2,
                         const dim_type d3, dtype ty=f32);
    /**
       @}
    */

    /**
       \defgroup data_func_range range
       Create an array with the specified range along a dimension

       @{

       \ingroup data_mat
       \ingroup arrayfire_func
    */
    AFAPI array range(const dim4 &dims, const int seq_dim = -1, dtype ty=f32);
    AFAPI array range(const dim_type d0, const dim_type d1 = 1, const dim_type d2 = 1,
                      const dim_type d3 = 1, const int seq_dim = -1, dtype ty=f32);

    /**
       @}
    */

    /**
       \defgroup data_func_iota iota
       Create an sequence and modify to specified dimensions

       @{

       \ingroup data_mat
       \ingroup arrafire_func
    */
    AFAPI array iota(const dim4 &dims, const dim4 &tile_dims = dim4(1), dtype ty=f32);

    /**
       @}
    */

    /**
       \defgroup data_func_diag diag
       Create a diagonal marix from input array or extract diagonal from a matrix

       @{
       \ingroup data_mat
       \ingroup arrayfire_func
    */
    AFAPI array diag(const array &in, const int num = 0, const bool extract = true);

    /**
      @}
    */
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       \defgroup data_func_constant constant
       Create constant array from the specified dimensions
       @{

       \ingroup arrayfire_func
       \ingroup data_mat
    */
    AFAPI af_err af_constant(af_array *arr, const double val, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    AFAPI af_err af_constant_complex(af_array *arr, const double real, const double imag,
                                     const unsigned ndims, const dim_type * const dims, const af_dtype type);

    AFAPI af_err af_constant_long (af_array *arr, const  intl val, const unsigned ndims, const dim_type * const dims);

    AFAPI af_err af_constant_ulong(af_array *arr, const uintl val, const unsigned ndims, const dim_type * const dims);
    /**
       @}
    */

    /**
       \defgroup data_func_range range
       Create an array with the specified range along a dimension

       @{

       \ingroup data_mat
       \ingroup arrayfire_func
    */
    AFAPI af_err af_range(af_array *arr, const unsigned ndims, const dim_type * const dims,
                          const int seq_dim, const af_dtype type);
    /**
       @}
    */

    /**
       \defgroup data_func_iota iota
       Create an sequence and modify to specified dimensions

       @{

       \ingroup data_mat
       \ingroup arrafire_func
    */
    AFAPI af_err af_iota(af_array *result, const unsigned ndims, const dim_type * const dims,
                         const unsigned t_ndims, const dim_type * const tdims, const af_dtype type);
    /**
       @}
    */

    /**
       \defgroup data_func_randu randu
       Create a random array sampled from uniform distribution

       The data is uniformly distributed between [0, 1]

       @{

       \ingroup data_mat
       \ingroup arrayfire_func
    */
    AFAPI af_err af_randu(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    /**
       @}
    */

    /**
       \defgroup data_func_randn randn
       Create a random array sampled from a normal distribution

       The distribution is centered around 0

       @{

       \ingroup data_mat
       \ingroup arrayfire_func
    */
    AFAPI af_err af_randn(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type);
    /**
       @}
    */

    /**
       \defgroup data_func_identity identity
       Create an identity array

       @{

       \ingroup data_mat
       \ingroup arrayfire_func
    */
    AFAPI af_err af_identity(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type);
    /**
       @}
    */

    /**
       \defgroup data_func_diag diag
       Create a diagonal marix from input array or extract diagonal from a matrix

       @{
       \ingroup data_mat
       \ingroup arrayfire_func
    */
    AFAPI af_err af_diag_create(af_array *out, const af_array in, const int num);

    AFAPI af_err af_diag_extract(af_array *out, const af_array in, const int num);
    /**
       @}
    */

#ifdef __cplusplus
}
#endif

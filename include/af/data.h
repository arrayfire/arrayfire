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
#include <af/dim4.hpp>

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
       \defgroup data_func_setseed setSeed
       Set the seed for the random number generator


       \param[in] seed is a 64 bit unsigned integer

       \ingroup data_mat
       \ingroup arrayfire_func
    */
    AFAPI void setSeed(const uintl seed);

    /**
       \defgroup data_func_getseed getSeed
       Get the seed for the random number generator


       \return seed which is a 64 bit unsigned integer

       \ingroup data_mat
       \ingroup arrayfire_func
    */
    AFAPI uintl getSeed();


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

    /**
       \defgroup manip_func_join join
       @{
       Join two arrays along specified dimension

       \param[in] dim is the dimension along which join occurs
       \param[in] first is the first input array
       \param[in] second is the first second array
       \return the array that joins \p first and \p second along \p dim

       \ingroup manip_mat
       \ingroup arrayfire_func
    */
    AFAPI array join(const int dim, const array &first, const array &second);

    AFAPI array join(const int dim, const array &first, const array &second, const array &third);

    AFAPI array join(const int dim, const array &first, const array &second,
                     const array &third, const array &fourth);
    /**
       @}
    */

   /**
      \defgroup manip_func_tile tile
      @{
      Tile the input array along specified dimensions

      \param[in] in is the input matrix
      \param[in] x is the number of times \p in is tiled along first dimension
      \param[in] y is the number of times \p in is tiled along second dimension
      \param[in] z is the number of times \p in is tiled along third dimension
      \param[in] w is the number of times \p in is tiled along fourth dimension
      \return the tiled output

      \ingroup manip_mat
      \ingroup arrayfire_func
   */
    AFAPI array tile(const array &in, const unsigned x, const unsigned y=1,
                     const unsigned z=1, const unsigned w=1);
    /**
       @}
    */

    /**
       \defgroup manip_func_tile tile
       @{
       Tile the input array along specified dimensions

       \param[in] in is the input matrix
       \param[in] dims dim4 of tile dimensions
       \return the tiled output

       \ingroup manip_mat
       \ingroup arrayfire_func
    */
    AFAPI array tile(const array &in, const dim4 &dims);
    /**
       @}
    */

    /**
       \defgroup manip_func_reorder reorder
       @{
       Reorder the input by in the specified order

       \param[in] in is the input matrix
       \param[in] x specifies which dimension should be first
       \param[in] y specifies which dimension should be second
       \param[in] z specifies which dimension should be third
       \param[in] w specifies which dimension should be fourth
       \return the reordered output

       \ingroup manip_mat
       \ingroup arrayfire_func
    */
    AFAPI array reorder(const array& in, const unsigned x,
                        const unsigned y=1, const unsigned z=2, const unsigned w=3);
    /**
       @}
    */

    /**
       \defgroup manip_func_shift shift
       @{
       Circular shift slong specified dimensions

       \param[in] in is the input matrix
       \param[in] x specifies the shift along first dimension
       \param[in] y specifies the shift along second dimension
       \param[in] z specifies the shift along third dimension
       \param[in] w specifies the shift along fourth dimension
       \return the shifted output

       \ingroup manip_mat
       \ingroup arrayfire_func
    */
    AFAPI array shift(const array& in, const int x, const int y=0, const int z=0, const int w=0);
    /**
       @}
    */

    /**
       \defgroup manip_func_moddims moddims
       @{

       Modify the input dimensions without changing the data order

       \ingroup manip_mat
       \ingroup arrayfire_func
    */
    AFAPI array moddims(const array& in, const unsigned ndims, const dim_type * const dims);

    AFAPI array moddims(const array& in, const dim4& dims);

    AFAPI array moddims(const array& in, dim_type d0, dim_type d1=1, dim_type d2=1, dim_type d3=1);
    /**
       @}
    */

   /**
      \defgroup manip_func_flat flat
      @{

      Flatten the input to a single dimension

      \ingroup manip_mat
      \ingroup arrayfire_func
   */
    AFAPI array flat(const array &in);
    /**
       @}
    */

   /**
    \defgroup manip_func_flip flip
    @{

    Flip the input along sepcified dimension

    \ingroup manip_mat
    \ingroup arrayfire_func
   */
    AFAPI array flip(const array &in, const unsigned dim);
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
       \defgroup data_func_setseed setSeed
       Set the seed for the random number generator


       \param[in] seed is a 64 bit unsigned integer

       \ingroup data_mat
       \ingroup arrayfire_func
    */
    AFAPI af_err af_set_seed(const uintl seed);

    /**
       \defgroup data_func_getseed getSeed
       Get the seed for the random number generator


       \param[out] seed which is a 64 bit unsigned integer

       \ingroup data_mat
       \ingroup arrayfire_func
    */
    AFAPI af_err af_get_seed(uintl *seed);


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

    /**
       \ingroup manip_func_join
    */
    AFAPI af_err af_join(af_array *out, const int dim, const af_array first, const af_array second);

    AFAPI af_err af_join_many(af_array *out, const int dim, const unsigned n_arrays, const af_array *inputs);

    /**
       \ingroup manip_func_tile
    */
    AFAPI af_err af_tile(af_array *out, const af_array in,
                         const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    /**
       \ingroup manip_func_reorder
    */
    AFAPI af_err af_reorder(af_array *out, const af_array in,
                            const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    /**
       \ingroup manip_func_shift
    */
    AFAPI af_err af_shift(af_array *out, const af_array in, const int x, const int y, const int z, const int w);

    /**
       \ingroup manip_func_moddims
    */
    AFAPI af_err af_moddims(af_array *out, const af_array in, const unsigned ndims, const dim_type * const dims);

    /**
       \ingroup manip_func_flat
    */
    AFAPI af_err af_flat(af_array *out, const af_array in);


#ifdef __cplusplus
}
#endif

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
#include <af/dim4.hpp>
#include <af/traits.hpp>
namespace af
{
    class array;

    /// C++ Interface to generate an array with elements set to a specified
    /// value.
    ///
    /// \param[in] val  constant value
    /// \param[in] dims dimensions of the array to be generated
    /// \param[in] ty   type
    /// \return         constant array
    ///
    /// \ingroup data_func_constant
    template<typename T>
    array constant(T val, const dim4 &dims, const dtype ty=(af_dtype)dtype_traits<T>::ctype);

    /// C++ Interface to generate a 1-D array with elements set to a specified
    /// value.
    ///
    /// \param[in] val constant value
    /// \param[in] d0  size of the first dimension
    /// \param[in] ty  type
    /// \return        constant 1-D array
    ///
    /// \ingroup data_func_constant
    template<typename T>
    array constant(T val, const dim_t d0, const af_dtype ty=(af_dtype)dtype_traits<T>::ctype);

    /// C++ Interface to generate a 2-D array with elements set to a specified
    /// value.
    ///
    /// \param[in] val constant value
    /// \param[in] d0  size of the first dimension
    /// \param[in] d1  size of the second dimension
    /// \param[in] ty  type
    /// \return        constant 2-D array
    ///
    /// \ingroup data_func_constant
    template<typename T>
    array constant(T val, const dim_t d0, const dim_t d1, const af_dtype ty=(af_dtype)dtype_traits<T>::ctype);

    /// C++ Interface to generate a 3-D array with elements set to a specified
    /// value.
    ///
    /// \param[in] val constant value
    /// \param[in] d0  size of the first dimension
    /// \param[in] d1  size of the second dimension
    /// \param[in] d2  size of the third dimension
    /// \param[in] ty  type
    /// \return        constant 3-D array
    ///
    /// \ingroup data_func_constant
    template<typename T>
    array constant(T val, const dim_t d0, const dim_t d1, const dim_t d2, const af_dtype ty=(af_dtype)dtype_traits<T>::ctype);

    /// C++ Interface to generate a 4-D array with elements set to a specified
    /// value.
    ///
    /// \param[in] val constant value
    /// \param[in] d0  size of the first dimension
    /// \param[in] d1  size of the second dimension
    /// \param[in] d2  size of the third dimension
    /// \param[in] d3  size of the fourth dimension
    /// \param[in] ty  type
    /// \return        constant 4-D array
    ///
    /// \ingroup data_func_constant
    template<typename T>
    array constant(T val, const dim_t d0, const dim_t d1, const dim_t d2, const dim_t d3, const af_dtype ty=(af_dtype)dtype_traits<T>::ctype);

    /// C++ Interface to generate an identity array.
    ///
    /// \param[in] dims size
    /// \param[in] ty   type
    /// \return         identity array
    ///
    /// \ingroup data_func_identity
    AFAPI array identity(const dim4 &dims, const dtype ty=f32);

    /// C++ Interface to generate a 1-D identity array.
    ///
    /// \param[in] d0 size of the first dimension
    /// \param[in] ty type
    /// \return       identity array
    ///
    /// \ingroup data_func_identity
    AFAPI array identity(const dim_t d0, const dtype ty=f32);

    /// C++ Interface to generate a 2-D identity array.
    ///
    /// \param[in] d0 size of the first dimension
    /// \param[in] d1 size of the second dimension
    /// \param[in] ty type
    /// \return       identity array
    ///
    /// \ingroup data_func_identity
    AFAPI array identity(const dim_t d0, const dim_t d1, const dtype ty=f32);

    /// C++ Interface to generate a 3-D identity array.
    ///
    /// \param[in] d0 size of the first dimension
    /// \param[in] d1 size of the second dimension
    /// \param[in] d2 size of the third dimension
    /// \param[in] ty type
    /// \return       identity array
    ///
    /// \ingroup data_func_identity
    AFAPI array identity(const dim_t d0, const dim_t d1,
                         const dim_t d2, const dtype ty=f32);

    /// C++ Interface to generate a 4-D identity array.
    ///
    /// \param[in] d0 size of the first dimension
    /// \param[in] d1 size of the second dimension
    /// \param[in] d2 size of the third dimension
    /// \param[in] d3 size of the fourth dimension
    /// \param[in] ty type
    /// \return       identity array
    ///
    /// \ingroup data_func_identity
    AFAPI array identity(const dim_t d0, const dim_t d1,
                         const dim_t d2, const dim_t d3, const dtype ty=f32);

    /// C++ Interface to generate an array with `[0, n-1]` values along the
    /// `seq_dim` dimension and tiled across other dimensions of shape `dim4`.
    ///
    /// \param[in] dims    size
    /// \param[in] seq_dim dimesion along which the range is created
    /// \param[in] ty      type
    /// \return            range array
    ///
    /// \ingroup data_func_range
    AFAPI array range(const dim4 &dims, const int seq_dim = -1, const dtype ty=f32);

    /// C++ Interface to generate an array with `[0, n-1]` values along the
    /// `seq_dim` dimension and tiled across other dimensions described by
    /// dimension parameters.
    ///
    /// \param[in] d0      size of the first dimension
    /// \param[in] d1      size of the second dimension
    /// \param[in] d2      size of the third dimension
    /// \param[in] d3      size of the fourth dimension
    /// \param[in] seq_dim dimesion along which the range is created
    /// \param[in] ty      type
    /// \return            range array
    ///
    /// \ingroup data_func_range
    AFAPI array range(const dim_t d0, const dim_t d1 = 1, const dim_t d2 = 1,
                      const dim_t d3 = 1, const int seq_dim = -1, const dtype ty=f32);

    /// C++ Interface to generate an array with `[0, n-1]` values modified to
    /// specified dimensions and tiling.
    ///
    /// \param[in] dims      size
    /// \param[in] tile_dims number of tiled repetitions in each dimension
    /// \param[in] ty        type
    /// \return              iota array
    ///
    /// \ingroup data_func_iota
    AFAPI array iota(const dim4 &dims, const dim4 &tile_dims = dim4(1), const dtype ty=f32);

    /// C++ Interface to extract the diagonal from an array.
    ///
    /// \param[in] in      input array
    /// \param[in] num     diagonal index
    /// \param[in] extract if true, returns an array containing diagonal of the
    ///                    matrix; if false, returns a diagonal matrix
    /// \return            diagonal array (or matrix)
    ///
    /// \ingroup data_func_diag
    AFAPI array diag(const array &in, const int num = 0, const bool extract = true);

    /// C++ Interface to join 2 arrays along a dimension.
    ///
    /// Empty arrays are ignored.
    ///
    /// \param[in] dim    dimension along which the join occurs
    /// \param[in] first  input array
    /// \param[in] second input array
    /// \return           joined array
    ///
    /// \ingroup manip_func_join
    AFAPI array join(const int dim, const array &first, const array &second);

    /// C++ Interface to join 3 arrays along a dimension.
    ///
    /// Empty arrays are ignored.
    ///
    /// \param[in] dim    dimension along which the join occurs
    /// \param[in] first  input array
    /// \param[in] second input array
    /// \param[in] third  input array
    /// \return           joined array
    ///
    /// \ingroup manip_func_join
    AFAPI array join(const int dim, const array &first, const array &second, const array &third);

    /// C++ Interface to join 4 arrays along a dimension.
    ///
    /// Empty arrays are ignored.
    ///
    /// \param[in] dim    dimension along which the join occurs
    /// \param[in] first  input array
    /// \param[in] second input array
    /// \param[in] third  input array
    /// \param[in] fourth input array
    /// \return           joined array
    ///
    /// \ingroup manip_func_join
    AFAPI array join(const int dim, const array &first, const array &second,
                     const array &third, const array &fourth);

    /// C++ Interface to generate a tiled array.
    ///
    /// Note, `x`, `y`, `z`, and `w` include the original in the count.
    ///
    /// \param[in] in input array
    /// \param[in] x  number tiles along the first dimension
    /// \param[in] y  number tiles along the second dimension
    /// \param[in] z  number tiles along the third dimension
    /// \param[in] w  number tiles along the fourth dimension
    /// \return       tiled array
    ///
    /// \ingroup manip_func_tile
    AFAPI array tile(const array &in, const unsigned x, const unsigned y=1,
                     const unsigned z=1, const unsigned w=1);

    /// C++ Interface to generate a tiled array.
    ///
    /// Each component of `dims` includes the original in the count. Thus, if
    /// no duplicates are needed in a certain dimension, it is left as 1, the
    /// default value for just one copy.
    ///
    /// \param[in] in   input array
    /// \param[in] dims number of times `in` is copied along each dimension
    /// \return         tiled array
    ///
    /// \ingroup manip_func_tile
    AFAPI array tile(const array &in, const dim4 &dims);

    /// C++ Interface to reorder an array. 
    ///
    /// \param[in] in input array
    /// \param[in] x  specifies which dimension should be first
    /// \param[in] y  specifies which dimension should be second
    /// \param[in] z  specifies which dimension should be third
    /// \param[in] w  specifies which dimension should be fourth
    /// \return       reordered array
    ///
    /// \ingroup manip_func_reorder
    AFAPI array reorder(const array& in, const unsigned x,
                        const unsigned y=1, const unsigned z=2, const unsigned w=3);

    /// C++ Interface to shift an array.
    ///
    /// \param[in] in input array
    /// \param[in] x  specifies the shift along the first dimension
    /// \param[in] y  specifies the shift along the second dimension
    /// \param[in] z  specifies the shift along the third dimension
    /// \param[in] w  specifies the shift along the fourth dimension
    /// \return       shifted array
    ///
    /// \ingroup manip_func_shift
    AFAPI array shift(const array& in, const int x, const int y=0, const int z=0, const int w=0);

    /// C++ Interface to modify the dimensions of an input array to a specified
    /// shape.
    ///
    /// \param[in] in   input array
    /// \param[in] dims new dimension sizes
    /// \return         modded output
    ///
    /// \ingroup manip_func_moddims
    AFAPI array moddims(const array& in, const dim4& dims);

    /// C++ Interface to modify the dimensions of an input array to a specified
    /// shape.
    ///
    /// \param[in] in input array
    /// \param[in] d0 new size of the first dimension
    /// \param[in] d1 new size of the second dimension (optional)
    /// \param[in] d2 new size of the third dimension (optional)
    /// \param[in] d3 new size of the fourth dimension (optional)
    /// \return       modded output
    ///
    /// \ingroup manip_func_moddims
    AFAPI array moddims(const array& in, const dim_t d0, const dim_t d1=1, const dim_t d2=1, const dim_t d3=1);

    /// C++ Interface to modify the dimensions of an input array to a specified
    /// shape.
    ///
    /// \param[in] in    input array
    /// \param[in] ndims number of dimensions
    /// \param[in] dims  new dimension sizes
    /// \return          modded output
    ///
    /// \ingroup manip_func_moddims
    AFAPI array moddims(const array& in, const unsigned ndims, const dim_t* const dims);

    /// C++ Interface to flatten an array.
    ///
    /// \param[in] in input array
    /// \return       flat array
    ///
    /// \ingroup manip_func_flat
    AFAPI array flat(const array &in);

    /// C++ Interface to flip an array.
    ///
    /// \param[in] in  input array
    /// \param[in] dim dimension to flip
    /// \return        flipped array
    ///
    /// \ingroup manip_func_flip
    AFAPI array flip(const array &in, const unsigned dim);

    /// C++ Interface to return the lower triangle array.
    ///
    /// \param[in] in           input array
    /// \param[in] is_unit_diag boolean specifying if diagonal elements are 1's
    /// \return                 lower triangle array
    ///
    /// \ingroup data_func_lower
    AFAPI array lower(const array &in, bool is_unit_diag=false);

    /// C++ Interface to return the upper triangle array.
    ///
    /// \param[in] in           input array
    /// \param[in] is_unit_diag boolean specifying if diagonal elements are 1's
    /// \return                 upper triangle matrix
    ///
    /// \ingroup data_func_upper
    AFAPI array upper(const array &in, bool is_unit_diag=false);

#if AF_API_VERSION >= 31
    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select array element
    /// \param[in] b    when false, select array element
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    AFAPI array select(const array &cond, const array  &a, const array  &b);
#endif

#if AF_API_VERSION >= 31
    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select array element
    /// \param[in] b    when false, select scalar value
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    AFAPI array select(const array &cond, const array  &a, const double &b);
#endif

#if AF_API_VERSION >= 31
    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select scalar value
    /// \param[in] b    when false, select array element
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    AFAPI array select(const array &cond, const double &a, const array  &b);
#endif

#if AF_API_VERSION >= 31
    /// C++ Interface to replace elements of an array with elements of another
    /// array.
    ///
    /// Elements of `a` are replaced with corresponding elements of `b` when
    /// `cond` is false.
    ///
    /// \param[inout] a    input array
    /// \param[in]    cond conditional array
    /// \param[in]    b    replacement array
    ///
    /// \ingroup data_func_replace
    AFAPI void replace(array &a, const array  &cond, const array  &b);
#endif

#if AF_API_VERSION >= 31
    /// C++ Interface to replace elements of an array with a scalar value.
    ///
    /// Elements of `a` are replaced with a scalar value when `cond` is false.
    ///
    /// \param[inout] a    input array
    /// \param[in]    cond conditional array
    /// \param[in]    b    replacement scalar value
    ///
    /// \ingroup data_func_replace
    AFAPI void replace(array &a, const array  &cond, const double &b);
#endif

#if AF_API_VERSION >= 37
    /// C++ Interface to pad an array.
    ///
    /// \param[in] in           input array
    /// \param[in] beginPadding number of elements to be padded at the start of
    ///                         each dimension
    /// \param[in] endPadding   number of elements to be padded at the end of
    ///                         each dimension
    /// \param[in] padFillType  values to fill into the padded region
    /// \return                 padded array
    ///
    /// \ingroup data_func_pad
    AFAPI array pad(const array &in, const dim4 &beginPadding,
                    const dim4 &endPadding, const borderType padFillType);
#endif

#if AF_API_VERSION >= 39
    /// C++ Interface to replace elements of an array with a scalar value.
    ///
    /// Elements of `a` are replaced with a scalar value when `cond` is false.
    ///
    /// \param[inout] a    input array
    /// \param[in]    cond conditional array
    /// \param[in]    b    replacement scalar value
    ///
    /// \ingroup data_func_replace
    AFAPI void replace(array &a, const array &cond, const long long b);

    /// C++ Interface to replace elements of an array with a scalar value.
    ///
    /// Elements of `a` are replaced with a scalar value when `cond` is false.
    ///
    /// \param[inout] a    input array
    /// \param[in]    cond conditional array
    /// \param[in]    b    replacement scalar value
    ///
    /// \ingroup data_func_replace
    AFAPI void replace(array &a, const array &cond,
                       const unsigned long long b);

    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select array element
    /// \param[in] b    when false, select scalar value
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    AFAPI array select(const array &cond, const array &a, const long long b);

    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select array element
    /// \param[in] b    when false, select scalar value
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    AFAPI array select(const array &cond, const array &a,
                       const unsigned long long b);

    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select scalar value
    /// \param[in] b    when false, select array element
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    AFAPI array select(const array &cond, const long long a, const array &b);

    /// C++ Interface to select elements based on a conditional array.
    ///
    /// \param[in] cond conditional array
    /// \param[in] a    when true, select scalar value
    /// \param[in] b    when false, select array element
    /// \return         `a` when `cond` is true, else `b`
    ///
    /// \ingroup data_func_select
    AFAPI array select(const array &cond, const unsigned long long a,
                       const array &b);
#endif
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    /**
       C Interface to generate an array with elements set to a specified value.

       \param[out] arr   constant array
       \param[in]  val   constant value
       \param[in]  ndims size of the dimension array
       \param[in]  dims  dimensions of the array to be generated
       \param[in]  type  type
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_constant
    */
    AFAPI af_err af_constant(af_array *arr, const double val, const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
       C Interface to generate a complex array with elements set to a specified
       value.

       \param[out] arr   constant complex array
       \param[in]  real  real constant value
       \param[in]  imag  imaginary constant value
       \param[in]  ndims size of the dimension array
       \param[in]  dims  dimensions of the array to be generated
       \param[in]  type  type, \ref c32 or \ref c64
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_constant
    */
    AFAPI af_err af_constant_complex(af_array *arr, const double real, const double imag,
                                     const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
       C Interface to generate an array with elements set to a specified value.

       Output type is \ref s64.

       \param[out] arr   constant array
       \param[in]  val   constant value
       \param[in]  ndims size of the dimension array
       \param[in]  dims  dimensions of the array to be generated
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_constant
    */
    AFAPI af_err af_constant_long (af_array *arr, const long long val, const unsigned ndims, const dim_t * const dims);

    /**
       C Interface to generate an array with elements set to a specified value.

       Output type is \ref u64.

       \param[out] arr   constant array
       \param[in]  val   constant value
       \param[in]  ndims size of the dimension array
       \param[in]  dims  dimensions of the array to be generated
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_constant
    */

    AFAPI af_err af_constant_ulong(af_array *arr, const unsigned long long val, const unsigned ndims, const dim_t * const dims);

    /**
       C Interface to generate an identity array.

       \param[out] out   identity array
       \param[in]  ndims number of dimensions
       \param[in]  dims  size
       \param[in]  type  type
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_identity
    */
    AFAPI af_err af_identity(af_array* out, const unsigned ndims, const dim_t* const dims, const af_dtype type);

    /**
       C Interface to generate an array with `[0, n-1]` values along the
       `seq_dim` dimension and tiled across other dimensions of shape `dim4`.

       \param[out] out     range array
       \param[in]  ndims   number of dimensions, specified by the size of `dims`
       \param[in]  dims    size
       \param[in]  seq_dim dimension along which the range is created
       \param[in]  type    type
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_range
    */
    AFAPI af_err af_range(af_array *out, const unsigned ndims, const dim_t * const dims,
                          const int seq_dim, const af_dtype type);

    /**
       C Interface to generate an array with `[0, n-1]` values modified to
       specified dimensions and tiling.

       \param[out] out     iota array
       \param[in]  ndims   number of dimensions
       \param[in]  dims    size
       \param[in]  t_ndims number of dimensions of tiled array
       \param[in]  tdims   number of tiled repetitions in each dimension
       \param[in]  type    type
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_iota
    */
    AFAPI af_err af_iota(af_array *out, const unsigned ndims, const dim_t * const dims,
                         const unsigned t_ndims, const dim_t * const tdims, const af_dtype type);

    /**
       C Interface to create a diagonal matrix from an extracted diagonal
       array.

       See also, \ref af_diag_extract.

       \param[out] out diagonal matrix
       \param[in]  in  diagonal array
       \param[in]  num diagonal index
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_diag
    */
    AFAPI af_err af_diag_create(af_array *out, const af_array in, const int num);

    /**
       C Interface to extract the diagonal from an array.

       See also, \ref af_diag_create.

       \param[out] out `num`-th diagonal array
       \param[in]  in  input array
       \param[in]  num diagonal index
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_diag
    */
    AFAPI af_err af_diag_extract(af_array *out, const af_array in, const int num);

    /**
       C Interface to join 2 arrays along a dimension.

       Empty arrays are ignored.

       \param[out] out    joined array
       \param[in]  dim    dimension along which the join occurs
       \param[in]  first  input array
       \param[in]  second input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup manip_func_join
    */
    AFAPI af_err af_join(af_array *out, const int dim, const af_array first, const af_array second);

    /**
       C Interface to join many arrays along a dimension.

       Limited to 10 arrays. Empty arrays are ignored.

       \param[out] out      joined array
       \param[in]  dim      dimension along which the join occurs
       \param[in]  n_arrays number of arrays to join
       \param[in]  inputs   array of af_arrays containing handles to the
                             arrays to be joined
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup manip_func_join
    */
    AFAPI af_err af_join_many(af_array *out, const int dim, const unsigned n_arrays, const af_array *inputs);

    /**
       C Interface to generate a tiled array.

       Note, `x`, `y`, `z`, and `w` include the original in the count.

       \param[out] out tiled array
       \param[in]  in  input array
       \param[in]  x   number of tiles along the first dimension
       \param[in]  y   number of tiles along the second dimension
       \param[in]  z   number of tiles along the third dimension
       \param[in]  w   number of tiles along the fourth dimension
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup manip_func_tile
    */
    AFAPI af_err af_tile(af_array *out, const af_array in,
                         const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    /**
       C Interface to reorder an array.

       \param[out] out reordered array
       \param[in]  in  input array
       \param[in]  x   specifies which dimension should be first
       \param[in]  y   specifies which dimension should be second
       \param[in]  z   specifies which dimension should be third
       \param[in]  w   specifies which dimension should be fourth
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup manip_func_reorder
    */
    AFAPI af_err af_reorder(af_array *out, const af_array in,
                            const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    /**
       C Interface to shift an array.

       \param[out] out shifted array
       \param[in]  in  input array
       \param[in]  x   specifies the shift along first dimension
       \param[in]  y   specifies the shift along second dimension
       \param[in]  z   specifies the shift along third dimension
       \param[in]  w   specifies the shift along fourth dimension
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup manip_func_shift
    */
    AFAPI af_err af_shift(af_array *out, const af_array in, const int x, const int y, const int z, const int w);

    /**
       C Interface to modify the dimensions of an input array to a specified
       shape.

       \param[out] out   modded output
       \param[in]  in    input array
       \param[in]  ndims number of dimensions
       \param[in]  dims  new dimension sizes
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup manip_func_moddims
    */
    AFAPI af_err af_moddims(af_array *out, const af_array in, const unsigned ndims, const dim_t * const dims);

    /**
       C Interface to flatten an array.

       \param[out] out flat array
       \param[in]  in  input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup manip_func_flat
    */
    AFAPI af_err af_flat(af_array *out, const af_array in);

    /**
       C Interface to flip an array.

       \param[out] out flipped array
       \param[in]  in  input array
       \param[in]  dim dimension to flip
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup manip_func_flip
    */
    AFAPI af_err af_flip(af_array *out, const af_array in, const unsigned dim);

    /**
       C Interface to return the lower triangle array.

       \param[out] out          lower traingle array
       \param[in]  in           input array
       \param[in]  is_unit_diag boolean specifying if diagonal elements are 1's
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_lower
    */
    AFAPI af_err af_lower(af_array *out, const af_array in, bool is_unit_diag);

    /**
       C Interface to return the upper triangle array.

       \param[out] out          upper triangle array
       \param[in]  in           input array
       \param[in]  is_unit_diag boolean specifying if diagonal elements are 1's
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_upper
    */
    AFAPI af_err af_upper(af_array *out, const af_array in, bool is_unit_diag);

#if AF_API_VERSION >= 31
    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select array element
       \param[in]  b    when false, select array element
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_select
    */
    AFAPI af_err af_select(af_array *out, const af_array cond, const af_array a, const af_array b);
#endif

#if AF_API_VERSION >= 31
    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select array element
       \param[in]  b    when false, select scalar value
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_select
    */
    AFAPI af_err af_select_scalar_r(af_array *out, const af_array cond, const af_array a, const double b);
#endif

#if AF_API_VERSION >= 31
    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select scalar value
       \param[in]  b    when false, select array element
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_select
    */
    AFAPI af_err af_select_scalar_l(af_array *out, const af_array cond, const double a, const af_array b);
#endif

#if AF_API_VERSION >= 31
    /**
       C Interface to replace elements of an array with elements of another
       array.

       Elements of `a` are replaced with corresponding elements of `b` when
       `cond` is false.

       \param[inout]  a    input array
       \param[in]     cond conditional array
       \param[in]     b    replacement array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_replace
    */
    AFAPI af_err af_replace(af_array a, const af_array cond, const af_array b);
#endif

#if AF_API_VERSION >= 31
    /**
       C Interface to replace elements of an array with a scalar value.

       Elements of `a` are replaced with a scalar value when `cond` is false.

       \param[inout] a    input array
       \param[in]    cond conditional array
       \param[in]    b    replacement scalar value
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_replace
    */
    AFAPI af_err af_replace_scalar(af_array a, const af_array cond, const double b);
#endif

#if AF_API_VERSION >= 37
    /**
       C Interface to pad an array.

       \param[out] out           padded array
       \param[in]  in            input array
       \param[in]  begin_ndims   number of dimensions for start padding
       \param[in]  begin_dims    number of elements to be padded at the start
                                 of each dimension
       \param[in]  end_ndims     number of dimensions for end padding
       \param[in]  end_dims      number of elements to be padded at the end of
                                 each dimension
       \param[in]  pad_fill_type values to fill into the padded region
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_pad
    */
    AFAPI af_err af_pad(af_array *out, const af_array in,
                        const unsigned begin_ndims,
                        const dim_t *const begin_dims, const unsigned end_ndims,
                        const dim_t *const end_dims,
                        const af_border_type pad_fill_type);
#endif

#if AF_API_VERSION >= 39
    /**
       C Interface to replace elements of an array with a scalar value.

       Elements of `a` are replaced with a scalar value when `cond` is false.

       \param[inout] a    input array
       \param[in]    cond conditional array
       \param[in]    b    replacement scalar value
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_replace
    */
    AFAPI af_err af_replace_scalar_long(af_array a, const af_array cond,
                                        const long long b);

    /**
       C Interface to replace elements of an array with a scalar value.

       Elements of `a` are replaced with a scalar value when `cond` is false.

       \param[inout] a    input array
       \param[in]    cond conditional array
       \param[in]    b    replacement scalar value
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_replace
    */
    AFAPI af_err af_replace_scalar_ulong(af_array a, const af_array cond,
                                         const unsigned long long b);

    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select array element
       \param[in]  b    when false, select scalar value
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_select
    */
    AFAPI af_err af_select_scalar_r_long(af_array *out, const af_array cond,
                                         const af_array a, const long long b);

    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select array element
       \param[in]  b    when false, select scalar value
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_select
    */
    AFAPI af_err af_select_scalar_r_ulong(af_array *out, const af_array cond,
                                          const af_array a,
                                          const unsigned long long b);

    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select scalar value
       \param[in]  b    when false, select array element
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_select
    */
    AFAPI af_err af_select_scalar_l_long(af_array *out, const af_array cond,
                                         const long long a, const af_array b);

    /**
       C Interface to select elements based on a conditional array.

       \param[out] out  `a` when `cond` is true, else `b`
       \param[in]  cond conditional array
       \param[in]  a    when true, select scalar value
       \param[in]  b    when false, select array element
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup data_func_select
    */
    AFAPI af_err af_select_scalar_l_ulong(af_array *out, const af_array cond,
                                          const unsigned long long a,
                                          const af_array b);
#endif

#ifdef __cplusplus
}
#endif

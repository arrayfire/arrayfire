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

    /**
        \param[in] val is the value of each element of the array be genrated
        \param[in] dims is the dimensions of the array to be generated
        \param[in] ty is the type of the array

        \return array of size \p dims

        \ingroup data_func_constant
    */

    template<typename T>
    array constant(T val, const dim4 &dims, const dtype ty=(af_dtype)dtype_traits<T>::ctype);

    /**
        \param[in] val is the value of each element of the array to be generated
        \param[in] d0 is the size of the array to be generated
        \param[in] ty is the type of the array

        \return array of size \p d0

        \ingroup data_func_constant
    */

    template<typename T>
    array constant(T val, const dim_t d0, const af_dtype ty=(af_dtype)dtype_traits<T>::ctype);

    /**
        \param[in] val is the value of each element of the array to be generated
        \param[in] d0 is the number of rows of the array to be generated
        \param[in] d1 is the number of columns of the array to be generated
        \param[in] ty is the type of the array

        \return array of size \p d0 x d1

        \ingroup data_func_constant
    */
    template<typename T>
    array constant(T val, const dim_t d0, const dim_t d1, const af_dtype ty=(af_dtype)dtype_traits<T>::ctype);

    /**
        \param[in] val is the value of each element of the array to be generated
        \param[in] d0 is the size of the 1st dimension of the array to be generated
        \param[in] d1 is the size of the 2nd dimension of the array to be generated
        \param[in] d2 is the size of the 3rd dimension of the array to be generated
        \param[in] ty is the type of the array

        \return array of size \p d0 x d1 x d2

        \ingroup data_func_constant
    */
    template<typename T>
    array constant(T val, const dim_t d0, const dim_t d1, const dim_t d2, const af_dtype ty=(af_dtype)dtype_traits<T>::ctype);

    /**
        \param[in] val is the value of each element of the array to be generated
        \param[in] d0 is the size of the 1st dimension of the array to be generated
        \param[in] d1 is the size of the 2nd dimension of the array to be generated
        \param[in] d2 is the size of the 3rd dimension of the array to be generated
        \param[in] d3 is the size of the 4rd dimension of the array to be generated
        \param[in] ty is the type of the array

        \return array of size \p d0 x d1 x d2 x d3

        \ingroup data_func_constant
    */
    template<typename T>
    array constant(T val, const dim_t d0, const dim_t d1, const dim_t d2, const dim_t d3, const af_dtype ty=(af_dtype)dtype_traits<T>::ctype);

    /**
        \param[in] dims is dim4 for size of all dimensions
        \param[in] ty is the type of array to generate

        \returns an identity array of specified dimension and type

        \ingroup data_func_identity
    */
    AFAPI array identity(const dim4 &dims, const dtype ty=f32);

    /**
        \param[in] d0 is size of first dimension
        \param[in] ty is the type of array to generate

        \returns an identity array of specified dimension and type

        \ingroup data_func_identity
    */
    AFAPI array identity(const dim_t d0, const dtype ty=f32);

    /**
        \param[in] d0 is size of first dimension
        \param[in] d1 is size of second dimension
        \param[in] ty is the type of array to generate

        \returns an identity array of specified dimension and type

        \ingroup data_func_identity
    */
    AFAPI array identity(const dim_t d0, const dim_t d1, const dtype ty=f32);

    /**
        \param[in] d0 is size of first dimension
        \param[in] d1 is size of second dimension
        \param[in] d2 is size of third dimension
        \param[in] ty is the type of array to generate

        \returns an identity array of specified dimension and type

        \ingroup data_func_identity
    */
    AFAPI array identity(const dim_t d0, const dim_t d1,
                         const dim_t d2, const dtype ty=f32);

    /**
        \param[in] d0 is size of first dimension
        \param[in] d1 is size of second dimension
        \param[in] d2 is size of third dimension
        \param[in] d3 is size of fourth dimension
        \param[in] ty is the type of array to generate

        \returns an identity array of specified dimension and type

        \ingroup data_func_identity
    */
    AFAPI array identity(const dim_t d0, const dim_t d1,
                         const dim_t d2, const dim_t d3, const dtype ty=f32);

    /**
    *  C++ Interface for creating an array with `[0, n-1]` values along the `seq_dim` dimension and tiled across other dimensions of shape `dim4`.
    *
        \param[in] dims the `dim4` object describing the shape of the generated array
        \param[in] seq_dim the dimesion along which `[0, dim[seq_dim] - 1]` is created
        \param[in] ty the type of the generated array

        \returns the generated array

        \ingroup data_func_range
    */
    AFAPI array range(const dim4 &dims, const int seq_dim = -1, const dtype ty=f32);

    /**
    *  C++ Interface for creating an array with `[0, n-1]` values along the `seq_dim` dimension and tiled across other dimensions described by dimension parameters.
    *
        \param[in] d0 the size of first dimension
        \param[in] d1 the size of second dimension
        \param[in] d2 the size of third dimension
        \param[in] d3 the size of fourth dimension
        \param[in] seq_dim the dimesion along which `[0, dim[seq_dim] - 1]` is created
        \param[in] ty the type of the generated array

        \returns the generated array

        \ingroup data_func_range
    */
    AFAPI array range(const dim_t d0, const dim_t d1 = 1, const dim_t d2 = 1,
                      const dim_t d3 = 1, const int seq_dim = -1, const dtype ty=f32);

    /**
        \param[in] dims is dim4 for unit dimensions of the sequence to be generated
        \param[in] tile_dims is dim4 for the number of repetitions of the unit dimensions
        \param[in] ty is the type of array to generate

        \returns an array of integral range specified dimension and type

        \ingroup data_func_iota
    */
    AFAPI array iota(const dim4 &dims, const dim4 &tile_dims = dim4(1), const dtype ty=f32);

    /**
        \param[in] in is the input array
        \param[in] num is the diagonal index
        \param[in] extract when true returns an array containing diagonal of tha matrix
        and when false returns a matrix with \p in as diagonal

        \returns an array with either the diagonal or the matrix based on \p extract

        \ingroup data_func_diag
    */
    AFAPI array diag(const array &in, const int num = 0, const bool extract = true);

    /**
        \brief Join 2 arrays along \p dim

        \param[in] dim is the dimension along which join occurs
        \param[in] first is the first input array
        \param[in] second is the second input array
        \return the array that joins input arrays along the given dimension

        \note empty arrays will be ignored

        \ingroup manip_func_join
    */
    AFAPI array join(const int dim, const array &first, const array &second);

    /**
        \brief Join 3 arrays along \p dim

        \param[in] dim is the dimension along which join occurs
        \param[in] first is the first input array
        \param[in] second is the second input array
        \param[in] third is the third input array
        \return the array that joins input arrays along the given dimension

        \note empty arrays will be ignored

        \ingroup manip_func_join
    */
    AFAPI array join(const int dim, const array &first, const array &second, const array &third);

    /**
        \brief Join 4 arrays along \p dim

        \param[in] dim is the dimension along which join occurs
        \param[in] first is the first input array
        \param[in] second is the second input array
        \param[in] third is the third input array
        \param[in] fourth is the fourth input array
        \return the array that joins input arrays along the given dimension

        \note empty arrays will be ignored

        \ingroup manip_func_join
    */
    AFAPI array join(const int dim, const array &first, const array &second,
                     const array &third, const array &fourth);

    /**
        \param[in] in is the input array
        \param[in] x is the number of times \p in is copied along the first dimension
        \param[in] y is the number of times \p in is copied along the the second dimension
        \param[in] z is the number of times \p in is copied along the third dimension
        \param[in] w is the number of times \p in is copied along the fourth dimension
        \return The tiled version of the input array

        \note \p x, \p y, \p z, and \p w includes the original in the count as
              well. Thus, if no duplicates are needed in a certain dimension,
              leave it as 1 (the default value for just one copy)

        \ingroup manip_func_tile
    */
    AFAPI array tile(const array &in, const unsigned x, const unsigned y=1,
                     const unsigned z=1, const unsigned w=1);

    /**
        \param[in] in is the input array
        \param[in] dims specifies the number of times \p in is copied along each dimension
        \return The tiled version of the input array

        \note Each component of \p dims includes the original in the count as
              well. Thus, if no duplicates are needed in a certain dimension,
              leave it as 1 (the default value for just one copy)

        \ingroup manip_func_tile
    */
    AFAPI array tile(const array &in, const dim4 &dims);

    /**
        \param[in] in is the input array
        \param[in] x specifies which dimension should be first
        \param[in] y specifies which dimension should be second
        \param[in] z specifies which dimension should be third
        \param[in] w specifies which dimension should be fourth
        \return the reordered output

        \ingroup manip_func_reorder
    */
    AFAPI array reorder(const array& in, const unsigned x,
                        const unsigned y=1, const unsigned z=2, const unsigned w=3);

    /**
        \param[in] in is the input array
        \param[in] x specifies the shift along first dimension
        \param[in] y specifies the shift along second dimension
        \param[in] z specifies the shift along third dimension
        \param[in] w specifies the shift along fourth dimension

        \return the shifted output

        \ingroup manip_func_shift
    */
    AFAPI array shift(const array& in, const int x, const int y=0, const int z=0, const int w=0);

    /**
    * C++ Interface for modifying the dimensions of an input array to the shape specified by a `dim4` object
    *
        \param[in] in the input array
        \param[in] dims the array of new dimension sizes
        \return the modded output

        \ingroup manip_func_moddims
    */
    AFAPI array moddims(const array& in, const dim4& dims);

    /**
    * C++ Interface for modifying the dimensions of an input array to the shape specified by dimension length parameters
    *
        \param[in] in the input array
        \param[in] d0 the new size of the first dimension
        \param[in] d1 the new size of the second dimension (optional)
        \param[in] d2 the new size of the third dimension (optional)
        \param[in] d3 the new size of the fourth dimension (optional)
        \return the modded output

        \ingroup manip_func_moddims
    */
    AFAPI array moddims(const array& in, const dim_t d0, const dim_t d1=1, const dim_t d2=1, const dim_t d3=1);

    /**
    * C++ Interface for modifying the dimensions of an input array to the shape specified by an array of `ndims` dimensions
    *
        \param[in] in the input array
        \param[in] ndims the number of dimensions
        \param[in] dims the array of new dimension sizes
        \return the modded output

        \ingroup manip_func_moddims
    */
    AFAPI array moddims(const array& in, const unsigned ndims, const dim_t* const dims);

    /**
        \param[in] in is the input array
        \return the flat array

        \ingroup manip_func_flat
    */
    AFAPI array flat(const array &in);

    /**
        \param[in] in is the input array
        \param[in] dim is the dimensions to flip the array
        \return the flipped array

        \ingroup manip_func_flip
    */
    AFAPI array flip(const array &in, const unsigned dim);

    /**
        \param[in] in is the input matrix
        \param[in] is_unit_diag is a boolean parameter specifying if the diagonal elements should be 1
        \return the lower triangle array

        \ingroup data_func_lower
    */
    AFAPI array lower(const array &in, bool is_unit_diag=false);

    /**
        \param[in] in is the input matrix
        \param[in] is_unit_diag is a boolean parameter specifying if the diagonal elements should be 1
        \return the upper triangle matrix

        \ingroup data_func_upper
    */
    AFAPI array upper(const array &in, bool is_unit_diag=false);

#if AF_API_VERSION >= 31
    /**
       \param[in]  cond is the conditional array
       \param[in]  a is the array containing elements from the true part of the condition
       \param[in]  b is the array containing elements from the false part of the condition
       \return  the output containing elements of \p a when \p cond is true else elements from \p b

       \ingroup data_func_select
    */
    AFAPI array select(const array &cond, const array  &a, const array  &b);
#endif

#if AF_API_VERSION >= 31
    /**
       \param[in]  cond is the conditional array
       \param[in]  a is the array containing elements from the true part of the condition
       \param[in]  b is a scalar assigned to \p out when \p cond is false
       \return  the output containing elements of \p a when \p cond is true else the value \p b

       \ingroup data_func_select
    */
    AFAPI array select(const array &cond, const array  &a, const double &b);
#endif

#if AF_API_VERSION >= 31
    /**
       \param[in]  cond is the conditional array
       \param[in]  a is a scalar assigned to \p out when \p cond is true
       \param[in]  b is the array containing elements from the false part of the condition
       \return  the output containing the value \p a when \p cond is true else elements from \p b

       \ingroup data_func_select
    */
    AFAPI array select(const array &cond, const double &a, const array  &b);
#endif

#if AF_API_VERSION >= 31
    /**
       \param[inout]  a is the input array
       \param[in]  cond is the conditional array.
       \param[in]  b is the replacement array.

       \note Values of \p a are replaced with corresponding values of \p b, when \p cond is false.

       \ingroup data_func_replace
    */
    AFAPI void replace(array &a, const array  &cond, const array  &b);
#endif

#if AF_API_VERSION >= 31
    /**
       \param[inout]  a is the input array
       \param[in]  cond is the conditional array.
       \param[in]  b is the replacement value.

       \note Values of \p a are replaced with corresponding values of \p b, when \p cond is false.

       \ingroup data_func_replace
    */
    AFAPI void replace(array &a, const array  &cond, const double &b);
#endif

#if AF_API_VERSION >= 37
    /**
       \param[in] in is the input array to be padded
       \param[in] beginPadding informs the number of elements to be
                  padded at beginning of each dimension
       \param[in] endPadding informs the number of elements to be
                  padded at end of each dimension
       \param[in] padFillType is indicates what values should fill padded region

       \return the padded array

       \ingroup data_func_pad
    */
    AFAPI array pad(const array &in, const dim4 &beginPadding,
                    const dim4 &endPadding, const borderType padFillType);
#endif
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    /**
        \param[out] arr is the generated array of given type
        \param[in] val is the value of each element in the generated array
        \param[in] ndims is size of dimension array \p dims
        \param[in] dims is the array containing sizes of the dimension
        \param[in] type is the type of array to generate

       \ingroup data_func_constant
    */
    AFAPI af_err af_constant(af_array *arr, const double val, const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
        \param[out] arr is the generated array of type \ref c32 or \ref c64
        \param[in] real is the real value of each element in the generated array
        \param[in] imag is the imaginary value of each element in the generated array
        \param[in] ndims is size of dimension array \p dims
        \param[in] dims is the array containing sizes of the dimension
        \param[in] type is the type of array to generate

       \ingroup data_func_constant
    */

    AFAPI af_err af_constant_complex(af_array *arr, const double real, const double imag,
                                     const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
        \param[out] arr is the generated array of type \ref s64
        \param[in] val is a complex value of each element in the generated array
        \param[in] ndims is size of dimension array \p dims
        \param[in] dims is the array containing sizes of the dimension

       \ingroup data_func_constant
    */

    AFAPI af_err af_constant_long (af_array *arr, const long long val, const unsigned ndims, const dim_t * const dims);

    /**
        \param[out] arr is the generated array of type \ref u64
        \param[in] val is a complex value of each element in the generated array
        \param[in] ndims is size of dimension array \p dims
        \param[in] dims is the array containing sizes of the dimension

       \ingroup data_func_constant
    */

    AFAPI af_err af_constant_ulong(af_array *arr, const unsigned long long val, const unsigned ndims, const dim_t * const dims);

    /**
    * C Interface for creating an array with `[0, n-1]` values along the `seq_dim` dimension and tiled across other dimensions specified by an array of `ndims` dimensions.
    *
        \param[out] out the generated array
        \param[in] ndims the size of dimension array `dims`
        \param[in] dims the array containing the dimension sizes
        \param[in] seq_dim the dimension along which `[0, dim[seq_dim] - 1]` is created
        \param[in] type the type of the generated array

        \ingroup data_func_range
    */
    AFAPI af_err af_range(af_array *out, const unsigned ndims, const dim_t * const dims,
                          const int seq_dim, const af_dtype type);

    /**
        \param[out] out is the generated array
        \param[in] ndims is size of dimension array \p dims
        \param[in] dims is the array containing sizes of the dimension
        \param[in] t_ndims is size of tile array \p tdims
        \param[in] tdims is array containing the number of repetitions of the unit dimensions
        \param[in] type is the type of array to generate

        \ingroup data_func_iota
    */
    AFAPI af_err af_iota(af_array *out, const unsigned ndims, const dim_t * const dims,
                         const unsigned t_ndims, const dim_t * const tdims, const af_dtype type);


    /**
        \param[out] out is the generated array
        \param[in] ndims is size of dimension array \p dims
        \param[in] dims is the array containing sizes of the dimension
        \param[in] type is the type of array to generate

        \ingroup data_func_identity
    */
    AFAPI af_err af_identity(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
        \param[out] out is the array created from the input array \p in
        \param[in] in is the input array which is the diagonal
        \param[in] num is the diagonal index

        \ingroup data_func_diag
    */
    AFAPI af_err af_diag_create(af_array *out, const af_array in, const int num);

    /**
        \param[out] out is the \p num -th diagonal of \p in
        \param[in] in is the input matrix
        \param[in] num is the diagonal index

        \ingroup data_func_diag
    */
    AFAPI af_err af_diag_extract(af_array *out, const af_array in, const int num);

    /**
        \brief Join 2 arrays along \p dim

        \param[out] out is the generated array
        \param[in] dim is the dimension along which join occurs
        \param[in] first is the first input array
        \param[in] second is the second input array

        \note empty arrays will be ignored

        \ingroup manip_func_join
    */
    AFAPI af_err af_join(af_array *out, const int dim, const af_array first, const af_array second);

    /**
        \brief Join many arrays along \p dim

        Current limit is set to 10 arrays.

        \param[out] out is the generated array
        \param[in] dim is the dimension along which join occurs
        \param[in] n_arrays number of arrays to join
        \param[in] inputs is an array of af_arrays containing handles to the arrays to be joined

        \note empty arrays will be ignored

        \ingroup manip_func_join
    */
    AFAPI af_err af_join_many(af_array *out, const int dim, const unsigned n_arrays, const af_array *inputs);

    /**
        \param[out] out is the tiled version of the input array
        \param[in] in is the input matrix
        \param[in] x is the number of times \p in is copied along the first dimension
        \param[in] y is the number of times \p in is copied along the the second dimension
        \param[in] z is the number of times \p in is copied along the third dimension
        \param[in] w is the number of times \p in is copied along the fourth dimension

        \note \p x, \p y, \p z, and \p w includes the original in the count as
              well. Thus, if no duplicates are needed in a certain dimension,
              leave it as 1 (the default value for just one copy)

        \ingroup manip_func_tile
    */
    AFAPI af_err af_tile(af_array *out, const af_array in,
                         const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    /**
        \param[out] out is the reordered array
        \param[in] in is the input matrix
        \param[in] x specifies which dimension should be first
        \param[in] y specifies which dimension should be second
        \param[in] z specifies which dimension should be third
        \param[in] w specifies which dimension should be fourth

        \ingroup manip_func_reorder
    */
    AFAPI af_err af_reorder(af_array *out, const af_array in,
                            const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    /**
        \param[in] out is the shifted array
        \param[in] in is the input array
        \param[in] x specifies the shift along first dimension
        \param[in] y specifies the shift along second dimension
        \param[in] z specifies the shift along third dimension
        \param[in] w specifies the shift along fourth dimension

        \ingroup manip_func_shift
    */
    AFAPI af_err af_shift(af_array *out, const af_array in, const int x, const int y, const int z, const int w);

    /**
    * C Interface for modifying the dimensions of an input array to the shape specified by an array of `ndims` dimensions
    *
        \param[out] out the modded output
        \param[in] in the input array
        \param[in] ndims the number of dimensions
        \param[in] dims the array of new dimension sizes

        \ingroup manip_func_moddims
    */
    AFAPI af_err af_moddims(af_array *out, const af_array in, const unsigned ndims, const dim_t * const dims);

    /**
        \param[out] out is the flat array
        \param[in] in is the input array

        \ingroup manip_func_flat
    */
    AFAPI af_err af_flat(af_array *out, const af_array in);

    /**
        \param[out] out is the flipped array
        \param[in] in is the input array
        \param[in] dim is the dimensions to flip the array

        \ingroup manip_func_flip
    */
    AFAPI af_err af_flip(af_array *out, const af_array in, const unsigned dim);

    /**
        \param[out] out is the lower traingle matrix
        \param[in] in is the input matrix
        \param[in] is_unit_diag is a boolean parameter specifying if the diagonal elements should be 1

        \ingroup data_func_lower
    */
    AFAPI af_err af_lower(af_array *out, const af_array in, bool is_unit_diag);

    /**
        \param[out] out is the upper triangle matrix
        \param[in] in is the input matrix
        \param[in] is_unit_diag is a boolean parameter specifying if the diagonal elements should be 1

        \ingroup data_func_upper
    */
    AFAPI af_err af_upper(af_array *out, const af_array in, bool is_unit_diag);

#if AF_API_VERSION >= 31
    /**
       \param[out] out is the output containing elements of \p a when \p cond is true else elements from \p b
       \param[in]  cond is the conditional array
       \param[in]  a is the array containing elements from the true part of the condition
       \param[in]  b is the array containing elements from the false part of the condition

       \ingroup data_func_select
    */
    AFAPI af_err af_select(af_array *out, const af_array cond, const af_array a, const af_array b);
#endif

#if AF_API_VERSION >= 31
    /**
       \param[out] out is the output containing elements of \p a when \p cond is true else elements from \p b
       \param[in]  cond is the conditional array
       \param[in]  a is the array containing elements from the true part of the condition
       \param[in]  b is a scalar assigned to \p out when \p cond is false

       \ingroup data_func_select
    */
    AFAPI af_err af_select_scalar_r(af_array *out, const af_array cond, const af_array a, const double b);
#endif

#if AF_API_VERSION >= 31
    /**
       \param[out] out is the output containing elements of \p a when \p cond is true else elements from \p b
       \param[in]  cond is the conditional array
       \param[in]  a is a scalar assigned to \p out when \p cond is true
       \param[in]  b is the array containing elements from the false part of the condition

       \ingroup data_func_select
    */
    AFAPI af_err af_select_scalar_l(af_array *out, const af_array cond, const double a, const af_array b);
#endif

#if AF_API_VERSION >= 31
    /**
       \param[inout]  a is the input array
       \param[in]  cond is the conditional array.
       \param[in]  b is the replacement array.

       \note Values of \p a are replaced with corresponding values of \p b, when \p cond is false.

       \ingroup data_func_replace
    */
    AFAPI af_err af_replace(af_array a, const af_array cond, const af_array b);
#endif

#if AF_API_VERSION >= 31
    /**
       \param[inout]  a is the input array
       \param[in]  cond is the conditional array.
       \param[in]  b is the replacement array.

       \note Values of \p a are replaced with corresponding values of \p b, when \p cond is false.

       \ingroup data_func_replace
    */
    AFAPI af_err af_replace_scalar(af_array a, const af_array cond, const double b);
#endif

#if AF_API_VERSION >= 37
    /**
       \param[out] out is the padded array
       \param[in] in is the input array to be padded
       \param[in] begin_ndims is size of \p l_dims array
       \param[in] begin_dims array contains padding size at beginning of each
                  dimension
       \param[in] end_ndims is size of \p u_dims array
       \param[in] end_dims array contains padding sizes at end of each dimension
       \param[in] pad_fill_type is indicates what values should fill
                  padded region

       \ingroup data_func_pad
    */
    AFAPI af_err af_pad(af_array *out, const af_array in,
                        const unsigned begin_ndims,
                        const dim_t *const begin_dims, const unsigned end_ndims,
                        const dim_t *const end_dims,
                        const af_border_type pad_fill_type);
#endif

#ifdef __cplusplus
}
#endif

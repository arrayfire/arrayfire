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
       C++ Interface

       \param[in] in is the input array
       \paran[in] dim The dimension along which the add operation occurs
       \return    result of sum all values along dimension \p dim

       \ingroup reduce_func_sum

       \note \p dim is -1 by default. -1 denotes the first non-signleton dimension.
    */
    AFAPI array sum(const array &in, const int dim = -1);

    /**
       C++ Interface

       \param[in] in is the input array
       \paran[in] dim The dimension along which the multiply operation occurs
       \return    result of product all values along dimension \p dim

       \ingroup reduce_func_product

       \note \p dim is -1 by default. -1 denotes the first non-signleton dimension.
    */
    AFAPI array product(const array &in, const int dim = -1);

    /**
       C++ Interface

       \param[in] in is the input array
       \paran[in] dim The dimension along which the minimum value needs to be extracted
       \return    result of minimum all values along dimension \p dim

       \ingroup reduce_func_min

       \note \p dim is -1 by default. -1 denotes the first non-signleton dimension.
    */
    AFAPI array min(const array &in, const int dim = -1);

    /**
       C++ Interface

       \param[in] in is the input array
       \paran[in] dim The dimension along which the maximum value needs to be extracted
       \return    result of maximum all values along dimension \p dim

       \ingroup reduce_func_max

       \note \p dim is -1 by default. -1 denotes the first non-signleton dimension.
    */
    AFAPI array max(const array &in, const int dim = -1);

    /**
       C++ Interface

       \param[in] in is the input array
       \paran[in] dim The dimension along which the values are checked to be all true
       \return    result of checking if values along dimension \p dim are all true

       \ingroup reduce_functrue

       \note \p dim is -1 by default. -1 denotes the first non-signleton dimension.
    */
    AFAPI array alltrue(const array &in, const int dim = -1);

    /**
       C++ Interface

       \param[in] in is the input array
       \paran[in] dim The dimension along which the values are checked to be any true
       \return    result of checking if values along dimension \p dim are any true

       \ingroup reduce_func_anytrue

       \note \p dim is -1 by default. -1 denotes the first non-signleton dimension.
    */
    AFAPI array anytrue(const array &in, const int dim = -1);

    /**
       C++ Interface

       \param[in] in is the input array
       \paran[in] dim The dimension along which the the number of non-zero values are counted
       \return    the number of non-zero values along dimension \p dim

       \ingroup reduce_func_count

       \note \p dim is -1 by default. -1 denotes the first non-signleton dimension.
    */
    AFAPI array count(const array &in, const int dim = -1);

    /**
       C++ Interface

       \param[in] in is the input array
       \return    the sum of all values of \p in

       \ingroup reduce_func_sum
    */
    template<typename T> T sum(const array &in);

    /**
       C++ Interface

       \param[in] in is the input array
       \return    the product of all values of \p in

       \ingroup reduce_func_product
    */
    template<typename T> T product(const array &in);

    /**
       C++ Interface

       \param[in] in is the input array
       \return    the minimum of all values of \p in

       \ingroup reduce_func_min
    */
    template<typename T> T min(const array &in);

    /**
       C++ Interface

       \param[in] in is the input array
       \return    the maximum of all values of \p in

       \ingroup reduce_func_max
    */
    template<typename T> T max(const array &in);

    /**
       C++ Interface

       \param[in] in is the input array
       \return    true if all values of \p in are true, false otherwise

       \ingroup reduce_func_alltrue
    */
    template<typename T> T alltrue(const array &in);

    /**
       C++ Interface

       \param[in] in is the input array
       \return    true if any values of \p in are true, false otherwise

       \ingroup reduce_func_anytrue
    */
    template<typename T> T anytrue(const array &in);

    /**
       C++ Interface

       \param[in] in is the input array
       \return    the number of non-zero values in \p in

       \ingroup reduce_func_count
    */
    template<typename T> T count(const array &in);

    /**
       C++ Interface

       \param[out] val contains the minimum values along dimension \p dim
       \param[out] idx contains the locations of minimum all values along dimension \p dim
       \param[in]  in is the input array
       \paran[in]  dim The dimension along which the minimum value needs to be extracted

       \ingroup reduce_func_min

       \note \p dim is -1 by default. -1 denotes the first non-signleton dimension.
    */
    AFAPI void min(array &val, array &idx, const array &in, const int dim = 0);

    /**
       C++ Interface

       \param[out] val contains the maximum values along dimension \p dim
       \param[out] idx contains the locations of maximum all values along dimension \p dim
       \param[in]  in is the input array
       \paran[in]  dim The dimension along which the maximum value needs to be extracted

       \ingroup reduce_func_max

       \note \p dim is -1 by default. -1 denotes the first non-signleton dimension.
    */
    AFAPI void max(array &val, array &idx, const array &in, const int dim = 0);

    /**
       C++ Interface

       \param[out] val contains the minimum values in the input
       \param[out] idx contains the locations of minimum all values in the input
       \param[in]  in is the input array

       \ingroup reduce_func_min
    */
    template<typename T> void min(T *val, unsigned *idx, const array &in);

    /**
       C++ Interface

       \param[out] val contains the maximum values in the input
       \param[out] idx contains the locations of maximum all values in the input
       \param[in]  in is the input array

       \ingroup reduce_func_max
    */
    template<typename T> void max(T *val, unsigned *idx, const array &in);


    AFAPI array diff1(const array &in, const int dim = 0);

    AFAPI array diff2(const array &in, const int dim = 0);

    AFAPI array accum(const array &in, const int dim = 0);

    AFAPI array where(const array &in);

    AFAPI array sort(const array &in, const unsigned dim = 0, const bool isAscending = true);

    AFAPI void  sort(array &out, array &indices, const array &in, const unsigned dim = 0,
                     const bool isAscending = true);

    AFAPI void  sort(array &out_keys, array & out_values, const array &keys, const array &values,
                     const unsigned dim = 0, const bool isAscending = true);

    AFAPI array setunique(const array &in, bool is_sorted=false);

    AFAPI array setunion(const array &first, const array &second, bool is_unique=false);

    AFAPI array setintersect(const array &first, const array &second, bool is_unique=false);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       C Interface

       \param[out] out contains the sum of all values in \p in along \p dim
       \param[in] in is the input array
       \paran[in] dim The dimension along which the add operation occurs
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_sum
    */
    AFAPI af_err af_sum(af_array *out, const af_array in, const int dim);

    /**
       C Interface

       \param[out] out contains the product of all values in \p in along \p dim
       \param[in] in is the input array
       \paran[in] dim The dimension along which the multiply operation occurs
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_product
    */
    AFAPI af_err af_product(af_array *out, const af_array in, const int dim);

    /**
       C Interface

       \param[out] out contains the minimum of all values in \p in along \p dim
       \param[in] in is the input array
       \paran[in] dim The dimension along which the minimum value is extracted
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_min
    */
    AFAPI af_err af_min(af_array *out, const af_array in, const int dim);

    /**
       C Interface

       \param[out] out contains the maximum of all values in \p in along \p dim
       \param[in] in is the input array
       \paran[in] dim The dimension along which the maximum value is extracted
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_max
    */
    AFAPI af_err af_max(af_array *out, const af_array in, const int dim);

    /**
       C Interface

       \param[out] out contains the result of "and" operation all values in \p in along \p dim
       \param[in] in is the input array
       \paran[in] dim The dimension along which the "and" operation occurs
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_alltrue
    */
    AFAPI af_err af_alltrue(af_array *out, const af_array in, const int dim);

    /**
       C Interface

       \param[out] out contains the result of "or" operation all values in \p in along \p dim
       \param[in] in is the input array
       \paran[in] dim The dimension along which the "or" operation occurs
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_anytrue
    */
    AFAPI af_err af_anytrue(af_array *out, const af_array in, const int dim);

    /**
       C Interface

       \param[out] out contains the number of non-zero values in \p in along \p dim
       \param[in] in is the input array
       \paran[in] dim The dimension along which the non-zero values are counted
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_count
    */
    AFAPI af_err af_count(af_array *out, const af_array in, const int dim);

    /**
       C Interface

       \param[out] real contains the real part of adding all elements in input \p in
       \param[out] imag contains the imaginary part of adding all elements in input \p in
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0 when \p in is real

       \ingroup reduce_func_sum
    */
    AFAPI af_err af_sum_all(double *real, double *imag, const af_array in);

    /**
       C Interface

       \param[out] real contains the real part of multiplying all elements in input \p in
       \param[out] imag contains the imaginary part of multiplying all elements in input \p in
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0 when \p in is real

       \ingroup reduce_func_product
    */
    AFAPI af_err af_product_all(double *real, double *imag, const af_array in);

    /**
       C Interface

       \param[out] real contains the real part of minimum value of all elements in input \p in
       \param[out] imag contains the imaginary part of minimum value of all elements in input \p in
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0 when \p in is real.

       \ingroup reduce_func_min
    */
    AFAPI af_err af_min_all(double *real, double *imag, const af_array in);

    /**
       C Interface

       \param[out] real contains the real part of maximum value of all elements in input \p in
       \param[out] imag contains the imaginary part of maximum value of all elements in input \p in
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0 when \p in is real.

       \ingroup reduce_func_max
    */
    AFAPI af_err af_max_all(double *real, double *imag, const af_array in);

    /**
       C Interface

       \param[out] real is 1 if all values of input \p in are true. 0 otherwise.
       \param[out] imag is always set to 0.
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0.

       \ingroup reduce_func_alltrue
    */
    AFAPI af_err af_alltrue_all(double *real, double *imag, const af_array in);

    /**
       C Interface

       \param[out] real is 1 if any value of input \p in is true. 0 otherwise.
       \param[out] imag is always set to 0.
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0.

       \ingroup reduce_func_anytrue
    */
    AFAPI af_err af_anytrue_all(double *real, double *imag, const af_array in);

    /**
       C Interface

       \param[out] real contains the number of non-zero values in \p in.
       \param[out] imag is always set to 0.
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0.

       \ingroup reduce_func_count
    */
    AFAPI af_err af_count_all(double *real, double *imag, const af_array in);

    /**
       C Interface

       \param[out] out contains the minimum of all values in \p in along \p dim
       \param[out] idx contains the location of minimum of all values in \p in along \p dim
       \param[in] in is the input array
       \paran[in] dim The dimension along which the minimum value is extracted
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_min
    */
    AFAPI af_err af_imin(af_array *out, af_array *idx, const af_array in, const int dim);

    /**
       C Interface

       \param[out] out contains the maximum of all values in \p in along \p dim
       \param[out] idx contains the location of maximum of all values in \p in along \p dim
       \param[in] in is the input array
       \paran[in] dim The dimension along which the maximum value is extracted
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_max
    */
    AFAPI af_err af_imax(af_array *out, af_array *idx, const af_array in, const int dim);

    /**
       C Interface

       \param[out] real contains the real part of minimum value of all elements in input \p in
       \param[out] imag contains the imaginary part of minimum value of all elements in input \p in
       \param[out] idx idx contains the location of minimum of all values in \p in
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0 when \p in is real.

       \ingroup reduce_func_min
    */
    AFAPI af_err af_imin_all(double *real, double *imag, unsigned *idx, const af_array in);

    /**
       C Interface

       \param[out] real contains the real part of maximum value of all elements in input \p in
       \param[out] imag contains the imaginary part of maximum value of all elements in input \p in
       \param[out] idx idx contains the location of maximum of all values in \p in
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0 when \p in is real.

       \ingroup reduce_func_max
    */
    AFAPI af_err af_imax_all(double *real, double *imag, unsigned *idx, const af_array in);

    // Compute first order difference along a given dimension.
    AFAPI af_err af_diff1(af_array *out, const af_array in, const int dim);

    // Compute second order difference along a given dimension.
    AFAPI af_err af_diff2(af_array *out, const af_array in, const int dim);

    // Inclusive sum of all the elements along an array
    AFAPI af_err af_accum(af_array *out, const af_array in, const int dim);

    AFAPI af_err af_where(af_array *idx, const af_array in);

    // Sort
    AFAPI af_err af_sort(af_array *out, const af_array in, const unsigned dim, const bool isAscending);

    AFAPI af_err af_sort_index(af_array *out, af_array *indices, const af_array in,
                               const unsigned dim, const bool isAscending);

    AFAPI af_err af_sort_by_key(af_array *out_keys, af_array *out_values,
                                const af_array keys, const af_array values, const unsigned dim, const bool isAscending);

    AFAPI af_err af_set_unique(af_array *out, const af_array in, const bool is_sorted);

    AFAPI af_err af_set_union(af_array *out, const af_array first, const af_array second, const bool is_unique);

    AFAPI af_err af_set_intersect(af_array *out, const af_array first, const af_array second, const bool is_unique);

#ifdef __cplusplus
}
#endif

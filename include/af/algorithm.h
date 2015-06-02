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
       C++ Interface for sum of elements in an array

       \param[in] in is the input array
       \param[in] dim The dimension along which the add operation occurs
       \return    result of sum all values along dimension \p dim

       \ingroup reduce_func_sum

       \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
    */
    AFAPI array sum(const array &in, const int dim = -1);

    /**
       C++ Interface for product of elements in an array

       \param[in] in is the input array
       \param[in] dim The dimension along which the multiply operation occurs
       \return    result of product all values along dimension \p dim

       \ingroup reduce_func_product

       \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
    */
    AFAPI array product(const array &in, const int dim = -1);

    /**
       C++ Interface for minimum values in an array

       \param[in] in is the input array
       \param[in] dim The dimension along which the minimum value needs to be extracted
       \return    result of minimum all values along dimension \p dim

       \ingroup reduce_func_min

       \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
    */
    AFAPI array min(const array &in, const int dim = -1);

    /**
       C++ Interface for maximum values in an array

       \param[in] in is the input array
       \param[in] dim The dimension along which the maximum value needs to be extracted
       \return    result of maximum all values along dimension \p dim

       \ingroup reduce_func_max

       \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
    */
    AFAPI array max(const array &in, const int dim = -1);

    /**
       C++ Interface for checking all true values in an array

       \param[in] in is the input array
       \param[in] dim The dimension along which the values are checked to be all true
       \return    result of checking if values along dimension \p dim are all true

       \ingroup reduce_func_all_true

       \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
    */
    AFAPI array allTrue(const array &in, const int dim = -1);

    /**
       C++ Interface for checking any true values in an array

       \param[in] in is the input array
       \param[in] dim The dimension along which the values are checked to be any true
       \return    result of checking if values along dimension \p dim are any true

       \ingroup reduce_func_any_true

       \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
    */
    AFAPI array anyTrue(const array &in, const int dim = -1);

    /**
       C++ Interface for counting non-zero values in an array

       \param[in] in is the input array
       \param[in] dim The dimension along which the the number of non-zero values are counted
       \return    the number of non-zero values along dimension \p dim

       \ingroup reduce_func_count

       \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
    */
    AFAPI array count(const array &in, const int dim = -1);

    /**
       C++ Interface for sum of all elements in an array

       \param[in] in is the input array
       \return    the sum of all values of \p in

       \ingroup reduce_func_sum
    */
    template<typename T> T sum(const array &in);

    /**
       C++ Interface for product of all elements in an array

       \param[in] in is the input array
       \return    the product of all values of \p in

       \ingroup reduce_func_product
    */
    template<typename T> T product(const array &in);

    /**
       C++ Interface for getting minimum value of an array

       \param[in] in is the input array
       \return    the minimum of all values of \p in

       \ingroup reduce_func_min
    */
    template<typename T> T min(const array &in);

    /**
       C++ Interface for getting maximum value of an array

       \param[in] in is the input array
       \return    the maximum of all values of \p in

       \ingroup reduce_func_max
    */
    template<typename T> T max(const array &in);

    /**
       C++ Interface for checking if all values in an array are true

       \param[in] in is the input array
       \return    true if all values of \p in are true, false otherwise

       \ingroup reduce_func_all_true
    */
    template<typename T> T allTrue(const array &in);

    /**
       C++ Interface for checking if any values in an array are true

       \param[in] in is the input array
       \return    true if any values of \p in are true, false otherwise

       \ingroup reduce_func_any_true
    */
    template<typename T> T anyTrue(const array &in);

    /**
       C++ Interface for counting total number of non-zero values in an array

       \param[in] in is the input array
       \return    the number of non-zero values in \p in

       \ingroup reduce_func_count
    */
    template<typename T> T count(const array &in);

    /**
       C++ Interface for getting minimum values and their locations in an array

       \param[out] val will contain the minimum values along dimension \p dim
       \param[out] idx will contain the locations of minimum all values along dimension \p dim
       \param[in]  in is the input array
       \param[in]  dim The dimension along which the minimum value needs to be extracted

       \ingroup reduce_func_min

       \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
    */
    AFAPI void min(array &val, array &idx, const array &in, const int dim = -1);

    /**
       C++ Interface for getting maximum values and their locations in an array

       \param[out] val will contain the maximum values along dimension \p dim
       \param[out] idx will contain the locations of maximum all values along dimension \p dim
       \param[in]  in is the input array
       \param[in]  dim The dimension along which the maximum value needs to be extracted

       \ingroup reduce_func_max

       \note \p dim is -1 by default. -1 denotes the first non-singleton dimension.
    */
    AFAPI void max(array &val, array &idx, const array &in, const int dim = -1);

    /**
       C++ Interface for getting minimum value and its location from the entire array

       \param[out] val will contain the minimum values in the input
       \param[out] idx will contain the locations of minimum all values in the input
       \param[in]  in is the input array

       \ingroup reduce_func_min
    */
    template<typename T> void min(T *val, unsigned *idx, const array &in);

    /**
       C++ Interface for getting maximum value and its location from the entire array

       \param[out] val contains the maximum values in the input
       \param[out] idx contains the locations of maximum all values in the input
       \param[in]  in is the input array

       \ingroup reduce_func_max
    */
    template<typename T> void max(T *val, unsigned *idx, const array &in);

    /**
       C++ Interface exclusive sum (cumulative sum) of an array

       \param[in] in is the input array
       \param[in] dim The dimension along which exclusive sum is performed
       \return the output containing exclusive sums of the input

       \ingroup scan_func_accum
    */
    AFAPI array accum(const array &in, const int dim = 0);

    /**
       C++ Interface for finding the locations of non-zero values in an array

       \param[in] in is the input array.
       \return linear indices where \p in is non-zero

       \ingroup scan_func_where
    */
    AFAPI array where(const array &in);

    /**
       C++ Interface for calculating first order differences in an array

       \param[in] in is the input array
       \param[in] dim The dimension along which numerical difference is performed
       \return array of first order numerical difference

       \ingroup calc_func_diff1
    */
    AFAPI array diff1(const array &in, const int dim = 0);

    /**
       C++ Interface for calculating second order differences in an array

       \param[in] in is the input array
       \param[in] dim The dimension along which numerical difference is performed
       \return array of second order numerical difference

       \ingroup calc_func_diff2
    */
    AFAPI array diff2(const array &in, const int dim = 0);

    /**
       C++ Interface for sorting an array

       \param[in] in is the input array
       \param[in] dim The dimension along which numerical difference is performed
       \param[in] isAscending specifies the sorting order
       \return the sorted output

       \ingroup sort_func_sort

       \note \p dim is currently restricted to 0.
    */
    AFAPI array sort(const array &in, const unsigned dim = 0, const bool isAscending = true);

    /**
       C++ Interface for sorting an array and getting original indices

       \param[out] out will contain the sorted output
       \param[out] indices will contain the indices in the original input
       \param[in] in is the input array
       \param[in] dim The dimension along which numerical difference is performed
       \param[in] isAscending specifies the sorting order

       \ingroup sort_func_sort_index

       \note \p dim is currently restricted to 0.
    */
    AFAPI void  sort(array &out, array &indices, const array &in, const unsigned dim = 0,
                     const bool isAscending = true);
    /**
       C++ Interface for sorting an array based on keys

       \param[out] out_keys will contain the keys based on sorted values
       \param[out] out_values will contain the sorted values
       \param[in] keys is the input array
       \param[in] values The dimension along which numerical difference is performed
       \param[in] dim The dimension along which numerical difference is performed
       \param[in] isAscending specifies the sorting order

       \ingroup sort_func_sort_keys

       \note \p dim is currently restricted to 0.
    */
    AFAPI void  sort(array &out_keys, array &out_values, const array &keys, const array &values,
                     const unsigned dim = 0, const bool isAscending = true);

    /**
       C++ Interface for getting unique values

       \param[in] in is the input array
       \param[in] is_sorted if true, skips the sorting steps internally
       \return the unique values from \p in

       \ingroup set_func_unique
    */
    AFAPI array setUnique(const array &in, const bool is_sorted=false);

    /**
       C++ Interface for performing union of two arrays

       \param[in] first is the first array
       \param[in] second is the second array
       \param[in] is_unique if true, skips calling unique internally
       \return the union of \p first and \p second

       \ingroup set_func_union
    */
    AFAPI array setUnion(const array &first, const array &second, const bool is_unique=false);

    /**
       C++ Interface for performing intersect of two arrays

       \param[in] first is the first array
       \param[in] second is the second array
       \param[in] is_unique if true, skips calling unique internally
       \return the intersection of \p first and \p second

       \ingroup set_func_intersect
    */
    AFAPI array setIntersect(const array &first, const array &second, const bool is_unique=false);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       C Interface for sum of elements in an array

       \param[out] out will contain the sum of all values in \p in along \p dim
       \param[in] in is the input array
       \param[in] dim The dimension along which the add operation occurs
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_sum
    */
    AFAPI af_err af_sum(af_array *out, const af_array in, const int dim);

    /**
       C Interface for product of elements in an array

       \param[out] out will contain the product of all values in \p in along \p dim
       \param[in] in is the input array
       \param[in] dim The dimension along which the multiply operation occurs
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_product
    */
    AFAPI af_err af_product(af_array *out, const af_array in, const int dim);

    /**
       C Interface for minimum values in an array

       \param[out] out will contain the minimum of all values in \p in along \p dim
       \param[in] in is the input array
       \param[in] dim The dimension along which the minimum value is extracted
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_min
    */
    AFAPI af_err af_min(af_array *out, const af_array in, const int dim);

    /**
       C Interface for maximum values in an array

       \param[out] out will contain the maximum of all values in \p in along \p dim
       \param[in] in is the input array
       \param[in] dim The dimension along which the maximum value is extracted
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_max
    */
    AFAPI af_err af_max(af_array *out, const af_array in, const int dim);

    /**
       C Interface for checking all true values in an array

       \param[out] out will contain the result of "and" operation all values in \p in along \p dim
       \param[in] in is the input array
       \param[in] dim The dimension along which the "and" operation occurs
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_all_true
    */
    AFAPI af_err af_all_true(af_array *out, const af_array in, const int dim);

    /**
       C Interface for checking any true values in an array

       \param[out] out will contain the result of "or" operation all values in \p in along \p dim
       \param[in] in is the input array
       \param[in] dim The dimension along which the "or" operation occurs
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_any_true
    */
    AFAPI af_err af_any_true(af_array *out, const af_array in, const int dim);

    /**
       C Interface for counting non-zero values in an array

       \param[out] out will contain the number of non-zero values in \p in along \p dim
       \param[in] in is the input array
       \param[in] dim The dimension along which the non-zero values are counted
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_count
    */
    AFAPI af_err af_count(af_array *out, const af_array in, const int dim);

    /**
       C Interface for sum of all elements in an array

       \param[out] real will contain the real part of adding all elements in input \p in
       \param[out] imag will contain the imaginary part of adding all elements in input \p in
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0 when \p in is real

       \ingroup reduce_func_sum
    */
    AFAPI af_err af_sum_all(double *real, double *imag, const af_array in);

    /**
       C Interface for product of all elements in an array

       \param[out] real will contain the real part of multiplying all elements in input \p in
       \param[out] imag will contain the imaginary part of multiplying all elements in input \p in
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0 when \p in is real

       \ingroup reduce_func_product
    */
    AFAPI af_err af_product_all(double *real, double *imag, const af_array in);

    /**
       C Interface for getting minimum value of an array

       \param[out] real will contain the real part of minimum value of all elements in input \p in
       \param[out] imag will contain the imaginary part of minimum value of all elements in input \p in
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0 when \p in is real.

       \ingroup reduce_func_min
    */
    AFAPI af_err af_min_all(double *real, double *imag, const af_array in);

    /**
       C Interface for getting maximum value of an array

       \param[out] real will contain the real part of maximum value of all elements in input \p in
       \param[out] imag will contain the imaginary part of maximum value of all elements in input \p in
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0 when \p in is real.

       \ingroup reduce_func_max
    */
    AFAPI af_err af_max_all(double *real, double *imag, const af_array in);

    /**
       C Interface for checking if all values in an array are true

       \param[out] real is 1 if all values of input \p in are true, 0 otherwise.
       \param[out] imag is always set to 0.
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0.

       \ingroup reduce_func_all_true
    */
    AFAPI af_err af_all_true_all(double *real, double *imag, const af_array in);

    /**
       C Interface for checking if any values in an array are true

       \param[out] real is 1 if any value of input \p in is true, 0 otherwise.
       \param[out] imag is always set to 0.
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0.

       \ingroup reduce_func_any_true
    */
    AFAPI af_err af_any_true_all(double *real, double *imag, const af_array in);

    /**
       C Interface for counting total number of non-zero values in an array

       \param[out] real will contain the number of non-zero values in \p in.
       \param[out] imag is always set to 0.
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0.

       \ingroup reduce_func_count
    */
    AFAPI af_err af_count_all(double *real, double *imag, const af_array in);

    /**
       C Interface for getting minimum values and their locations in an array

       \param[out] out will contain the minimum of all values in \p in along \p dim
       \param[out] idx will contain the location of minimum of all values in \p in along \p dim
       \param[in] in is the input array
       \param[in] dim The dimension along which the minimum value is extracted
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_min
    */
    AFAPI af_err af_imin(af_array *out, af_array *idx, const af_array in, const int dim);

    /**
       C Interface for getting maximum values and their locations in an array

       \param[out] out will contain the maximum of all values in \p in along \p dim
       \param[out] idx will contain the location of maximum of all values in \p in along \p dim
       \param[in] in is the input array
       \param[in] dim The dimension along which the maximum value is extracted
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup reduce_func_max
    */
    AFAPI af_err af_imax(af_array *out, af_array *idx, const af_array in, const int dim);

    /**
       C Interface for getting minimum value and its location from the entire array

       \param[out] real will contain the real part of minimum value of all elements in input \p in
       \param[out] imag will contain the imaginary part of minimum value of all elements in input \p in
       \param[out] idx will contain the location of minimum of all values in \p in
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0 when \p in is real.

       \ingroup reduce_func_min
    */
    AFAPI af_err af_imin_all(double *real, double *imag, unsigned *idx, const af_array in);

    /**
       C Interface for getting maximum value and it's location from the entire array

       \param[out] real will contain the real part of maximum value of all elements in input \p in
       \param[out] imag will contain the imaginary part of maximum value of all elements in input \p in
       \param[out] idx will contain the location of maximum of all values in \p in
       \param[in] in is the input array
       \return \ref AF_SUCCESS if the execution completes properly

       \note \p imag is always set to 0 when \p in is real.

       \ingroup reduce_func_max
    */
    AFAPI af_err af_imax_all(double *real, double *imag, unsigned *idx, const af_array in);

    /**
       C Interface exclusive sum (cumulative sum) of an array

       \param[out] out will contain exclusive sums of the input
       \param[in] in is the input array
       \param[in] dim The dimension along which exclusive sum is performed
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup scan_func_accum
    */
    AFAPI af_err af_accum(af_array *out, const af_array in, const int dim);

    /**
       C Interface for finding the locations of non-zero values in an array

       \param[out] idx will contain indices where \p in is non-zero
       \param[in] in is the input array.
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup scan_func_where
    */
    AFAPI af_err af_where(af_array *idx, const af_array in);

    /**
       C Interface for calculating first order differences in an array

       \param[out] out will contain the first order numerical differences of \p in
       \param[in] in is the input array
       \param[in] dim The dimension along which numerical difference is performed
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup calc_func_diff1
    */
    AFAPI af_err af_diff1(af_array *out, const af_array in, const int dim);

    /**
       C Interface for calculating second order differences in an array

       \param[out] out will contain the second order numerical differences of \p in
       \param[in] in is the input array
       \param[in] dim The dimension along which numerical difference is performed
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup calc_func_diff2
    */
    AFAPI af_err af_diff2(af_array *out, const af_array in, const int dim);

    /**
       C Interface for sorting an array

       \param[out] out will contain the sorted output
       \param[in] in is the input array
       \param[in] dim The dimension along which numerical difference is performed
       \param[in] isAscending specifies the sorting order
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sort_func_sort

       \note \p dim is currently restricted to 0.
    */
    AFAPI af_err af_sort(af_array *out, const af_array in, const unsigned dim, const bool isAscending);

    /**
       C Interface for sorting an array and getting original indices

       \param[out] out will contain the sorted output
       \param[out] indices will contain the indices in the original input
       \param[in] in is the input array
       \param[in] dim The dimension along which numerical difference is performed
       \param[in] isAscending specifies the sorting order
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sort_func_sort_index

       \note \p dim is currently restricted to 0.
    */
    AFAPI af_err af_sort_index(af_array *out, af_array *indices, const af_array in,
                               const unsigned dim, const bool isAscending);
    /**
       C Interface for sorting an array based on keys

       \param[out] out_keys will contain the keys based on sorted values
       \param[out] out_values will contain the sorted values
       \param[in] keys is the input array
       \param[in] values The dimension along which numerical difference is performed
       \param[in] dim The dimension along which numerical difference is performed
       \param[in] isAscending specifies the sorting order
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup sort_func_sort_keys

       \note \p dim is currently restricted to 0.
    */
    AFAPI af_err af_sort_by_key(af_array *out_keys, af_array *out_values,
                                const af_array keys, const af_array values,
                                const unsigned dim, const bool isAscending);

    /**
       C Interface for getting unique values

       \param[out] out will contain the unique values from \p in
       \param[in] in is the input array
       \param[in] is_sorted if true, skips the sorting steps internally
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup set_func_unique
    */
    AFAPI af_err af_set_unique(af_array *out, const af_array in, const bool is_sorted);

    /**
       C Interface for performing union of two arrays

       \param[out] out will contain the union of \p first and \p second
       \param[in] first is the first array
       \param[in] second is the second array
       \param[in] is_unique if true, skips calling unique internally
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup set_func_union
    */
    AFAPI af_err af_set_union(af_array *out, const af_array first, const af_array second, const bool is_unique);

    /**
       C Interface for performing intersect of two arrays

       \param[out] out will contain the intersection of \p first and \p second
       \param[in] first is the first array
       \param[in] second is the second array
       \param[in] is_unique if true, skips calling unique internally
       \return \ref AF_SUCCESS if the execution completes properly

       \ingroup set_func_intersect
    */
    AFAPI af_err af_set_intersect(af_array *out, const af_array first, const af_array second, const bool is_unique);

#ifdef __cplusplus
}
#endif

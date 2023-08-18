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
       C++ Interface to sum array elements over a given dimension.

       \param[in] in  input array
       \param[in] dim dimension along which the summation occurs, -1 denotes
                      the first non-singleton dimension
       \return        sum

       \ingroup reduce_func_sum
    */
    AFAPI array sum(const array &in, const int dim = -1);

#if AF_API_VERSION >= 31
    /**
       C++ Interface to sum array elements over a given dimension, replacing
       any NaNs with a specified value.

       \param[in] in     input array
       \param[in] dim    dimension along which the summation occurs
       \param[in] nanval value that replaces NaNs
       \return           sum

       \ingroup reduce_func_sum
    */
    AFAPI array sum(const array &in, const int dim, const double nanval);
#endif

#if AF_API_VERSION >= 37
    /**
       C++ Interface to sum array elements over a given dimension, according to
       an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out sum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the summation occurs, -1
                            denotes the first non-singleton dimension

       \ingroup reduce_func_sum_by_key
    */
    AFAPI void sumByKey(array &keys_out, array &vals_out,
                        const array &keys, const array &vals,
                        const int dim = -1);

    /**
       C++ Interface to sum array elements over a given dimension, replacing
       any NaNs with a specified value, according to an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out sum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the summation occurs
       \param[in]  nanval   value that replaces NaNs

       \ingroup reduce_func_sum_by_key
    */
    AFAPI void sumByKey(array &keys_out, array &vals_out,
                        const array &keys, const array &vals,
                        const int dim, const double nanval);
#endif

    /**
       C++ Interface to multiply array elements over a given dimension.

       \param[in] in  input array
       \param[in] dim dimension along which the product occurs, -1 denotes the
                      first non-singleton dimension
       \return        product

       \ingroup reduce_func_product
    */
    AFAPI array product(const array &in, const int dim = -1);

#if AF_API_VERSION >= 31
    /**
       C++ Interface to multiply array elements over a given dimension,
       replacing any NaNs with a specified value.

       \param[in] in     input array
       \param[in] dim    dimension along which the product occurs
       \param[in] nanval value that replaces NaNs
       \return           product

       \ingroup reduce_func_product
    */
    AFAPI array product(const array &in, const int dim, const double nanval);
#endif

#if AF_API_VERSION >= 37
    /**
       C++ Interface to multiply array elements over a given dimension,
       according to an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out product
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the product occurs, -1
                            denotes the first non-singleton dimension

       \ingroup reduce_func_product_by_key
    */
    AFAPI void productByKey(array &keys_out, array &vals_out,
                            const array &keys, const array &vals,
                            const int dim = -1);

    /**
       C++ Interface to multiply array elements over a given dimension,
       replacing any NaNs with a specified value, according to an array of
       keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out product
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the product occurs
       \param[in]  nanval   value that replaces NaNs

       \ingroup reduce_func_product_by_key

    */
    AFAPI void productByKey(array &keys_out, array &vals_out,
                            const array &keys, const array &vals,
                            const int dim, const double nanval);
#endif

    /**
       C++ Interface to return the minimum along a given dimension.

       NaN values are ignored.

       \param[in] in  input array
       \param[in] dim dimension along which the minimum is found, -1 denotes
                      the first non-singleton dimension
       \return        minimum

       \ingroup reduce_func_min
    */
    AFAPI array min(const array &in, const int dim = -1);

#if AF_API_VERSION >= 37
    /**
       C++ Interface to return the minimum along a given dimension, according
       to an array of keys.

       NaN values are ignored.

       \param[out] keys_out reduced keys
       \param[out] vals_out minimum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the minimum is found, -1
                            denotes the first non-singleton dimension

       \ingroup reduce_func_min_by_key
    */
    AFAPI void minByKey(array &keys_out, array &vals_out,
                        const array &keys, const array &vals,
                        const int dim = -1);
#endif

    /**
       C++ Interface to return the maximum along a given dimension.

       NaN values are ignored.

       \param[in] in  input array
       \param[in] dim dimension along which the maximum is found, -1 denotes
                      the first non-singleton dimension
       \return        maximum

       \ingroup reduce_func_max
    */
    AFAPI array max(const array &in, const int dim = -1);

#if AF_API_VERSION >= 37
    /**
       C++ Interface to return the maximum along a given dimension, according
       to an array of keys.

       NaN values are ignored.

       \param[out] keys_out reduced keys
       \param[out] vals_out maximum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the maximum is found, -1
                            denotes the first non-singleton dimension

       \ingroup reduce_func_max_by_key
    */
    AFAPI void maxByKey(array &keys_out, array &vals_out,
                        const array &keys, const array &vals,
                        const int dim = -1);
#endif

#if AF_API_VERSION >= 38
    /**
       C++ Interface to return the ragged maximum along a given dimension.

       Input parameter `ragged_len` sets the number of elements to consider.

       NaN values are ignored.

       \param[out] val        ragged maximum
       \param[out] idx        locations of the maximum ragged values
       \param[in]  in         input array
       \param[in]  ragged_len array containing the number of elements to use
       \param[in]  dim        dimension along which the maximum is found

       \ingroup reduce_func_max
    */
    AFAPI void max(array &val, array &idx, const array &in, const array &ragged_len, const int dim);
#endif

    /**
       C++ Interface to check if all values along a given dimension are true.

       NaN values are ignored.

       \param[in] in  input array
       \param[in] dim dimension along which the check occurs, -1 denotes the
                      first non-singleton dimension
       \return        array containing 1's if all true; 0's otherwise

       \ingroup reduce_func_all_true
    */
    AFAPI array allTrue(const array &in, const int dim = -1);

#if AF_API_VERSION >= 37
    /**
       C++ Interface to check if all values along a given dimension are true,
       according to an array of keys.

       NaN values are ignored.

       \param[out] keys_out reduced keys
       \param[out] vals_out array containing 1's if all true; 0's otherwise
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the check occurs

       \ingroup reduce_func_alltrue_by_key
    */
    AFAPI void allTrueByKey(array &keys_out, array &vals_out,
                            const array &keys, const array &vals,
                            const int dim = -1);
#endif

    /**
       C++ Interface to check if any values along a given dimension are true.

       NaN values are ignored.

       \param[in] in  input array
       \param[in] dim dimension along which the check occurs, -1 denotes the
                      first non-singleton dimension
       \return        array containing 1's if any true; 0's otherwise

       \ingroup reduce_func_any_true
    */
    AFAPI array anyTrue(const array &in, const int dim = -1);

#if AF_API_VERSION >= 37
    /**
       C++ Interface to check if any values along a given dimension are true,
       according to an array of keys.

       NaN values are ignored.

       \param[out] keys_out reduced keys
       \param[out] vals_out array containing 1's if any true; 0's otherwise
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the check occurs

       \ingroup reduce_func_anytrue_by_key
    */
    AFAPI void anyTrueByKey(array &keys_out, array &vals_out,
                            const array &keys, const array &vals,
                            const int dim = -1);
#endif

    /**
       C++ Interface to count non-zero values in an array along a given
       dimension.

       NaN values are treated as non-zero.

       \param[in] in  input array
       \param[in] dim dimension along which the count occurs, -1 denotes the
                      first non-singleton dimension
       \return        count

       \ingroup reduce_func_count
    */
    AFAPI array count(const array &in, const int dim = -1);

#if AF_API_VERSION >= 37
    /**
       C++ Interface to count non-zero values in an array, according to an
       array of keys.

       NaN values are treated as non-zero.

       \param[out] keys_out reduced keys
       \param[out] vals_out count
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the count occurs, -1 denotes
                            the first non-singleton dimension

       \ingroup reduce_func_count_by_key
    */
    AFAPI void countByKey(array &keys_out, array &vals_out,
                          const array &keys, const array &vals,
                          const int dim = -1);
#endif

    /**
       C++ Interface to sum array elements over all dimensions.

       Results in a single value as an output, which may be a single element
       `af::array`.

       \param[in] in  input array
       \return        sum

       \ingroup reduce_func_sum
    */
    template<typename T> T sum(const array &in);

#if AF_API_VERSION >= 31
    /**
       C++ Interface to sum array elements over all dimensions, replacing any
       NaNs with a specified value.

       Results in a single value as an output, which may be a single element
       `af::array`.

       \param[in] in     input array
       \param[in] nanval value that replaces NaNs
       \return           sum

       \ingroup reduce_func_sum
    */
    template<typename T> T sum(const array &in, double nanval);
#endif

    /**
       C++ Interface to multiply array elements over the first non-singleton
       dimension.

       \param[in] in input array
       \return       product

       \ingroup reduce_func_product
    */
    template<typename T> T product(const array &in);

#if AF_API_VERSION >= 31
    /**
       C++ Interface to multiply array elements over the first non-singleton
       dimension, replacing any NaNs with a specified value.

       \param[in] in     input array
       \param[in] nanval value that replaces NaNs
       \return           product

       \ingroup reduce_func_product
    */
    template<typename T> T product(const array &in, double nanval);
#endif

    /**
       C++ Interface to return the minimum along the first non-singleton
       dimension.

       NaN values are ignored.

       \param[in] in input array
       \return       minimum

       \ingroup reduce_func_min
    */
    template<typename T> T min(const array &in);

    /**
       C++ Interface to return the maximum along the first non-singleton
       dimension.

       NaN values are ignored.

       \param[in] in input array
       \return       maximum

       \ingroup reduce_func_max
    */
    template<typename T> T max(const array &in);

    /**
       C++ Interface to check if all values along the first non-singleton
       dimension are true.

       NaN values are ignored.

       \param[in] in input array
       \return       array containing 1's if all true; 0's otherwise

       \ingroup reduce_func_all_true
    */
    template<typename T> T allTrue(const array &in);

    /**
       C++ Interface to check if any values along the first non-singleton
       dimension are true.

       NaN values are ignored.

       \param[in] in input array
       \return       array containing 1's if any true; 0's otherwise

       \ingroup reduce_func_any_true
    */
    template<typename T> T anyTrue(const array &in);

    /**
       C++ Interface to count non-zero values along the first non-singleton
       dimension.

       NaN values are treated as non-zero.

       \param[in] in input array
       \return       count

       \ingroup reduce_func_count
    */
    template<typename T> T count(const array &in);

    /**
       C++ Interface to return the minimum and its location along a given
       dimension.

       NaN values are ignored.

       \param[out] val minimum
       \param[out] idx location
       \param[in]  in  input array
       \param[in]  dim dimension along which the minimum is found, -1 denotes
                       the first non-singleton dimension

       \ingroup reduce_func_min
    */
    AFAPI void min(array &val, array &idx, const array &in, const int dim = -1);

    /**
       C++ Interface to return the maximum and its location along a given
       dimension.

       NaN values are ignored.

       \param[out] val maximum
       \param[out] idx location
       \param[in]  in  input array
       \param[in]  dim dimension along which the maximum is found, -1 denotes
                       the first non-singleton dimension

       \ingroup reduce_func_max
    */
    AFAPI void max(array &val, array &idx, const array &in, const int dim = -1);

    /**
       C++ Interface to return the minimum and its location over all
       dimensions.

       NaN values are ignored.

       Often used to return values directly to the host.

       \param[out] val minimum
       \param[out] idx location
       \param[in]  in  input array

       \ingroup reduce_func_min
    */
    template<typename T> void min(T *val, unsigned *idx, const array &in);

    /**
       C++ Interface to return the maximum and its location over all
       dimensions.

       NaN values are ignored.

       Often used to return values directly to the host.

       \param[out] val maximum
       \param[out] idx location
       \param[in]  in  input array

       \ingroup reduce_func_max
    */
    template<typename T> void max(T *val, unsigned *idx, const array &in);

    /**
       C++ Interface to evaluate the cumulative sum (inclusive) along a given
       dimension.

       \param[in] in  input array
       \param[in] dim dimension along which the sum is accumulated, 0 denotes
                      the first non-singleton dimension
       \return        cumulative sum

       \ingroup scan_func_accum
    */
    AFAPI array accum(const array &in, const int dim = 0);

#if AF_API_VERSION >=34
    /**
       C++ Interface to scan an array (generalized) over a given dimension.

       \param[in] in             input array
       \param[in] dim            dimension along which the scan occurs, 0
                                 denotes the first non-singleton dimension
       \param[in] op             type of binary operation used
       \param[in] inclusive_scan flag specifying whether the scan is inclusive
       \return                   scan

       \ingroup scan_func_scan
    */
    AFAPI array scan(const array &in, const int dim = 0,
                     binaryOp op = AF_BINARY_ADD, bool inclusive_scan = true);

    /**
       C++ Interface to scan an array (generalized) over a given dimension,
       according to an array of keys.

       \param[in] key            keys array
       \param[in] in             input array
       \param[in] dim            dimension along which the scan occurs, 0
                                 denotes the first non-singleton dimension
       \param[in] op             type of binary operation used
       \param[in] inclusive_scan flag specifying whether the scan is inclusive
       \return                   scan

       \ingroup scan_func_scanbykey
    */
    AFAPI array scanByKey(const array &key, const array& in, const int dim = 0,
                          binaryOp op = AF_BINARY_ADD, bool inclusive_scan = true);
#endif

    /**
       C++ Interface to locate the indices of the non-zero values in an array.

       \param[in] in input array
       \return       linear indices where `in` is non-zero

       \ingroup scan_func_where
    */
    AFAPI array where(const array &in);

    /**
       C++ Interface to calculate the first order difference in an array over a
       given dimension.

       \param[in] in  input array
       \param[in] dim dimension along which the difference occurs, 0
                      denotes the first non-singleton dimension
       \return        first order numerical difference

       \ingroup calc_func_diff1
    */
    AFAPI array diff1(const array &in, const int dim = 0);

    /**
       C++ Interface to calculate the second order difference in an array over
       a given dimension.

       \param[in] in  input array
       \param[in] dim dimension along which the difference occurs, 0
                      denotes the first non-singleton dimension
       \return        second order numerical difference

       \ingroup calc_func_diff2
    */
    AFAPI array diff2(const array &in, const int dim = 0);

    /**
       C++ Interface to sort an array over a given dimension.

       \param[in] in          input array
       \param[in] dim         dimension along which the sort occurs, 0 denotes
                              the first non-singleton dimension
       \param[in] isAscending specifies the sorting order
       \return                sorted output

       \ingroup sort_func_sort
    */
    AFAPI array sort(const array &in, const unsigned dim = 0,
                     const bool isAscending = true);

    /**
       C++ Interface to sort an array over a given dimension and to return the
       original indices.

       \param[out] out         sorted output
       \param[out] indices     indices from the input
       \param[in]  in          input array
       \param[in]  dim         dimension along which the sort occurs, 0 denotes
                               the first non-singleton dimension
       \param[in]  isAscending specifies the sorting order

       \ingroup sort_func_sort_index
    */
    AFAPI void  sort(array &out, array &indices, const array &in, const unsigned dim = 0,
                     const bool isAscending = true);

    /**
       C++ Interface to sort an array over a given dimension, according to an
       array of keys.

       \param[out] out_keys    sorted keys
       \param[out] out_values  sorted output
       \param[in]  keys        keys array
       \param[in]  values      input array
       \param[in]  dim         dimension along which the sort occurs, 0 denotes
                               the first non-singleton dimension
       \param[in]  isAscending specifies the sorting order

       \ingroup sort_func_sort_keys
    */
    AFAPI void  sort(array &out_keys, array &out_values, const array &keys,
                     const array &values, const unsigned dim = 0,
                     const bool isAscending = true);

    /**
       C++ Interface to return the unique values in an array.

       \param[in] in        input array
       \param[in] is_sorted if true, skip the sorting steps internally
       \return              unique values

       \ingroup set_func_unique
    */
    AFAPI array setUnique(const array &in, const bool is_sorted=false);

    /**
       C++ Interface to evaluate the union of two arrays.

       \param[in] first     input array
       \param[in] second    input array
       \param[in] is_unique if true, skip calling setUnique internally
       \return              union, values in increasing order

       \ingroup set_func_union
    */
    AFAPI array setUnion(const array &first, const array &second,
                         const bool is_unique=false);

    /**
       C++ Interface to evaluate the intersection of two arrays.

       \param[in] first     input array
       \param[in] second    input array
       \param[in] is_unique if true, skip calling setUnique internally
       \return              intersection, values in increasing order

       \ingroup set_func_intersect
    */
    AFAPI array setIntersect(const array &first, const array &second,
                             const bool is_unique=false);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       C Interface to sum array elements over a given dimension.

       \param[out] out sum
       \param[in]  in  input array
       \param[in]  dim dimension along which the summation occurs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_sum
    */
    AFAPI af_err af_sum(af_array *out, const af_array in, const int dim);

#if AF_API_VERSION >= 39
    /**
       C Interface to sum array elements over all dimensions.

       Results in a single element `af::array`.

       \param[out] out sum
       \param[in]  in  input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_sum
    */
    AFAPI af_err af_sum_all_array(af_array *out, const af_array in);
#endif

#if AF_API_VERSION >= 31
    /**
       C Interface to sum array elements over a given dimension, replacing any
       NaNs with a specified value.

       \param[out] out    sum
       \param[in]  in     input array
       \param[in]  dim    dimension along which the summation occurs
       \param[in]  nanval value that replaces NaNs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_sum
    */
    AFAPI af_err af_sum_nan(af_array *out, const af_array in,
                            const int dim, const double nanval);
#endif

#if AF_API_VERSION >= 39
    /**
       C Interface to sum array elements over all dimensions, replacing any
       NaNs with a specified value.

       Results in a single element `af::array`.

       \param[out] out    sum
       \param[in]  in     input array
       \param[in]  nanval value that replaces NaNs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_sum
    */
    AFAPI af_err af_sum_nan_all_array(af_array *out, const af_array in, const double nanval);
#endif

#if AF_API_VERSION >= 37
    /**
       C Interface to sum array elements over a given dimension, according to
       an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out sum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the summation occurs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_sum_by_key
    */
    AFAPI af_err af_sum_by_key(af_array *keys_out, af_array *vals_out,
                               const af_array keys, const af_array vals, const int dim);

    /**
       C Interface to sum array elements over a given dimension, replacing any
       NaNs with a specified value, according to an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out sum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the summation occurs
       \param[in]  nanval   value that replaces NaNs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_sum_by_key
    */
    AFAPI af_err af_sum_by_key_nan(af_array *keys_out, af_array *vals_out,
                                   const af_array keys, const af_array vals,
                                   const int dim, const double nanval);
#endif

    /**
       C Interface to multiply array elements over a given dimension.

       \param[out] out product
       \param[in]  in  input array
       \param[in]  dim dimension along which the product occurs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_product
    */
    AFAPI af_err af_product(af_array *out, const af_array in, const int dim);

#if AF_API_VERSION >= 39
    /**
       C Interface to multiply array elements over all dimensions.

       Results in a single element `af::array`.

       \param[out] out product
       \param[in]  in  input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_product
    */
    AFAPI af_err af_product_all_array(af_array *out, const af_array in);
#endif

#if AF_API_VERSION >= 31
    /**
       C Interface to multiply array elements over a given dimension, replacing
       any NaNs with a specified value.

       \param[out] out    product
       \param[in]  in     input array
       \param[in]  dim    dimension along with the product occurs
       \param[in]  nanval value that replaces NaNs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_product
    */
    AFAPI af_err af_product_nan(af_array *out, const af_array in, const int dim, const double nanval);
#endif

#if AF_API_VERSION >= 39
    /**
       C Interface to multiply array elements over all dimensions, replacing
       any NaNs with a specified value.

       \param[out] out    product
       \param[in]  in     input array
       \param[in]  nanval value that replaces NaNs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_product
    */
    AFAPI af_err af_product_nan_all_array(af_array *out, const af_array in, const double nanval);
#endif

#if AF_API_VERSION >= 37
    /**
       C Interface to multiply array elements over a given dimension, according
       to an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out product
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the product occurs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_product_by_key
    */
    AFAPI af_err af_product_by_key(af_array *keys_out, af_array *vals_out,
                                   const af_array keys, const af_array vals, const int dim);

    /**
       C Interface to multiply array elements over a given dimension, replacing
       any NaNs with a specified value, according to an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out product
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the product occurs
       \param[in]  nanval   value that replaces NaNs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_product_by_key
    */
    AFAPI af_err af_product_by_key_nan(af_array *keys_out, af_array *vals_out,
                                       const af_array keys, const af_array vals,
                                       const int dim, const double nanval);
#endif

    /**
       C Interface to return the minimum along a given dimension.

       \param[out] out minimum
       \param[in]  in  input array
       \param[in]  dim dimension along which the minimum is found
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_min
    */
    AFAPI af_err af_min(af_array *out, const af_array in, const int dim);

#if AF_API_VERSION >= 37
    /**
       C Interface to return the minimum along a given dimension, according to
       an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out minimum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the minimum is found
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_min_by_key
    */
    AFAPI af_err af_min_by_key(af_array *keys_out, af_array *vals_out,
                               const af_array keys, const af_array vals,
                               const int dim);
#endif

    /**
       C Interface to return the maximum along a given dimension.

       \param[out] out  maximum
       \param[in]  in   input array
       \param[in]  dim dimension along which the maximum is found
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_max
    */
    AFAPI af_err af_max(af_array *out, const af_array in, const int dim);

#if AF_API_VERSION >= 37
    /**
       C Interface to return the maximum along a given dimension, according to
       an array of keys.

       \param[out] keys_out reduced keys
       \param[out] vals_out maximum
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the maximum is found
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_max_by_key
    */
    AFAPI af_err af_max_by_key(af_array *keys_out, af_array *vals_out,
                               const af_array keys, const af_array vals,
                               const int dim);
#endif

#if AF_API_VERSION >= 38
    /**
       C Interface to return the ragged maximum over a given dimension.

       Input parameter `ragged_len` sets the number of elements to consider.

       NaN values are ignored.

       \param[out] val        ragged maximum
       \param[out] idx        locations of the maximum ragged values
       \param[in]  in         input array
       \param[in]  ragged_len array containing the number of elements to use
       \param[in]  dim        dimension along which the maximum is found
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_max
    */
    AFAPI af_err af_max_ragged(af_array *val, af_array *idx, const af_array in, const af_array ragged_len, const int dim);
#endif

    /**
       C Interface  to check if all values along a given dimension are true.

       NaN values are ignored.

       \param[out] out array containing 1's if all true; 0's otherwise
       \param[in]  in  input array
       \param[in]  dim dimention along which the check occurs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_all_true
    */
    AFAPI af_err af_all_true(af_array *out, const af_array in, const int dim);

#if AF_API_VERSION >= 37
    /**
       C Interface to check if all values along a given dimension are true,
       according to an array of keys.

       NaN values are ignored.

       \param[out] keys_out reduced keys
       \param[out] vals_out array containing 1's if all true; 0's otherwise
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the check occurs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_alltrue_by_key
    */
    AFAPI af_err af_all_true_by_key(af_array *keys_out, af_array *vals_out,
                                    const af_array keys, const af_array vals,
                                    const int dim);
#endif

    /**
       C Interface to check if any values along a given dimension are true.

       NaN values are ignored.

       \param[out] out array containing 1's if any true; 0's otherwise
       \param[in]  in  input array
       \param[in]  dim dimension along which the check occurs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_any_true
    */
    AFAPI af_err af_any_true(af_array *out, const af_array in, const int dim);

#if AF_API_VERSION >= 37
    /**
       C Interface to check if any values along a given dimension are true.

       NaN values are ignored.

       \param[out] keys_out reduced keys
       \param[out] vals_out array containing 1's if any true; 0's otherwise
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimensions along which the check occurs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_anytrue_by_key
    */
    AFAPI af_err af_any_true_by_key(af_array *keys_out, af_array *vals_out,
                                    const af_array keys, const af_array vals,
                                    const int dim);
#endif

    /**
       C Interface to count non-zero values in an array along a given
       dimension.

       NaN values are treated as non-zero.

       \param[out] out count
       \param[in]  in  input array
       \param[in]  dim dimension along which the count occurs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_count
    */
    AFAPI af_err af_count(af_array *out, const af_array in, const int dim);

#if AF_API_VERSION >= 37
    /**
       C Interface to count non-zero values in an array, according to an array
       of keys.

       NaN values are treated as non-zero.

       \param[out] keys_out reduced keys
       \param[out] vals_out count
       \param[in]  keys     keys array
       \param[in]  vals     input array
       \param[in]  dim      dimension along which the count occurs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_count_by_key
    */
    AFAPI af_err af_count_by_key(af_array *keys_out, af_array *vals_out,
                                 const af_array keys, const af_array vals,
                                 const int dim);
#endif

    /**
       C Interface to sum array elements over all dimensions.

       If `in` is real, `imag` will be set to zeros.

       \param[out] real sum of all real components
       \param[out] imag sum of all imaginary components
       \param[in]  in   input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_sum
    */
    AFAPI af_err af_sum_all(double *real, double *imag, const af_array in);

#if AF_API_VERSION >= 31
    /**
       C Interface to sum array elements over all dimensions, replacing any
       NaNs with a specified value.

       If `in` is real, `imag` will be set to zeros.

       \param[out] real   sum of all real components
       \param[out] imag   sum of all imaginary components
       \param[in]  in     input array
       \param[in]  nanval value that replaces NaNs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_sum
    */
    AFAPI af_err af_sum_nan_all(double *real, double *imag,
                                const af_array in, const double nanval);
#endif

    /**
       C Interface to multiply array elements over all dimensions.

       If `in` is real, `imag` will be set to zeros.

       \param[out] real product of all real components
       \param[out] imag product of all imaginary components
       \param[in]  in   input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_product
    */
    AFAPI af_err af_product_all(double *real, double *imag, const af_array in);

#if AF_API_VERSION >= 31
    /**
       C Interface to multiply array elements over all dimensions, replacing
       any NaNs with a specified value.

       If `in` is real, `imag` will be set to zeros.

       \param[out] real   product of all real components
       \param[out] imag   product of all imaginary components
       \param[in]  in     input array
       \param[in]  nanval value that replaces NaNs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_product
    */
    AFAPI af_err af_product_nan_all(double *real, double *imag,
                                    const af_array in, const double nanval);
#endif

    /**
       C Interface to return the minimum over all dimensions.

       If `in` is real, `imag` will be set to zeros.

       \param[out] real real component of the minimum
       \param[out] imag imaginary component of the minimum
       \param[in]  in   input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_min
    */
    AFAPI af_err af_min_all(double *real, double *imag, const af_array in);

#if AF_API_VERSION >= 39
    /**
       C Interface to return the minimum over all dimensions.

       \param[out] out minimum
       \param[in]  in  input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_min
    */
    AFAPI af_err af_min_all_array(af_array *out, const af_array in);
#endif

    /**
       C Interface to return the maximum over all dimensions.

       If `in` is real, `imag` will be set to zeros.

       \param[out] real real component of the maximum
       \param[out] imag imaginary component of the maximum
       \param[in]  in   input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_max
    */
    AFAPI af_err af_max_all(double *real, double *imag, const af_array in);

#if AF_API_VERSION >= 39
    /**
       C Interface to return the maximum over all dimensions.

       \param[out] out maximum
       \param[in]  in  input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_max
    */
    AFAPI af_err af_max_all_array(af_array *out, const af_array in);
#endif

    /**
       C Interface to check if all values over all dimensions are true.
 
       \param[out] real 1 if all true; 0 otherwise
       \param[out] imag 0
       \param[in]  in   input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_all_true
    */
    AFAPI af_err af_all_true_all(double *real, double *imag, const af_array in);

#if AF_API_VERSION >= 39
    /**
       C Interface to check if all values over all dimensions are true.
 
       \param[out] out 1 if all true; 0 otherwise
       \param[in]  in  input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_all_true
    */
    AFAPI af_err af_all_true_all_array(af_array *out, const af_array in);
#endif

    /**
       C Interface to check if any values over all dimensions are true.

       \param[out] real 1 if any true; 0 otherwise
       \param[out] imag 0
       \param[in]  in   input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_any_true
    */
    AFAPI af_err af_any_true_all(double *real, double *imag, const af_array in);

#if AF_API_VERSION >= 39
    /**
       C Interface to check if any values over all dimensions are true.

       \param[out] out 1 if any true; 0 otherwise
       \param[in]  in  input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_any_true
    */
    AFAPI af_err af_any_true_all_array(af_array *out, const af_array in);
#endif

    /**
       C Interface to count non-zero values over all dimensions.

       \param[out] real count
       \param[out] imag 0
       \param[in]  in   input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_count
    */
    AFAPI af_err af_count_all(double *real, double *imag, const af_array in);

#if AF_API_VERSION >= 39
    /**
       C Interface to count non-zero values over all dimensions.

       \param[out] out count
       \param[in]  in  input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_count
    */
    AFAPI af_err af_count_all_array(af_array *out, const af_array in);
#endif

    /**
       C Interface to return the minimum and its location along a given
       dimension.

       \param[out] out minimum
       \param[out] idx location
       \param[in]  in  input array
       \param[in]  dim dimension along which the minimum is found
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_min
    */
    AFAPI af_err af_imin(af_array *out, af_array *idx, const af_array in,
                         const int dim);

    /**
       C Interface to return the maximum and its location along a given
       dimension.

       \param[out] out maximum
       \param[out] idx location
       \param[in]  in  input array
       \param[in]  dim dimension along which the maximum is found
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_max
    */
    AFAPI af_err af_imax(af_array *out, af_array *idx, const af_array in,
                         const int dim);

    /**
       C Interface to return the minimum and its location over all dimensions.

       NaN values are ignored.

       \param[out] real real component of the minimum
       \param[out] imag imaginary component of the minimum; 0 if `idx` is real
       \param[out] idx  location
       \param[in]  in   input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_min
    */
    AFAPI af_err af_imin_all(double *real, double *imag, unsigned *idx,
                             const af_array in);

    /**
       C Interface to return the maximum and its location over all dimensions.

       NaN values are ignored.

       \param[out] real real component of the maximum
       \param[out] imag imaginary component of the maximum; 0 if `idx` is real
       \param[out] idx  location
       \param[in]  in   input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup reduce_func_max
    */
    AFAPI af_err af_imax_all(double *real, double *imag, unsigned *idx, const af_array in);

    /**
       C Interface to evaluate the cumulative sum (inclusive) along a given
       dimension.

       \param[out] out cumulative sum
       \param[in]  in  input array
       \param[in]  dim dimension along which the sum is accumulated
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup scan_func_accum
    */
    AFAPI af_err af_accum(af_array *out, const af_array in, const int dim);

#if AF_API_VERSION >=34
    /**
       C Interface to scan an array (generalized) over a given dimension.

       \param[out] out            scan
       \param[in]  in             input array
       \param[in]  dim            dimension along which the scan occurs
       \param[in]  op             type of binary operation used
       \param[in]  inclusive_scan flag specifying whether the scan is inclusive
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup scan_func_scan
    */
    AFAPI af_err af_scan(af_array *out, const af_array in, const int dim,
                         af_binary_op op, bool inclusive_scan);

    /**
       C Interface to scan an array (generalized) over a given dimension,
       according to an array of keys.

       \param[out] out            scan
       \param[in]  key            keys array
       \param[in]  in             input array
       \param[in]  dim            dimension along which the scan occurs
       \param[in]  op             type of binary operation used
       \param[in]  inclusive_scan flag specifying whether the scan is inclusive
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup scan_func_scanbykey
    */
    AFAPI af_err af_scan_by_key(af_array *out, const af_array key,
                                const af_array in, const int dim,
                                af_binary_op op, bool inclusive_scan);

#endif

    /**
       C Interface to locate the indices of the non-zero values in an array.

       \param[out] idx linear indices where `in` is non-zero
       \param[in]  in  input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup scan_func_where
    */
    AFAPI af_err af_where(af_array *idx, const af_array in);

    /**
       C Interface to calculate the first order difference in an array over a
       given dimension.

       \param[out] out first order numerical difference
       \param[in]  in  input array
       \param[in]  dim dimension along which the difference occurs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup calc_func_diff1
    */
    AFAPI af_err af_diff1(af_array *out, const af_array in, const int dim);

    /**
       C Interface to calculate the second order difference in an array over a
       given dimension.

       \param[out] out second order numerical difference
       \param[in]  in  input array
       \param[in]  dim dimension along which the difference occurs
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup calc_func_diff2
    */
    AFAPI af_err af_diff2(af_array *out, const af_array in, const int dim);

    /**
       C Interface to sort an array over a given dimension.

       \param[out] out         sorted output
       \param[in]  in          input array
       \param[in]  dim         dimension along which the sort occurs
       \param[in]  isAscending specifies the sorting order
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup sort_func_sort
    */
    AFAPI af_err af_sort(af_array *out, const af_array in, const unsigned dim,
                         const bool isAscending);

    /**
       C Interface to sort an array over a given dimension and to return the
       original indices.

       \param[out] out         sorted output
       \param[out] indices     indices from the input
       \param[in]  in          input array
       \param[in]  dim         dimension along which the sort occurs
       \param[in]  isAscending specifies the sorting order
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup sort_func_sort_index
    */
    AFAPI af_err af_sort_index(af_array *out, af_array *indices, const af_array in,
                               const unsigned dim, const bool isAscending);
    /**
       C Interface to sort an array over a given dimension, according to an
       array of keys.

       \param[out] out_keys    sorted keys
       \param[out] out_values  sorted output
       \param[in]  keys        keys array
       \param[in]  values      input array
       \param[in]  dim         dimension along which the sort occurs
       \param[in]  isAscending specifies the sorting order
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup sort_func_sort_keys
    */
    AFAPI af_err af_sort_by_key(af_array *out_keys, af_array *out_values,
                                const af_array keys, const af_array values,
                                const unsigned dim, const bool isAscending);

    /**
       C Interface to return the unique values in an array.

       \param[out] out       unique values
       \param[in]  in        input array
       \param[in]  is_sorted if true, skip the sorting steps internally
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup set_func_unique
    */
    AFAPI af_err af_set_unique(af_array *out, const af_array in, const bool is_sorted);

    /**
       C Interface to evaluate the union of two arrays.

       \param[out] out       union, values in increasing order
       \param[in]  first     input array
       \param[in]  second    input array
       \param[in]  is_unique if true, skip calling unique internally
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup set_func_union
    */
    AFAPI af_err af_set_union(af_array *out, const af_array first,
                              const af_array second, const bool is_unique);

    /**
       C Interface to evaluate the intersection of two arrays.

       \param[out] out       intersection, values in increasing order
       \param[in]  first     input array
       \param[in]  second    input array
       \param[in]  is_unique if true, skip calling unique internally
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup set_func_intersect
    */
    AFAPI af_err af_set_intersect(af_array *out, const af_array first,
                                  const af_array second, const bool is_unique);

#ifdef __cplusplus
}
#endif

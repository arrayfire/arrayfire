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

    // Add all the elements along a dimension
    AFAPI array sum(const array &in, const int dim = 0);

    // Add all the elements along a dimension
    AFAPI array product(const array &in, const int dim = 0);

    // Get the minimum of all elements along a dimension
    AFAPI array min(const array &in, const int dim = 0);

    // Get the maximum of all elements along a dimension
    AFAPI array max(const array &in, const int dim = 0);

    // Check if all elements along a dimension are true
    AFAPI array alltrue(const array &in, const int dim = 0);

    // Check if any elements along a dimension are true
    AFAPI array anytrue(const array &in, const int dim = 0);

    // Count number of non zero elements along a dimension
    AFAPI array count(const array &in, const int dim = 0);

    // Add all the elements in an array
    template<typename T> T sum(const array &in);

    // Add all the elements in an array
    template<typename T> T product(const array &in);

    // Get the minimum of all elements in an array
    template<typename T> T min(const array &in);

    // Get the maximum of all elements in an array
    template<typename T> T max(const array &in);

    // Check if all elements in an array are true
    template<typename T> T alltrue(const array &in);

    // Check if any elements in an array are true
    template<typename T> T anytrue(const array &in);

    // Count number of non zero elements in an array
    template<typename T> T count(const array &in);

    // Get the minimum of all elements along a dimension
    AFAPI void min(array &val, array &idx, const array &in, const int dim = 0);

    // Get the maximum of all elements along a dimension
    AFAPI void max(array &val, array &idx, const array &in, const int dim = 0);

    // Get the minimum of all elements in an array
    template<typename T> void min(T *val, unsigned *idx, const array &in);

    // Get the maximum of all elements in an array
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
    // Add all the elements along a dimension
    AFAPI af_err af_sum(af_array *out, const af_array in, const int dim);

    // multiply all the elements along a dimension
    AFAPI af_err af_product(af_array *out, const af_array in, const int dim);

    // Get the minimum of all elements along a dimension
    AFAPI af_err af_min(af_array *out, const af_array in, const int dim);

    // Get the maximum of all elements along a dimension
    AFAPI af_err af_max(af_array *out, const af_array in, const int dim);

    // Check if all elements along a dimension are true
    AFAPI af_err af_alltrue(af_array *out, const af_array in, const int dim);

    // Check if any elements along a dimension are true
    AFAPI af_err af_anytrue(af_array *out, const af_array in, const int dim);

    // Count number of non zero elements along a dimension
    AFAPI af_err af_count(af_array *out, const af_array in, const int dim);

    // Add all the elements
    AFAPI af_err af_sum_all(double *real, double *imag, const af_array in);

    // multiply all the elements
    AFAPI af_err af_product_all(double *real, double *imag, const af_array in);

    // Get the minimum of all elements
    AFAPI af_err af_min_all(double *real, double *imag, const af_array in);

    // Get the maximum of all elements
    AFAPI af_err af_max_all(double *real, double *imag, const af_array in);

    // Check if all elements are true
    AFAPI af_err af_alltrue_all(double *real, double *imag, const af_array in);

    // Check if any elements are true
    AFAPI af_err af_anytrue_all(double *real, double *imag, const af_array in);

    // Count number of non zero elements
    AFAPI af_err af_count_all(double *real, double *imag, const af_array in);

        // Get the minimum values and their indices along a dimension
    AFAPI af_err af_imin(af_array *out, af_array *idx, const af_array in, const int dim);

    // Get the maximum values and their indices along a dimension
    AFAPI af_err af_imax(af_array *out, af_array *idx, const af_array in, const int dim);

     // Get the minimum of all elements and its location
    AFAPI af_err af_imin_all(double *real, double *imag, unsigned *idx, const af_array in);

    // Get the maximum of all elements and its location
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

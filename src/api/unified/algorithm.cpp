/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/algorithm.h>
#include <af/array.h>
#include "symbol_manager.hpp"

#define ALGO_HAPI_DEF(af_func)                                        \
    af_err af_func(af_array *out, const af_array in, const int dim) { \
        CHECK_ARRAYS(in);                                             \
        CALL(af_func, out, in, dim);                                  \
    }

ALGO_HAPI_DEF(af_sum)
ALGO_HAPI_DEF(af_product)
ALGO_HAPI_DEF(af_min)
ALGO_HAPI_DEF(af_max)
ALGO_HAPI_DEF(af_all_true)
ALGO_HAPI_DEF(af_any_true)
ALGO_HAPI_DEF(af_count)
ALGO_HAPI_DEF(af_accum)
ALGO_HAPI_DEF(af_diff1)
ALGO_HAPI_DEF(af_diff2)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF_BYKEY(af_func)                                          \
    af_err af_func(af_array *keys_out, af_array *vals_out,                    \
                   const af_array keys, const af_array vals, const int dim) { \
        CHECK_ARRAYS(keys, vals);                                             \
        CALL(af_func, keys_out, vals_out, keys, vals, dim);                   \
    }

ALGO_HAPI_DEF_BYKEY(af_sum_by_key)
ALGO_HAPI_DEF_BYKEY(af_product_by_key)
ALGO_HAPI_DEF_BYKEY(af_min_by_key)
ALGO_HAPI_DEF_BYKEY(af_max_by_key)
ALGO_HAPI_DEF_BYKEY(af_all_true_by_key)
ALGO_HAPI_DEF_BYKEY(af_any_true_by_key)
ALGO_HAPI_DEF_BYKEY(af_count_by_key)

#undef ALGO_HAPI_DEF_BYKEY

#define ALGO_HAPI_DEF(af_func_nan)                                      \
    af_err af_func_nan(af_array *out, const af_array in, const int dim, \
                       const double nanval) {                           \
        CHECK_ARRAYS(in);                                               \
        CALL(af_func_nan, out, in, dim, nanval);                        \
    }

ALGO_HAPI_DEF(af_sum_nan)
ALGO_HAPI_DEF(af_product_nan)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF_BYKEY(af_func_nan)                                \
    af_err af_func_nan(af_array *keys_out, af_array *vals_out,          \
                       const af_array keys, const af_array vals,        \
                       const int dim, const double nanval) {            \
        CHECK_ARRAYS(keys, vals);                                       \
        CALL(af_func_nan, keys_out, vals_out, keys, vals, dim, nanval); \
    }

ALGO_HAPI_DEF_BYKEY(af_sum_by_key_nan)
ALGO_HAPI_DEF_BYKEY(af_product_by_key_nan)

#undef ALGO_HAPI_DEF_BYKEY

#define ALGO_HAPI_DEF(af_func_all)                                      \
    af_err af_func_all(double *real, double *imag, const af_array in) { \
        CHECK_ARRAYS(in);                                               \
        CALL(af_func_all, real, imag, in);                              \
    }

ALGO_HAPI_DEF(af_sum_all)
ALGO_HAPI_DEF(af_product_all)
ALGO_HAPI_DEF(af_min_all)
ALGO_HAPI_DEF(af_max_all)
ALGO_HAPI_DEF(af_all_true_all)
ALGO_HAPI_DEF(af_any_true_all)
ALGO_HAPI_DEF(af_count_all)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF(af_func_nan_all)                                    \
    af_err af_func_nan_all(double *real, double *imag, const af_array in, \
                           const double nanval) {                         \
        CHECK_ARRAYS(in);                                                 \
        CALL(af_func_nan_all, real, imag, in, nanval);                    \
    }

ALGO_HAPI_DEF(af_sum_nan_all)
ALGO_HAPI_DEF(af_product_nan_all)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF(af_ifunc)                                      \
    af_err af_ifunc(af_array *out, af_array *idx, const af_array in, \
                    const int dim) {                                 \
        CHECK_ARRAYS(in);                                            \
        CALL(af_ifunc, out, idx, in, dim);                           \
    }

ALGO_HAPI_DEF(af_imin)
ALGO_HAPI_DEF(af_imax)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF(af_ifunc_all)                                \
    af_err af_ifunc_all(double *real, double *imag, unsigned *idx, \
                        const af_array in) {                       \
        CHECK_ARRAYS(in);                                          \
        CALL(af_ifunc_all, real, imag, idx, in);                   \
    }

ALGO_HAPI_DEF(af_imin_all)
ALGO_HAPI_DEF(af_imax_all)

#undef ALGO_HAPI_DEF

af_err af_where(af_array *idx, const af_array in) {
    CHECK_ARRAYS(in);
    CALL(af_where, idx, in);
}

af_err af_scan(af_array *out, const af_array in, const int dim, af_binary_op op,
               bool inclusive_scan) {
    CHECK_ARRAYS(in);
    CALL(af_scan, out, in, dim, op, inclusive_scan);
}

af_err af_scan_by_key(af_array *out, const af_array key, const af_array in,
                      const int dim, af_binary_op op, bool inclusive_scan) {
    CHECK_ARRAYS(in, key);
    CALL(af_scan_by_key, out, key, in, dim, op, inclusive_scan);
}

af_err af_sort(af_array *out, const af_array in, const unsigned dim,
               const bool isAscending) {
    CHECK_ARRAYS(in);
    CALL(af_sort, out, in, dim, isAscending);
}

af_err af_sort_index(af_array *out, af_array *indices, const af_array in,
                     const unsigned dim, const bool isAscending) {
    CHECK_ARRAYS(in);
    CALL(af_sort_index, out, indices, in, dim, isAscending);
}

af_err af_sort_by_key(af_array *out_keys, af_array *out_values,
                      const af_array keys, const af_array values,
                      const unsigned dim, const bool isAscending) {
    CHECK_ARRAYS(keys, values);
    CALL(af_sort_by_key, out_keys, out_values, keys, values, dim, isAscending);
}

af_err af_set_unique(af_array *out, const af_array in, const bool is_sorted) {
    CHECK_ARRAYS(in);
    CALL(af_set_unique, out, in, is_sorted);
}

af_err af_set_union(af_array *out, const af_array first, const af_array second,
                    const bool is_unique) {
    CHECK_ARRAYS(first, second);
    CALL(af_set_union, out, first, second, is_unique);
}

af_err af_set_intersect(af_array *out, const af_array first,
                        const af_array second, const bool is_unique) {
    CHECK_ARRAYS(first, second);
    CALL(af_set_intersect, out, first, second, is_unique);
}

af_err af_max_ragged(af_array *vals, af_array *idx, const af_array in,
                     const af_array ragged_len, const int dim) {
    CHECK_ARRAYS(in, ragged_len);
    CALL(af_max_ragged, vals, idx, in, ragged_len, dim);
}

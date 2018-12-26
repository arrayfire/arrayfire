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
        return CALL(out, in, dim);                                    \
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

#define ALGO_HAPI_DEF(af_func_nan)                                      \
    af_err af_func_nan(af_array *out, const af_array in, const int dim, \
                       const double nanval) {                           \
        CHECK_ARRAYS(in);                                               \
        return CALL(out, in, dim, nanval);                              \
    }

ALGO_HAPI_DEF(af_sum_nan)
ALGO_HAPI_DEF(af_product_nan)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF(af_func_all)                                      \
    af_err af_func_all(double *real, double *imag, const af_array in) { \
        CHECK_ARRAYS(in);                                               \
        return CALL(real, imag, in);                                    \
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
        return CALL(real, imag, in, nanval);                              \
    }

ALGO_HAPI_DEF(af_sum_nan_all)
ALGO_HAPI_DEF(af_product_nan_all)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF(af_ifunc)                                      \
    af_err af_ifunc(af_array *out, af_array *idx, const af_array in, \
                    const int dim) {                                 \
        CHECK_ARRAYS(in);                                            \
        return CALL(out, idx, in, dim);                              \
    }

ALGO_HAPI_DEF(af_imin)
ALGO_HAPI_DEF(af_imax)

#undef ALGO_HAPI_DEF

#define ALGO_HAPI_DEF(af_ifunc_all)                                \
    af_err af_ifunc_all(double *real, double *imag, unsigned *idx, \
                        const af_array in) {                       \
        CHECK_ARRAYS(in);                                          \
        return CALL(real, imag, idx, in);                          \
    }

ALGO_HAPI_DEF(af_imin_all)
ALGO_HAPI_DEF(af_imax_all)

#undef ALGO_HAPI_DEF

af_err af_where(af_array *idx, const af_array in) {
    CHECK_ARRAYS(in);
    return CALL(idx, in);
}

af_err af_scan(af_array *out, const af_array in, const int dim, af_binary_op op,
               bool inclusive_scan) {
    CHECK_ARRAYS(in);
    return CALL(out, in, dim, op, inclusive_scan);
}

af_err af_scan_by_key(af_array *out, const af_array key, const af_array in,
                      const int dim, af_binary_op op, bool inclusive_scan) {
    CHECK_ARRAYS(in, key);
    return CALL(out, key, in, dim, op, inclusive_scan);
}

af_err af_sort(af_array *out, const af_array in, const unsigned dim,
               const bool isAscending) {
    CHECK_ARRAYS(in);
    return CALL(out, in, dim, isAscending);
}

af_err af_sort_index(af_array *out, af_array *indices, const af_array in,
                     const unsigned dim, const bool isAscending) {
    CHECK_ARRAYS(in);
    return CALL(out, indices, in, dim, isAscending);
}

af_err af_sort_by_key(af_array *out_keys, af_array *out_values,
                      const af_array keys, const af_array values,
                      const unsigned dim, const bool isAscending) {
    CHECK_ARRAYS(keys, values);
    return CALL(out_keys, out_values, keys, values, dim, isAscending);
}

af_err af_set_unique(af_array *out, const af_array in, const bool is_sorted) {
    CHECK_ARRAYS(in);
    return CALL(out, in, is_sorted);
}

af_err af_set_union(af_array *out, const af_array first, const af_array second,
                    const bool is_unique) {
    CHECK_ARRAYS(first, second);
    return CALL(out, first, second, is_unique);
}

af_err af_set_intersect(af_array *out, const af_array first,
                        const af_array second, const bool is_unique) {
    CHECK_ARRAYS(first, second);
    return CALL(out, first, second, is_unique);
}

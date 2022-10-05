/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_oneapi.hpp>
//#include <kernel/reduce.hpp>
//#include <kernel/reduce_by_key.hpp>
#include <reduce.hpp>
#include <af/dim4.hpp>
#include <complex>

using af::dim4;
using std::swap;
namespace oneapi {
template<af_op_t op, typename Ti, typename To>
Array<To> reduce(const Array<Ti> &in, const int dim, bool change_nan,
                 double nanval) {
    ONEAPI_NOT_SUPPORTED("");
    Array<To> out = createEmptyArray<To>(1);
    return out;
}

template<af_op_t op, typename Ti, typename Tk, typename To>
void reduce_by_key(Array<Tk> &keys_out, Array<To> &vals_out,
                   const Array<Tk> &keys, const Array<Ti> &vals, const int dim,
                   bool change_nan, double nanval) {
    ONEAPI_NOT_SUPPORTED("");
}

template<af_op_t op, typename Ti, typename To>
Array<To> reduce_all(const Array<Ti> &in, bool change_nan, double nanval) {
    ONEAPI_NOT_SUPPORTED("");
    Array<To> out = createEmptyArray<To>(1);
    return out;
}

}  // namespace oneapi

#define INSTANTIATE(Op, Ti, To)                                                \
    template Array<To> reduce<Op, Ti, To>(const Array<Ti> &in, const int dim,  \
                                          bool change_nan, double nanval);     \
    template void reduce_by_key<Op, Ti, int, To>(                              \
        Array<int> & keys_out, Array<To> & vals_out, const Array<int> &keys,   \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval); \
    template void reduce_by_key<Op, Ti, uint, To>(                             \
        Array<uint> & keys_out, Array<To> & vals_out, const Array<uint> &keys, \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval); \
    template Array<To> reduce_all<Op, Ti, To>(const Array<Ti> &in,             \
                                              bool change_nan, double nanval);

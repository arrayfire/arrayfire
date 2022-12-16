/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/Binary.hpp>
#include <common/Transform.hpp>
#include <common/half.hpp>
#include <kernel/reduce.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <reduce.hpp>
#include <af/dim4.hpp>

#include <complex>
#include <functional>

using af::dim4;
using arrayfire::common::Binary;
using arrayfire::common::half;
using arrayfire::common::Transform;
using arrayfire::cpu::cdouble;

namespace arrayfire {
namespace common {

template<>
struct Binary<cdouble, af_add_t> {
    static cdouble init() { return cdouble(0, 0); }

    cdouble operator()(cdouble lhs, cdouble rhs) {
        return cdouble(real(lhs) + real(rhs), imag(lhs) + imag(rhs));
    }
};

}  // namespace common
namespace cpu {

template<af_op_t op, typename Ti, typename To>
using reduce_dim_func = std::function<void(
    Param<To>, const dim_t, CParam<Ti>, const dim_t, const int, bool, double)>;

template<af_op_t op, typename Ti, typename To>
Array<To> reduce(const Array<Ti> &in, const int dim, bool change_nan,
                 double nanval) {
    dim4 odims = in.dims();
    odims[dim] = 1;

    Array<To> out = createEmptyArray<To>(odims);
    static const reduce_dim_func<op, Ti, To> reduce_funcs[4] = {
        kernel::reduce_dim<op, Ti, To, 1>(),
        kernel::reduce_dim<op, Ti, To, 2>(),
        kernel::reduce_dim<op, Ti, To, 3>(),
        kernel::reduce_dim<op, Ti, To, 4>()};

    getQueue().enqueue(reduce_funcs[in.ndims() - 1], out, 0, in, 0, dim,
                       change_nan, nanval);

    return out;
}

template<af_op_t op, typename Ti, typename Tk, typename To>
using reduce_dim_func_by_key =
    std::function<void(Param<To> ovals, const dim_t ovOffset, CParam<Tk> keys,
                       CParam<Ti> vals, const dim_t vOffset, int *n_reduced,
                       const int dim, bool change_nan, double nanval)>;

template<af_op_t op, typename Ti, typename Tk, typename To>
void reduce_by_key(Array<Tk> &keys_out, Array<To> &vals_out,
                   const Array<Tk> &keys, const Array<Ti> &vals, const int dim,
                   bool change_nan, double nanval) {
    dim4 okdims = keys.dims();
    dim4 ovdims = vals.dims();

    int n_reduced;
    Array<Tk> fullsz_okeys = createEmptyArray<Tk>(okdims);
    getQueue().enqueue(kernel::n_reduced_keys<Tk>, fullsz_okeys, &n_reduced,
                       keys);
    getQueue().sync();

    okdims[0]   = n_reduced;
    ovdims[dim] = n_reduced;

    std::vector<af_seq> index;
    for (int i = 0; i < keys.ndims(); ++i) {
        af_seq s = {0.0, static_cast<double>(okdims[i]) - 1, 1.0};
        index.push_back(s);
    }
    Array<Tk> okeys = createSubArray<Tk>(fullsz_okeys, index, true);
    Array<To> ovals = createEmptyArray<To>(ovdims);

    static const reduce_dim_func_by_key<op, Ti, Tk, To> reduce_funcs[4] = {
        kernel::reduce_dim_by_key<op, Ti, Tk, To, 1>(),
        kernel::reduce_dim_by_key<op, Ti, Tk, To, 2>(),
        kernel::reduce_dim_by_key<op, Ti, Tk, To, 3>(),
        kernel::reduce_dim_by_key<op, Ti, Tk, To, 4>()};

    getQueue().enqueue(reduce_funcs[vals.ndims() - 1], ovals, 0, keys, vals, 0,
                       &n_reduced, dim, change_nan, nanval);

    keys_out = okeys;
    vals_out = ovals;
}

template<af_op_t op, typename Ti, typename Taccumulate>
Taccumulate reduce_all(const Array<Ti> &in, bool change_nan, double nanval) {
    in.eval();
    getQueue().sync();

    Transform<Ti, compute_t<Taccumulate>, op> transform;
    Binary<compute_t<Taccumulate>, op> reduce;

    compute_t<Taccumulate> out = Binary<compute_t<Taccumulate>, op>::init();

    // Decrement dimension of select dimension
    af::dim4 dims           = in.dims();
    af::dim4 strides        = in.strides();
    const data_t<Ti> *inPtr = in.get();

    for (dim_t l = 0; l < dims[3]; l++) {
        dim_t off3 = l * strides[3];

        for (dim_t k = 0; k < dims[2]; k++) {
            dim_t off2 = k * strides[2];

            for (dim_t j = 0; j < dims[1]; j++) {
                dim_t off1 = j * strides[1];

                for (dim_t i = 0; i < dims[0]; i++) {
                    dim_t idx = i + off1 + off2 + off3;

                    compute_t<Taccumulate> in_val = transform(inPtr[idx]);
                    if (change_nan) {
                        in_val = IS_NAN(in_val) ? nanval : in_val;
                    }
                    out = reduce(in_val, out);
                }
            }
        }
    }

    return data_t<Taccumulate>(out);
}

#define INSTANTIATE(ROp, Ti, To)                                               \
    template Array<To> reduce<ROp, Ti, To>(const Array<Ti> &in, const int dim, \
                                           bool change_nan, double nanval);    \
    template To reduce_all<ROp, Ti, To>(const Array<Ti> &in, bool change_nan,  \
                                        double nanval);                        \
    template void reduce_by_key<ROp, Ti, int, To>(                             \
        Array<int> & keys_out, Array<To> & vals_out, const Array<int> &keys,   \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval); \
    template void reduce_by_key<ROp, Ti, uint, To>(                            \
        Array<uint> & keys_out, Array<To> & vals_out, const Array<uint> &keys, \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval);

// min
INSTANTIATE(af_min_t, float, float)
INSTANTIATE(af_min_t, double, double)
INSTANTIATE(af_min_t, cfloat, cfloat)
INSTANTIATE(af_min_t, cdouble, cdouble)
INSTANTIATE(af_min_t, int, int)
INSTANTIATE(af_min_t, uint, uint)
INSTANTIATE(af_min_t, intl, intl)
INSTANTIATE(af_min_t, uintl, uintl)
INSTANTIATE(af_min_t, char, char)
INSTANTIATE(af_min_t, uchar, uchar)
INSTANTIATE(af_min_t, short, short)
INSTANTIATE(af_min_t, ushort, ushort)
INSTANTIATE(af_min_t, half, half)

// max
INSTANTIATE(af_max_t, float, float)
INSTANTIATE(af_max_t, double, double)
INSTANTIATE(af_max_t, cfloat, cfloat)
INSTANTIATE(af_max_t, cdouble, cdouble)
INSTANTIATE(af_max_t, int, int)
INSTANTIATE(af_max_t, uint, uint)
INSTANTIATE(af_max_t, intl, intl)
INSTANTIATE(af_max_t, uintl, uintl)
INSTANTIATE(af_max_t, char, char)
INSTANTIATE(af_max_t, uchar, uchar)
INSTANTIATE(af_max_t, short, short)
INSTANTIATE(af_max_t, ushort, ushort)
INSTANTIATE(af_max_t, half, half)

// sum
INSTANTIATE(af_add_t, float, float)
INSTANTIATE(af_add_t, double, double)
INSTANTIATE(af_add_t, cfloat, cfloat)
INSTANTIATE(af_add_t, cdouble, cdouble)
INSTANTIATE(af_add_t, int, int)
INSTANTIATE(af_add_t, int, float)
INSTANTIATE(af_add_t, uint, uint)
INSTANTIATE(af_add_t, uint, float)
INSTANTIATE(af_add_t, intl, intl)
INSTANTIATE(af_add_t, intl, double)
INSTANTIATE(af_add_t, uintl, uintl)
INSTANTIATE(af_add_t, uintl, double)
INSTANTIATE(af_add_t, char, int)
INSTANTIATE(af_add_t, char, float)
INSTANTIATE(af_add_t, uchar, uint)
INSTANTIATE(af_add_t, uchar, float)
INSTANTIATE(af_add_t, short, int)
INSTANTIATE(af_add_t, short, float)
INSTANTIATE(af_add_t, ushort, uint)
INSTANTIATE(af_add_t, ushort, float)
INSTANTIATE(af_add_t, half, float)
INSTANTIATE(af_add_t, half, half)

// mul
INSTANTIATE(af_mul_t, float, float)
INSTANTIATE(af_mul_t, double, double)
INSTANTIATE(af_mul_t, cfloat, cfloat)
INSTANTIATE(af_mul_t, cdouble, cdouble)
INSTANTIATE(af_mul_t, int, int)
INSTANTIATE(af_mul_t, uint, uint)
INSTANTIATE(af_mul_t, intl, intl)
INSTANTIATE(af_mul_t, uintl, uintl)
INSTANTIATE(af_mul_t, char, int)
INSTANTIATE(af_mul_t, uchar, uint)
INSTANTIATE(af_mul_t, short, int)
INSTANTIATE(af_mul_t, ushort, uint)
INSTANTIATE(af_mul_t, half, float)

// count
INSTANTIATE(af_notzero_t, float, uint)
INSTANTIATE(af_notzero_t, double, uint)
INSTANTIATE(af_notzero_t, cfloat, uint)
INSTANTIATE(af_notzero_t, cdouble, uint)
INSTANTIATE(af_notzero_t, int, uint)
INSTANTIATE(af_notzero_t, uint, uint)
INSTANTIATE(af_notzero_t, intl, uint)
INSTANTIATE(af_notzero_t, uintl, uint)
INSTANTIATE(af_notzero_t, char, uint)
INSTANTIATE(af_notzero_t, uchar, uint)
INSTANTIATE(af_notzero_t, short, uint)
INSTANTIATE(af_notzero_t, ushort, uint)
INSTANTIATE(af_notzero_t, half, uint)

// anytrue
INSTANTIATE(af_or_t, float, char)
INSTANTIATE(af_or_t, double, char)
INSTANTIATE(af_or_t, cfloat, char)
INSTANTIATE(af_or_t, cdouble, char)
INSTANTIATE(af_or_t, int, char)
INSTANTIATE(af_or_t, uint, char)
INSTANTIATE(af_or_t, intl, char)
INSTANTIATE(af_or_t, uintl, char)
INSTANTIATE(af_or_t, char, char)
INSTANTIATE(af_or_t, uchar, char)
INSTANTIATE(af_or_t, short, char)
INSTANTIATE(af_or_t, ushort, char)
INSTANTIATE(af_or_t, half, char)

// alltrue
INSTANTIATE(af_and_t, float, char)
INSTANTIATE(af_and_t, double, char)
INSTANTIATE(af_and_t, cfloat, char)
INSTANTIATE(af_and_t, cdouble, char)
INSTANTIATE(af_and_t, int, char)
INSTANTIATE(af_and_t, uint, char)
INSTANTIATE(af_and_t, intl, char)
INSTANTIATE(af_and_t, uintl, char)
INSTANTIATE(af_and_t, char, char)
INSTANTIATE(af_and_t, uchar, char)
INSTANTIATE(af_and_t, short, char)
INSTANTIATE(af_and_t, ushort, char)
INSTANTIATE(af_and_t, half, char)

}  // namespace cpu
}  // namespace arrayfire

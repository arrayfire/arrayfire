/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <ireduce.hpp>
#include <kernel/ireduce.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

#include <complex>

using af::dim4;
using arrayfire::common::half;

namespace arrayfire {
namespace cpu {

template<af_op_t op, typename T>
using ireduce_dim_func =
    std::function<void(Param<T>, Param<uint>, const dim_t, CParam<T>,
                       const dim_t, const int, CParam<uint>)>;

template<af_op_t op, typename T>
void ireduce(Array<T> &out, Array<uint> &loc, const Array<T> &in,
             const int dim) {
    dim4 odims       = in.dims();
    odims[dim]       = 1;
    Array<uint> rlen = createEmptyArray<uint>(af::dim4(0));
    static const ireduce_dim_func<op, T> ireduce_funcs[] = {
        kernel::ireduce_dim<op, T, 1>(), kernel::ireduce_dim<op, T, 2>(),
        kernel::ireduce_dim<op, T, 3>(), kernel::ireduce_dim<op, T, 4>()};

    getQueue().enqueue(ireduce_funcs[in.ndims() - 1], out, loc, 0, in, 0, dim,
                       rlen);
}

template<af_op_t op, typename T>
void rreduce(Array<T> &out, Array<uint> &loc, const Array<T> &in, const int dim,
             const Array<uint> &rlen) {
    dim4 odims = in.dims();
    odims[dim] = 1;

    static const ireduce_dim_func<op, T> ireduce_funcs[] = {
        kernel::ireduce_dim<op, T, 1>(), kernel::ireduce_dim<op, T, 2>(),
        kernel::ireduce_dim<op, T, 3>(), kernel::ireduce_dim<op, T, 4>()};

    getQueue().enqueue(ireduce_funcs[in.ndims() - 1], out, loc, 0, in, 0, dim,
                       rlen);
}

template<af_op_t op, typename T>
T ireduce_all(unsigned *loc, const Array<T> &in) {
    getQueue().sync();

    af::dim4 dims    = in.dims();
    af::dim4 strides = in.strides();
    const T *inPtr   = in.get();

    kernel::MinMaxOp<op, T> Op(inPtr[0], 0);

    for (dim_t l = 0; l < dims[3]; l++) {
        dim_t off3 = l * strides[3];

        for (dim_t k = 0; k < dims[2]; k++) {
            dim_t off2 = k * strides[2];

            for (dim_t j = 0; j < dims[1]; j++) {
                dim_t off1 = j * strides[1];

                for (dim_t i = 0; i < dims[0]; i++) {
                    dim_t idx = i + off1 + off2 + off3;
                    Op(inPtr[idx], idx);
                }
            }
        }
    }

    *loc = Op.m_idx;
    return Op.m_val;
}

#define INSTANTIATE(ROp, T)                                           \
    template void ireduce<ROp, T>(Array<T> & out, Array<uint> & loc,  \
                                  const Array<T> &in, const int dim); \
    template void rreduce<ROp, T>(Array<T> & out, Array<uint> & loc,  \
                                  const Array<T> &in, const int dim,  \
                                  const Array<uint> &rlen);           \
    template T ireduce_all<ROp, T>(unsigned *loc, const Array<T> &in);

// min
INSTANTIATE(af_min_t, float)
INSTANTIATE(af_min_t, double)
INSTANTIATE(af_min_t, cfloat)
INSTANTIATE(af_min_t, cdouble)
INSTANTIATE(af_min_t, int)
INSTANTIATE(af_min_t, uint)
INSTANTIATE(af_min_t, intl)
INSTANTIATE(af_min_t, uintl)
INSTANTIATE(af_min_t, char)
INSTANTIATE(af_min_t, uchar)
INSTANTIATE(af_min_t, short)
INSTANTIATE(af_min_t, ushort)
INSTANTIATE(af_min_t, half)

// max
INSTANTIATE(af_max_t, float)
INSTANTIATE(af_max_t, double)
INSTANTIATE(af_max_t, cfloat)
INSTANTIATE(af_max_t, cdouble)
INSTANTIATE(af_max_t, int)
INSTANTIATE(af_max_t, uint)
INSTANTIATE(af_max_t, intl)
INSTANTIATE(af_max_t, uintl)
INSTANTIATE(af_max_t, char)
INSTANTIATE(af_max_t, uchar)
INSTANTIATE(af_max_t, short)
INSTANTIATE(af_max_t, ushort)
INSTANTIATE(af_max_t, half)

}  // namespace cpu
}  // namespace arrayfire

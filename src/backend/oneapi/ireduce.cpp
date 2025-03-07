/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <ireduce.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <err_oneapi.hpp>
#include <kernel/ireduce.hpp>
#include <optypes.hpp>
#include <af/dim4.hpp>
#include <complex>

using af::dim4;
using arrayfire::common::half;

namespace arrayfire {
namespace oneapi {

template<af_op_t op, typename T>
void ireduce(Array<T> &out, Array<uint> &loc, const Array<T> &in,
             const int dim) {
    Array<uint> rlen = createEmptyArray<uint>(af::dim4(0));
    kernel::ireduce<T, op>(out, loc, in, dim, rlen);
}

template<af_op_t op, typename T>
void rreduce(Array<T> &out, Array<uint> &loc, const Array<T> &in, const int dim,
             const Array<uint> &rlen) {
    kernel::ireduce<T, op>(out, loc, in, dim, rlen);
}

template<af_op_t op, typename T>
T ireduce_all(unsigned *loc, const Array<T> &in) {
    return kernel::ireduce_all<T, op>(loc, in);
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
}  // namespace oneapi
}  // namespace arrayfire

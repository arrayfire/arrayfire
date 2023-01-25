/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <cast.hpp>
#include <common/half.hpp>
#include <convolve.hpp>
#include <err_cuda.hpp>
#include <kernel/convolve.hpp>
#include <platform.hpp>
#include <af/dim4.hpp>
#include <type_traits>

using af::dim4;
using arrayfire::common::half;
using std::conditional;
using std::is_same;

namespace arrayfire {
namespace cuda {

template<typename T, typename accT>
Array<T> convolve(Array<T> const &signal, Array<accT> const &filter,
                  AF_BATCH_KIND kind, const int rank, const bool expand) {
    const dim4 &sDims = signal.dims();
    const dim4 &fDims = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for (int d = 0; d < AF_MAX_DIMS; ++d) {
            if (kind == AF_BATCH_NONE || kind == AF_BATCH_RHS) {
                oDims[d] = sDims[d] + fDims[d] - 1;
            } else {
                oDims[d] = (d < rank ? sDims[d] + fDims[d] - 1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind == AF_BATCH_RHS) {
            for (int i = rank; i < AF_MAX_DIMS; ++i) { oDims[i] = fDims[i]; }
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);

    kernel::convolve_nd<T, accT>(out, signal, filter, kind, rank, expand);

    return out;
}

template<typename T, typename accT>
Array<T> convolve2(Array<T> const &signal, Array<accT> const &c_filter,
                   Array<accT> const &r_filter, const bool expand) {
    const dim4 &cfDims = c_filter.dims();
    const dim4 &rfDims = r_filter.dims();

    const dim_t cfLen = cfDims.elements();
    const dim_t rfLen = rfDims.elements();

    const dim4 &sDims = signal.dims();
    dim4 tDims        = sDims;
    dim4 oDims        = sDims;

    if (expand) {
        tDims[0] += cfLen - 1;
        oDims[0] += cfLen - 1;
        oDims[1] += rfLen - 1;
    }

    Array<T> temp = createEmptyArray<T>(tDims);
    Array<T> out  = createEmptyArray<T>(oDims);

    kernel::convolve2<T, accT>(temp, signal, c_filter, 0, expand);
    kernel::convolve2<T, accT>(out, temp, r_filter, 1, expand);

    return out;
}

#define INSTANTIATE(T, accT)                                                   \
    template Array<T> convolve<T, accT>(Array<T> const &, Array<accT> const &, \
                                        AF_BATCH_KIND, const int, const bool); \
    template Array<T> convolve2<T, accT>(Array<T> const &,                     \
                                         Array<accT> const &,                  \
                                         Array<accT> const &, const bool);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat, cfloat)
INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(uint, float)
INSTANTIATE(int, float)
INSTANTIATE(uchar, float)
INSTANTIATE(char, float)
INSTANTIATE(ushort, float)
INSTANTIATE(short, float)
INSTANTIATE(uintl, float)
INSTANTIATE(intl, float)
#undef INSTANTIATE

}  // namespace cuda
}  // namespace arrayfire

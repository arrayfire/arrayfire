/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <convolve.hpp>
#include <err_cpu.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/convolve.hpp>

using af::dim4;

namespace cpu
{

template<typename T, typename accT, dim_t baseDim, bool expand>
Array<T> convolve(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind)
{
    signal.eval();
    filter.eval();

    auto sDims    = signal.dims();
    auto fDims    = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for(dim_t d=0; d<4; ++d) {
            if (kind==AF_BATCH_NONE || kind==AF_BATCH_RHS) {
                oDims[d] = sDims[d]+fDims[d]-1;
            } else {
                oDims[d] = (d<baseDim ? sDims[d]+fDims[d]-1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind==AF_BATCH_RHS) {
            for (dim_t i=baseDim; i<4; ++i)
                oDims[i] = fDims[i];
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);

    getQueue().enqueue(kernel::convolve_nd<T, accT, baseDim, expand>,out, signal, filter, kind);

    return out;
}

template<typename T, typename accT, bool expand>
Array<T> convolve2(Array<T> const& signal, Array<accT> const& c_filter, Array<accT> const& r_filter)
{
    signal.eval();
    c_filter.eval();
    r_filter.eval();

    auto sDims = signal.dims();
    dim4 tDims = sDims;
    dim4 oDims = sDims;

    if (expand) {
        auto cfDims   = c_filter.dims();
        auto rfDims   = r_filter.dims();

        dim_t cflen = (dim_t)cfDims.elements();
        dim_t rflen = (dim_t)rfDims.elements();
        // separable convolve only does AF_BATCH_NONE and standard batch(AF_BATCH_LHS)
        tDims[0] += cflen - 1;
        oDims[0] += cflen - 1;
        oDims[1] += rflen - 1;
    }

    Array<T> out  = createEmptyArray<T>(oDims);
    Array<T> temp = createEmptyArray<T>(tDims);

    getQueue().enqueue(kernel::convolve2<T, accT, expand>, out, signal, c_filter, r_filter, temp);

    return out;
}

#define INSTANTIATE(T, accT)                                            \
    template Array<T> convolve <T, accT, 1, true >(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind); \
    template Array<T> convolve <T, accT, 1, false>(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind); \
    template Array<T> convolve <T, accT, 2, true >(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind); \
    template Array<T> convolve <T, accT, 2, false>(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind); \
    template Array<T> convolve <T, accT, 3, true >(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind); \
    template Array<T> convolve <T, accT, 3, false>(Array<T> const& signal, Array<accT> const& filter, AF_BATCH_KIND kind); \
    template Array<T> convolve2<T, accT, true >(Array<T> const& signal, Array<accT> const& c_filter, Array<accT> const& r_filter); \
    template Array<T> convolve2<T, accT, false>(Array<T> const& signal, Array<accT> const& c_filter, Array<accT> const& r_filter);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)
INSTANTIATE(ushort ,   float)
INSTANTIATE(short  ,   float)
INSTANTIATE(uintl  ,   float)
INSTANTIATE(intl   ,   float)

}

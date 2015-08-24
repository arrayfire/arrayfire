/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <convolve.hpp>
#include <kernel/convolve_separable.hpp>
#include <err_opencl.hpp>

using af::dim4;

namespace opencl
{

template<typename T, typename accT, bool expand>
Array<T> convolve2(Array<T> const& signal, Array<accT> const& c_filter, Array<accT> const& r_filter)
{
    const dim_t cflen = (dim_t)c_filter.elements();
    const dim_t rflen = (dim_t)r_filter.elements();

    if ((cflen > kernel::MAX_SCONV_FILTER_LEN) ||
        (rflen > kernel::MAX_SCONV_FILTER_LEN)) {
        // call upon fft
        OPENCL_NOT_SUPPORTED();
    }

    const dim4 sDims = signal.dims();
    dim4 tDims = sDims;
    dim4 oDims = sDims;

    if (expand) {
        tDims[0] += cflen - 1;
        oDims[0] += cflen - 1;
        oDims[1] += rflen - 1;
    }

    Array<T> temp= createEmptyArray<T>(tDims);
    Array<T> out = createEmptyArray<T>(oDims);

    kernel::convSep<T, accT, 0, expand>(temp, signal, c_filter);
    kernel::convSep<T, accT, 1, expand>( out,   temp, r_filter);

    return out;
}

#define INSTANTIATE(T, accT)  \
    template Array<T> convolve2<T, accT, true >(Array<T> const& signal, Array<accT> const& c_filter, Array<accT> const& r_filter);  \
    template Array<T> convolve2<T, accT, false>(Array<T> const& signal, Array<accT> const& c_filter, Array<accT> const& r_filter);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)

}

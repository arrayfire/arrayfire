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

template<typename T, typename aT, dim_type cDim, bool expand>
void conv2Helper(Array<T>& out, const Array<T>& sig, const Array<T>& filt, dim_type f)
{
    switch(f) {
        case  2: kernel::convolve2<T, aT, cDim, expand,  2>(out, sig, filt); break;
        case  3: kernel::convolve2<T, aT, cDim, expand,  3>(out, sig, filt); break;
        case  4: kernel::convolve2<T, aT, cDim, expand,  4>(out, sig, filt); break;
        case  5: kernel::convolve2<T, aT, cDim, expand,  5>(out, sig, filt); break;
        case  6: kernel::convolve2<T, aT, cDim, expand,  6>(out, sig, filt); break;
        case  7: kernel::convolve2<T, aT, cDim, expand,  7>(out, sig, filt); break;
        case  8: kernel::convolve2<T, aT, cDim, expand,  8>(out, sig, filt); break;
        case  9: kernel::convolve2<T, aT, cDim, expand,  9>(out, sig, filt); break;
        case 10: kernel::convolve2<T, aT, cDim, expand, 10>(out, sig, filt); break;
        case 11: kernel::convolve2<T, aT, cDim, expand, 11>(out, sig, filt); break;
        case 12: kernel::convolve2<T, aT, cDim, expand, 12>(out, sig, filt); break;
        case 13: kernel::convolve2<T, aT, cDim, expand, 13>(out, sig, filt); break;
        case 14: kernel::convolve2<T, aT, cDim, expand, 14>(out, sig, filt); break;
        case 15: kernel::convolve2<T, aT, cDim, expand, 15>(out, sig, filt); break;
        case 16: kernel::convolve2<T, aT, cDim, expand, 16>(out, sig, filt); break;
        case 17: kernel::convolve2<T, aT, cDim, expand, 17>(out, sig, filt); break;
        case 18: kernel::convolve2<T, aT, cDim, expand, 18>(out, sig, filt); break;
        case 19: kernel::convolve2<T, aT, cDim, expand, 19>(out, sig, filt); break;
        case 20: kernel::convolve2<T, aT, cDim, expand, 20>(out, sig, filt); break;
        case 21: kernel::convolve2<T, aT, cDim, expand, 21>(out, sig, filt); break;
        case 22: kernel::convolve2<T, aT, cDim, expand, 22>(out, sig, filt); break;
        case 23: kernel::convolve2<T, aT, cDim, expand, 23>(out, sig, filt); break;
        case 24: kernel::convolve2<T, aT, cDim, expand, 24>(out, sig, filt); break;
        case 25: kernel::convolve2<T, aT, cDim, expand, 25>(out, sig, filt); break;
        case 26: kernel::convolve2<T, aT, cDim, expand, 26>(out, sig, filt); break;
        case 27: kernel::convolve2<T, aT, cDim, expand, 27>(out, sig, filt); break;
        case 28: kernel::convolve2<T, aT, cDim, expand, 28>(out, sig, filt); break;
        case 29: kernel::convolve2<T, aT, cDim, expand, 29>(out, sig, filt); break;
        case 30: kernel::convolve2<T, aT, cDim, expand, 30>(out, sig, filt); break;
        case 31: kernel::convolve2<T, aT, cDim, expand, 31>(out, sig, filt); break;
        default: OPENCL_NOT_SUPPORTED();
    }
}

template<typename T, typename accT, bool expand>
Array<T> convolve2(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter)
{
    if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
        !isDoubleSupported(getActiveDeviceId())) {
        OPENCL_NOT_SUPPORTED();
    }
    const dim4 cfDims   = c_filter.dims();
    const dim4 rfDims   = r_filter.dims();

    if ((cfDims[0] > kernel::MAX_SCONV_FILTER_LEN) ||
            (rfDims[0] > kernel::MAX_SCONV_FILTER_LEN)) {
        // call upon fft
        OPENCL_NOT_SUPPORTED();
    }

    const dim4 sDims = signal.dims();
    dim4 oDims(1);
    if (expand) {
        oDims[0] = sDims[0]+cfDims[0]-1;
        oDims[1] = sDims[1]+rfDims[0]-1;
        oDims[2] = sDims[2];
    } else {
        oDims = sDims;
    }

    Array<T> temp= createEmptyArray<T>(oDims);
    Array<T> out = createEmptyArray<T>(oDims);

    conv2Helper<T, accT, 0, expand>(temp, signal, c_filter, cfDims[0]);
    conv2Helper<T, accT, 1, expand>(out, temp, r_filter, rfDims[0]);

    return out;
}

#define INSTANTIATE(T, accT)  \
    template Array<T> convolve2<T, accT, true >(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter);  \
    template Array<T> convolve2<T, accT, false>(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)

}

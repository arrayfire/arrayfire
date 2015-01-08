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
#include <kernel/convolve.hpp>
#include <err_opencl.hpp>

using af::dim4;

namespace opencl
{

template<typename T, typename accT, dim_type baseDim, bool expand>
Array<T> * convolve(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind)
{
    if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
        !isDoubleSupported(getActiveDeviceId())) {
        OPENCL_NOT_SUPPORTED();
    }
    const dim4 sDims    = signal.dims();
    const dim4 fDims    = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for(dim_type d=0; d<4ll; ++d) {
            if (kind==ONE2ONE || kind==ONE2ALL) {
                oDims[d] = sDims[d]+fDims[d]-1;
            } else {
                oDims[d] = (d<baseDim ? sDims[d]+fDims[d]-1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind==ONE2ALL) oDims[baseDim] = fDims[baseDim];
    }

    Array<T> *out   = createEmptyArray<T>(oDims);
    bool callKernel = true;

    dim_type MCFL2 = kernel::MAX_CONV2_FILTER_LEN;
    dim_type MCFL3 = kernel::MAX_CONV3_FILTER_LEN;
    switch(baseDim) {
        case 1:
            if (fDims[0]>kernel::MAX_CONV1_FILTER_LEN)
                callKernel = false;
            break;
        case 2:
            if ((fDims[0]*fDims[1]) > (MCFL2 * MCFL2))
                callKernel = false;
            break;
        case 3:
            if ((fDims[0]*fDims[1]*fDims[2]) > (MCFL3 * MCFL3 * MCFL3))
                callKernel = false;
            break;
    }

    if (callKernel)
        kernel::convolve_nd<T, accT, baseDim, expand>(*out, signal, filter, kind);
    else {
        // call upon fft
        OPENCL_NOT_SUPPORTED();
    }

    return out;
}

template<typename T, typename accT, bool expand>
Array<T> * convolve2(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter)
{
    if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
        !isDoubleSupported(getActiveDeviceId())) {
        OPENCL_NOT_SUPPORTED();
    }
    const dim4 cfDims   = c_filter.dims();
    const dim4 rfDims   = r_filter.dims();

    if((cfDims[0]*rfDims[0]) > (kernel::MAX_CONV2_FILTER_LEN * kernel::MAX_CONV2_FILTER_LEN)) {
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

    Array<T> *temp= createEmptyArray<T>(oDims);
    Array<T> *out = createEmptyArray<T>(oDims);

    kernel::convolve2<T, accT, 0, expand>(*temp, signal, c_filter);
    kernel::convolve2<T, accT, 1, expand>(*out, *temp, r_filter);

    destroyArray<T>(*temp);

    return out;
}

#define INSTANTIATE(T, accT)  \
    template Array<T> * convolve <T, accT, 1, true >(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 1, false>(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 2, true >(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 2, false>(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 3, true >(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 3, false>(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve2<T, accT, true >(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter);  \
    template Array<T> * convolve2<T, accT, false>(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)

}

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

template<typename T, typename accT, dim_t baseDim, bool expand>
Array<T> convolve(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind)
{
    const dim4 sDims    = signal.dims();
    const dim4 fDims    = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for(dim_t d=0; d<4; ++d) {
            if (kind==CONVOLVE_BATCH_NONE || kind==CONVOLVE_BATCH_KERNEL) {
                oDims[d] = sDims[d]+fDims[d]-1;
            } else {
                oDims[d] = (d<baseDim ? sDims[d]+fDims[d]-1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind==CONVOLVE_BATCH_KERNEL) {
            for (dim_t i=baseDim; i<4; ++i)
                oDims[i] = fDims[i];
        }
    }

    Array<T> out   = createEmptyArray<T>(oDims);
    bool callKernel = true;

    dim_t MCFL2 = kernel::MAX_CONV2_FILTER_LEN;
    dim_t MCFL3 = kernel::MAX_CONV3_FILTER_LEN;
    switch(baseDim) {
        case 1: if (fDims[0]>kernel::MAX_CONV1_FILTER_LEN) callKernel = false; break;
        case 2: if ((fDims[0]*fDims[1]) > (MCFL2 * MCFL2)) callKernel = false; break;
        case 3: if ((fDims[0]*fDims[1]*fDims[2]) > (MCFL3 * MCFL3 * MCFL3)) callKernel = false; break;
    }

    if(!callKernel) { OPENCL_NOT_SUPPORTED(); }

    kernel::convolve_nd<T, accT, baseDim, expand>(out, signal, filter, kind);

    return out;
}

#define INSTANTIATE(T, accT)                                            \
    template Array<T> convolve <T, accT, 1, true >(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 1, false>(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 2, true >(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 2, false>(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 3, true >(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \
    template Array<T> convolve <T, accT, 3, false>(Array<T> const& signal, Array<accT> const& filter, ConvolveBatchKind kind); \

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)

}

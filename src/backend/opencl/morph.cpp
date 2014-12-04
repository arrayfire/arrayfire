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
#include <math.hpp>
#include <morph.hpp>
#include <kernel/morph.hpp>
#include <err_opencl.hpp>

using af::dim4;

namespace opencl
{

template<typename T, bool isDilation>
Array<T> * morph(const Array<T> &in, const Array<T> &mask)
{
    if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
        !isDoubleSupported(getActiveDeviceId())) {
        OPENCL_NOT_SUPPORTED();
    }
    const dim4 mdims    = mask.dims();

    if (mdims[0]!=mdims[1])
        AF_ERROR("Only square masks are supported in opencl morph currently", AF_ERR_SIZE);
    if (mdims[0]>19)
        AF_ERROR("Upto 19x19 square kernels are only supported in opencl currently", AF_ERR_SIZE);

    const dim4 dims = in.dims();
    Array<T>* out   = createEmptyArray<T>(dims);

    switch(mdims[0]) {
        case  3: kernel::morph<T, isDilation,  3>(*out, in, mask); break;
        case  5: kernel::morph<T, isDilation,  5>(*out, in, mask); break;
        case  7: kernel::morph<T, isDilation,  7>(*out, in, mask); break;
        case  9: kernel::morph<T, isDilation,  9>(*out, in, mask); break;
        case 11: kernel::morph<T, isDilation, 11>(*out, in, mask); break;
        case 13: kernel::morph<T, isDilation, 13>(*out, in, mask); break;
        case 15: kernel::morph<T, isDilation, 15>(*out, in, mask); break;
        case 17: kernel::morph<T, isDilation, 17>(*out, in, mask); break;
        case 19: kernel::morph<T, isDilation, 19>(*out, in, mask); break;
        default: kernel::morph<T, isDilation,  3>(*out, in, mask); break;
    }


    return out;
}

template<typename T, bool isDilation>
Array<T> * morph3d(const Array<T> &in, const Array<T> &mask)
{
    if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
        !isDoubleSupported(getActiveDeviceId())) {
        OPENCL_NOT_SUPPORTED();
    }
    const dim4 mdims    = mask.dims();

    if (mdims[0]!=mdims[1] || mdims[0]!=mdims[2])
        AF_ERROR("Only cube masks are supported in opencl morph currently", AF_ERR_SIZE);
    if (mdims[0]>7)
        AF_ERROR("Upto 7x7x7 kernels are only supported in opencl currently", AF_ERR_SIZE);

    const dim4 dims     = in.dims();
    if (dims[3]>1)
        AF_ERROR("Batch not supported for volumetic morphological operations", AF_ERR_NOT_SUPPORTED);

    Array<T>* out   = createEmptyArray<T>(dims);

    switch(mdims[0]) {
        case  3: kernel::morph3d<T, isDilation,  3>(*out, in, mask); break;
        case  5: kernel::morph3d<T, isDilation,  5>(*out, in, mask); break;
        case  7: kernel::morph3d<T, isDilation,  7>(*out, in, mask); break;
        default: kernel::morph3d<T, isDilation,  3>(*out, in, mask); break;
    }

    return out;
}

#define INSTANTIATE(T)\
    template Array<T> * morph  <T, true >(const Array<T> &in, const Array<T> &mask);\
    template Array<T> * morph  <T, false>(const Array<T> &in, const Array<T> &mask);\
    template Array<T> * morph3d<T, true >(const Array<T> &in, const Array<T> &mask);\
    template Array<T> * morph3d<T, false>(const Array<T> &in, const Array<T> &mask);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}

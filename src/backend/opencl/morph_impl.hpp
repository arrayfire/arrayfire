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
Array<T> morph(const Array<T> &in, const Array<T> &mask)
{
    const dim4 mdims    = mask.dims();

    if (mdims[0]!=mdims[1])
        AF_ERROR("Only square masks are supported in opencl morph currently", AF_ERR_SIZE);
    if (mdims[0]>19)
        AF_ERROR("Upto 19x19 square kernels are only supported in opencl currently", AF_ERR_SIZE);

    const dim4 dims = in.dims();
    Array<T> out   = createEmptyArray<T>(dims);

    switch(mdims[0]) {
        case  3: kernel::morph<T, isDilation,  3>(out, in, mask); break;
        case  5: kernel::morph<T, isDilation,  5>(out, in, mask); break;
        case  7: kernel::morph<T, isDilation,  7>(out, in, mask); break;
        case  9: kernel::morph<T, isDilation,  9>(out, in, mask); break;
        case 11: kernel::morph<T, isDilation, 11>(out, in, mask); break;
        case 13: kernel::morph<T, isDilation, 13>(out, in, mask); break;
        case 15: kernel::morph<T, isDilation, 15>(out, in, mask); break;
        case 17: kernel::morph<T, isDilation, 17>(out, in, mask); break;
        case 19: kernel::morph<T, isDilation, 19>(out, in, mask); break;
        default: kernel::morph<T, isDilation,  3>(out, in, mask); break;
    }

    return out;
}

}

#define INSTANTIATE(T, ISDILATE)                                                 \
    template Array<T> morph  <T, ISDILATE>(const Array<T> &in, const Array<T> &mask);

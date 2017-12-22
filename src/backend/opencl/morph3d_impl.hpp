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
#include <math.hpp>
#include <morph.hpp>
#include <kernel/morph.hpp>
#include <err_opencl.hpp>

using af::dim4;

namespace opencl
{
template<typename T, bool isDilation>
Array<T> morph3d(const Array<T> &in, const Array<T> &mask)
{
    const dim4 mdims    = mask.dims();

    if (mdims[0]!=mdims[1] || mdims[0]!=mdims[2])
        OPENCL_NOT_SUPPORTED("Only cubic masks are supported");

    if (mdims[0]>7)
        OPENCL_NOT_SUPPORTED("Kernels > 7x7x7 masks are not supported");

    const dim4 dims= in.dims();
    Array<T> out   = createEmptyArray<T>(dims);

    switch(mdims[0]) {
        case  2: kernel::morph3d<T, isDilation,  2>(out, in, mask); break;
        case  3: kernel::morph3d<T, isDilation,  3>(out, in, mask); break;
        case  4: kernel::morph3d<T, isDilation,  4>(out, in, mask); break;
        case  5: kernel::morph3d<T, isDilation,  5>(out, in, mask); break;
        case  6: kernel::morph3d<T, isDilation,  6>(out, in, mask); break;
        case  7: kernel::morph3d<T, isDilation,  7>(out, in, mask); break;
        default: assert(mdims[0] < 7 & "Kernel size should be haandled above."); break;
    }

    return out;
}

#define INSTANTIATE(T, ISDILATE)                                                 \
    template Array<T> morph3d<T, ISDILATE>(const Array<T> &in, const Array<T> &mask);
}

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
#include <morph.hpp>
#include <kernel/morph.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename T, bool isDilation>
Array<T> morph3d(const Array<T> &in, const Array<T> &mask)
{
    const dim4 mdims = mask.dims();

    if (mdims[0] != mdims[1] || mdims[0] != mdims[2])
        AF_ERROR("Only cube masks are supported in CUDA backend", AF_ERR_SIZE);

    if (mdims[0] > 7)
        AF_ERROR("Upto 7x7x7 kernels are only supported in CUDA backend", AF_ERR_SIZE);

    Array<T> out       = createEmptyArray<T>(in.dims());

    CUDA_CHECK(cudaMemcpyToSymbol(kernel::cFilter, mask.get(),
                                  mdims[0] * mdims[1] *mdims[2] * sizeof(T),
                                  0, cudaMemcpyDeviceToDevice));

    if (isDilation)
        kernel::morph3d<T, true >(out, in, mdims[0]);
    else
        kernel::morph3d<T, false>(out, in, mdims[0]);

    return out;
}

}

#define INSTANTIATE(T, ISDILATE)                                        \
    template Array<T> morph3d<T, ISDILATE>(const Array<T> &in, const Array<T> &mask);

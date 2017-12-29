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
#include <morph.hpp>
#include <kernel/morph.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename T, bool isDilation>
Array<T>  morph(const Array<T> &in, const Array<T> &mask)
{
    const dim4 mdims = mask.dims();

    if (mdims[0] != mdims[1])
        CUDA_NOT_SUPPORTED("Rectangular masks are not supported");

    if (mdims[0] > 19)
        CUDA_NOT_SUPPORTED("Kernels > 19x19 are not supported");

    Array<T> out = createEmptyArray<T>(in.dims());

    CUDA_CHECK(cudaMemcpyToSymbolAsync(kernel::cFilter, mask.get(),
                                  mdims[0] * mdims[1] * sizeof(T),
                                  0, cudaMemcpyDeviceToDevice,
                                  cuda::getActiveStream()));

    if (isDilation)
        kernel::morph<T, true >(out, in, mdims[0]);
    else
        kernel::morph<T, false>(out, in, mdims[0]);

    return out;
}

}

#define INSTANTIATE(T, ISDILATE)                                        \
    template Array<T> morph  <T, ISDILATE>(const Array<T> &in, const Array<T> &mask);

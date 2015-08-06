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
Array<T>  morph(const Array<T> &in, const Array<T> &mask)
{
    const dim4 mdims = mask.dims();

    if (mdims[0] != mdims[1])
        AF_ERROR("Only square masks are supported in cuda morph currently", AF_ERR_SIZE);
    if (mdims[0] > 19)
        AF_ERROR("Upto 19x19 square kernels are only supported in cuda currently", AF_ERR_SIZE);

    Array<T> out = createEmptyArray<T>(in.dims());

    CUDA_CHECK(cudaMemcpyToSymbolAsync(kernel::cFilter, mask.get(),
                                  mdims[0] * mdims[1] * sizeof(T),
                                  0, cudaMemcpyDeviceToDevice,
                                  cuda::getStream(cuda::getActiveDeviceId())));

    if (isDilation)
        kernel::morph<T, true >(out, in, mdims[0]);
    else
        kernel::morph<T, false>(out, in, mdims[0]);

    return out;
}

}

#define INSTANTIATE(T, ISDILATE)                                        \
    template Array<T> morph  <T, ISDILATE>(const Array<T> &in, const Array<T> &mask);

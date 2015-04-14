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
#include <regions.hpp>
#include <kernel/regions.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename T>
Array<T>  regions(const Array<char> &in, af_connectivity connectivity)
{
    ARG_ASSERT(2, (connectivity==AF_CONNECTIVITY_4 || connectivity==AF_CONNECTIVITY_8));

    const dim4 dims = in.dims();

    Array<T>  out  = createEmptyArray<T>(dims);

    // Create bindless texture object for the equiv map.
    cudaTextureObject_t tex = 0;
    // FIXME: Currently disabled, only supported on capaibility >= 3.0
    //if (compute >= 3.0) {
    //    cudaResourceDesc resDesc;
    //    memset(&resDesc, 0, sizeof(resDesc));
    //    resDesc.resType = cudaResourceTypeLinear;
    //    resDesc.res.linear.devPtr = out->get();
    //    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    //    resDesc.res.linear.desc.x = 32; // bits per channel
    //    resDesc.res.linear.sizeInBytes = dims[0] * dims[1] * sizeof(float);
    //    cudaTextureDesc texDesc;
    //    memset(&texDesc, 0, sizeof(texDesc));
    //    texDesc.readMode = cudaReadModeElementType;
    //    CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    //}

    switch(connectivity) {
        case AF_CONNECTIVITY_4:
            ::regions<T, false, 2>(out, in, tex);
            break;
        case AF_CONNECTIVITY_8:
            ::regions<T, true,  2>(out, in, tex);
            break;
    }

    return out;
}

#define INSTANTIATE(T)\
    template Array<T>  regions<T>(const Array<char> &in, af_connectivity connectivity);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(int   )
INSTANTIATE(uint  )

}

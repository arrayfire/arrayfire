/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_cuda.hpp>
#include <kernel/regions.hpp>
#include <regions.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace cuda {

template<typename T>
Array<T> regions(const Array<char> &in, af_connectivity connectivity) {
    const dim4 dims = in.dims();

    Array<T> out = createEmptyArray<T>(dims);

    // Create bindless texture object for the equiv map.
    cudaTextureObject_t tex = 0;

    // Use texture objects with compute 3.0 or higher
    if (!std::is_same<T, double>::value) {
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType           = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = out.get();

        if (std::is_signed<T>::value)
            resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
        else if (std::is_unsigned<T>::value)
            resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        else
            resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;

        resDesc.res.linear.desc.x      = sizeof(T) * 8;  // bits per channel
        resDesc.res.linear.sizeInBytes = dims[0] * dims[1] * sizeof(T);
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    }

    switch (connectivity) {
        case AF_CONNECTIVITY_4: ::regions<T, false, 2>(out, in, tex); break;
        case AF_CONNECTIVITY_8: ::regions<T, true, 2>(out, in, tex); break;
    }

    // Iterative procedure(while loop) in kernel::regions
    // does stream synchronization towards loop end. So, it is
    // safe to destroy the texture object
    CUDA_CHECK(cudaDestroyTextureObject(tex));

    return out;
}

#define INSTANTIATE(T)                                  \
    template Array<T> regions<T>(const Array<char> &in, \
                                 af_connectivity connectivity);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cuda
}  // namespace arrayfire

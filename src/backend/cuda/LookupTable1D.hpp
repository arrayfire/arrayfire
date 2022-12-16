/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <err_cuda.hpp>

#include <type_traits>

namespace arrayfire {
namespace cuda {

template<typename T>
class LookupTable1D {
   public:
    LookupTable1D()                                     = delete;
    LookupTable1D(const LookupTable1D& arg)             = delete;
    LookupTable1D(const LookupTable1D&& arg)            = delete;
    LookupTable1D& operator=(const LookupTable1D& arg)  = delete;
    LookupTable1D& operator=(const LookupTable1D&& arg) = delete;

    LookupTable1D(const Array<T>& lutArray) : mTexture(0), mData(lutArray) {
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));

        resDesc.resType                = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr      = mData.get();
        resDesc.res.linear.desc.x      = sizeof(T) * 8;
        resDesc.res.linear.sizeInBytes = mData.elements() * sizeof(T);

        if (std::is_signed<T>::value)
            resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
        else if (std::is_unsigned<T>::value)
            resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        else
            resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;

        texDesc.readMode = cudaReadModeElementType;

        CUDA_CHECK(
            cudaCreateTextureObject(&mTexture, &resDesc, &texDesc, NULL));
    }

    ~LookupTable1D() {
        if (mTexture) { cudaDestroyTextureObject(mTexture); }
    }

    cudaTextureObject_t get() const noexcept { return mTexture; }

   private:
    // Keep a copy so that ref count doesn't go down to zero when
    // original Array<T> goes out of scope before LookupTable1D object does.
    Array<T> mData;
    cudaTextureObject_t mTexture;
};

}  // namespace cuda
}  // namespace arrayfire

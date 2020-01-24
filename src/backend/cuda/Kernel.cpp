/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Kernel.hpp>

#include <platform.hpp>

namespace cuda {

Kernel::DevPtrType Kernel::get(const char *name) {
    Kernel::DevPtrType out = 0;
    size_t size            = 0;
    CU_CHECK(cuModuleGetGlobal(&out, &size, this->getModule(), name));
    return out;
}

void Kernel::copyToReadOnly(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                            size_t bytes) {
    CU_CHECK(cuMemcpyDtoDAsync(dst, src, bytes, cuda::getActiveStream()));
}

void Kernel::setScalar(Kernel::DevPtrType dst, int value) {
    CU_CHECK(
        cuMemcpyHtoDAsync(dst, &value, sizeof(int), cuda::getActiveStream()));
    CU_CHECK(cuStreamSynchronize(cuda::getActiveStream()));
}

int Kernel::getScalar(Kernel::DevPtrType src) {
    int retVal = 0;
    CU_CHECK(
        cuMemcpyDtoHAsync(&retVal, src, sizeof(int), cuda::getActiveStream()));
    CU_CHECK(cuStreamSynchronize(cuda::getActiveStream()));
    return retVal;
}

}  // namespace cuda

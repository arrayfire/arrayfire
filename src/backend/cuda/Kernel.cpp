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

namespace arrayfire {
namespace cuda {

Kernel::DevPtrType Kernel::getDevPtr(const char* name) {
    Kernel::DevPtrType out = 0;
    size_t size            = 0;
    CU_CHECK(cuModuleGetGlobal(&out, &size, this->getModuleHandle(), name));
    return out;
}

void Kernel::copyToReadOnly(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                            size_t bytes) {
    CU_CHECK(cuMemcpyDtoDAsync(dst, src, bytes, getActiveStream()));
}

void Kernel::setFlag(Kernel::DevPtrType dst, int* scalarValPtr,
                     const bool syncCopy) {
    CU_CHECK(
        cuMemcpyHtoDAsync(dst, scalarValPtr, sizeof(int), getActiveStream()));
    if (syncCopy) { CU_CHECK(cuStreamSynchronize(getActiveStream())); }
}

int Kernel::getFlag(Kernel::DevPtrType src) {
    int retVal = 0;
    CU_CHECK(cuMemcpyDtoHAsync(&retVal, src, sizeof(int), getActiveStream()));
    CU_CHECK(cuStreamSynchronize(getActiveStream()));
    return retVal;
}

}  // namespace cuda
}  // namespace arrayfire

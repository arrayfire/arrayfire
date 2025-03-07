/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Kernel.hpp>

#include <backend.hpp>
#include <cl2hpp.hpp>
#include <common/defines.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace opencl {

Kernel::DevPtrType Kernel::getDevPtr(const char* name) {
    UNUSED(name);
    return nullptr;
}

void Kernel::copyToReadOnly(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                            size_t bytes) {
    getQueue().enqueueCopyBuffer(*src, *dst, 0, 0, bytes);
}

void Kernel::setFlag(Kernel::DevPtrType dst, int* scalarValPtr,
                     const bool syncCopy) {
    UNUSED(syncCopy);
    getQueue().enqueueFillBuffer(*dst, *scalarValPtr, 0, sizeof(int));
}

int Kernel::getFlag(Kernel::DevPtrType src) {
    int retVal = 0;
    getQueue().enqueueReadBuffer(*src, CL_TRUE, 0, sizeof(int), &retVal);
    return retVal;
}

}  // namespace opencl
}  // namespace arrayfire

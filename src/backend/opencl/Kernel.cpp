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
#include <platform.hpp>

namespace opencl {

Kernel::DevPtrType Kernel::get(const char *name) { return nullptr; }

void Kernel::copyToReadOnly(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                            size_t bytes) {
    getQueue().enqueueCopyBuffer(*src, *dst, 0, 0, bytes);
}

void Kernel::setScalar(Kernel::DevPtrType dst, int value) {
    getQueue().enqueueWriteBuffer(*dst, CL_FALSE, 0, sizeof(int), &value);
}

int Kernel::getScalar(Kernel::DevPtrType src) {
    int retVal = 0;
    getQueue().enqueueReadBuffer(*src, CL_TRUE, 0, sizeof(int), &retVal);
    return retVal;
}

}  // namespace opencl

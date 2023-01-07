/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/opencl.h>
#include <cstring>

namespace af {
template<>
AFAPI cl_mem *array::device() const {
    auto *mem_ptr = new cl_mem;
    void *dptr    = nullptr;
    af_err err    = af_get_device_ptr(&dptr, get());
    memcpy(mem_ptr, &dptr, sizeof(void *));
    if (err != AF_SUCCESS) {
        throw af::exception("Failed to get cl_mem from array object");
    }
    return mem_ptr;
}
}  // namespace af

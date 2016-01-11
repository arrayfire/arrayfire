/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <platform.hpp>
#include <af/defines.h>

namespace opencl
{

    cl::Buffer *bufferAlloc(const size_t &bytes);
    void bufferFree(cl::Buffer *buf);

    template<typename T> T *memAlloc(const size_t &elements);

    // Need these as 2 separate function and not a default argument
    // This is because it is used as the deleter in shared pointer
    // which cannot support default arguments
    template<typename T> void memFree(T* ptr);
    template<typename T> void memFreeLocked(T* ptr, bool user_unlock);
    template<typename T> void memLock(const T *ptr);
    template<typename T> void memUnlock(const T *ptr);

    template<typename T> T* pinnedAlloc(const size_t &elements);
    template<typename T> void pinnedFree(T* ptr);

    static const unsigned MAX_BUFFERS   = 100;
    static const unsigned MAX_BYTES     = (1 << 30);

    void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                          size_t *lock_bytes,  size_t *lock_buffers);
    void garbageCollect();
    void pinnedGarbageCollect();

    void printMemInfo(const char *msg, const int device);

    void setMemStepSize(size_t step_bytes);
    size_t getMemStepSize(void);
}

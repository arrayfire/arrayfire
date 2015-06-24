/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/defines.h>
namespace cpu
{
    template<typename T> T* memAlloc(const size_t &elements);
    template<typename T> void memFree(T* ptr);
    template<typename T> void memPop(const T *ptr);
    template<typename T> void memPush(const T *ptr);

    template<typename T> T* pinnedAlloc(const size_t &elements);
    template<typename T> void pinnedFree(T* ptr);

    static const unsigned MAX_BUFFERS = 100;
    static const unsigned MAX_BYTES = 100 * (1 << 20);

    void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                          size_t *lock_bytes,  size_t *lock_buffers);
    void garbageCollect();
    void pinnedGarbageCollect();

    void setMemStepSize(size_t step_bytes);
    size_t getMemStepSize(void);
}

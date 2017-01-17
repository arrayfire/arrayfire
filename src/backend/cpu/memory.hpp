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
#include <common/MemoryManager.hpp>

namespace cpu
{
template<typename T> T* memAlloc(const size_t &elements);
void *memAllocUser(const size_t &bytes);

// Need these as 2 separate function and not a default argument
// This is because it is used as the deleter in shared pointer
// which cannot support default arguments
template<typename T> void memFree(T* ptr);
void memFreeUser(void* ptr);

void memLock(const void *ptr);
void memUnlock(const void *ptr);
bool isLocked(const void *ptr);

template<typename T> T* pinnedAlloc(const size_t &elements);
template<typename T> void pinnedFree(T* ptr);

size_t getMaxBytes();
unsigned getMaxBuffers();

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes,  size_t *lock_buffers);
void garbageCollect();
void pinnedGarbageCollect();

void printMemInfo(const char *msg, const int device);

void setMemStepSize(size_t step_bytes);
size_t getMemStepSize(void);
bool checkMemoryLimit();

class MemoryManager : public common::MemoryManager<cpu::MemoryManager>
{
    public:
        MemoryManager();
        ~MemoryManager();
        int getActiveDeviceId();
        size_t getMaxMemorySize(int id);
        void *nativeAlloc(const size_t bytes);
        void nativeFree(void *ptr);
};
}

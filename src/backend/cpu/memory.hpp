/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <common/MemoryManager.hpp>
#include <af/defines.h>

#include <functional>
#include <memory>

namespace cpu {

template<typename T>
using uptr = std::unique_ptr<T[], std::function<void(T[])>>;

template<typename T>
std::unique_ptr<T[], std::function<void(T *)>> memAlloc(const size_t &elements);
void *memAllocUser(const size_t &bytes);

// Need these as 2 separate function and not a default argument
// This is because it is used as the deleter in shared pointer
// which cannot support default arguments
template<typename T>
void memFree(T *ptr);
void memFreeUser(void *ptr);

void memLock(const void *ptr);
void memUnlock(const void *ptr);
bool isLocked(const void *ptr);

template<typename T>
T *pinnedAlloc(const size_t &elements);
template<typename T>
void pinnedFree(T *ptr);

size_t getMaxBytes();
unsigned getMaxBuffers();

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes, size_t *lock_buffers);
void garbageCollect();
void shutdownMemoryManager();
void pinnedGarbageCollect();

void printMemInfo(const char *msg, const int device);

void setMemStepSize(size_t step_bytes);
size_t getMemStepSize(void);
bool checkMemoryLimit();

class NativeMemoryInterface : public common::memory::NativeMemoryInterface {
   public:
    NativeMemoryInterface();
    ~NativeMemoryInterface();
    int getActiveDeviceId() override;
    size_t getMaxMemorySize(int id) override;
    void *nativeAlloc(const size_t bytes) override;
    void nativeFree(void *ptr) override;
};

}  // namespace cpu

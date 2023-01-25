/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <common/AllocatorInterface.hpp>

#include <cstdlib>
#include <functional>
#include <memory>

namespace arrayfire {
namespace cuda {
float getMemoryPressure();
float getMemoryPressureThreshold();

template<typename T>
void memFree(T *ptr);

template<typename T>
using uptr = std::unique_ptr<T[], std::function<void(T[])>>;

template<typename T>
uptr<T> memAlloc(const size_t &elements);

void *memAllocUser(const size_t &bytes);

// Need these as 2 separate function and not a default argument
// This is because it is used as the deleter in shared pointer
// which cannot support default arguments

void memFreeUser(void *ptr);

void memLock(const void *ptr);
void memUnlock(const void *ptr);
bool isLocked(const void *ptr);

template<typename T>
T *pinnedAlloc(const size_t &elements);
template<typename T>
void pinnedFree(T *ptr);

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes, size_t *lock_buffers);
void signalMemoryCleanup();
void shutdownMemoryManager();
void pinnedGarbageCollect();

void printMemInfo(const char *msg, const int device);

float getMemoryPressure();
float getMemoryPressureThreshold();
bool jitTreeExceedsMemoryPressure(size_t bytes);
void setMemStepSize(size_t step_bytes);
size_t getMemStepSize(void);

class Allocator final : public arrayfire::common::AllocatorInterface {
   public:
    Allocator();
    ~Allocator() = default;
    void shutdown() override;
    int getActiveDeviceId() override;
    size_t getMaxMemorySize(int id) override;
    void *nativeAlloc(const size_t bytes) override;
    void nativeFree(void *ptr) override;
};

// CUDA Pinned Memory does not depend on device
// So we pass 1 as numDevices to the constructor so that it creates 1 vector
// of memory_info
// When allocating and freeing, it doesn't really matter which device is active
class AllocatorPinned final : public arrayfire::common::AllocatorInterface {
   public:
    AllocatorPinned();
    ~AllocatorPinned() = default;
    void shutdown() override;
    int getActiveDeviceId() override;
    size_t getMaxMemorySize(int id) override;
    void *nativeAlloc(const size_t bytes) override;
    void nativeFree(void *ptr) override;
};

}  // namespace cuda
}  // namespace arrayfire

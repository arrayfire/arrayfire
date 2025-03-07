/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <common/AllocatorInterface.hpp>

#include <sycl/sycl.hpp>

#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace arrayfire {
namespace oneapi {

template<typename T>
using bufptr =
    std::unique_ptr<sycl::buffer<T>, std::function<void(sycl::buffer<T> *)>>;

template<typename T>
bufptr<T> memAlloc(const size_t &elements);
void *memAllocUser(const size_t &bytes);

// Need these as 2 separate function and not a default argument
// This is because it is used as the deleter in shared pointer
// which cannot support default arguments
template<typename T>
void memFree(sycl::buffer<T> *ptr);
void memFreeUser(void *ptr);

template<typename T>
void memLock(const sycl::buffer<T> *ptr);

template<typename T>
void memUnlock(const sycl::buffer<T> *ptr);

bool isLocked(const void *ptr);

template<typename T>
T *pinnedAlloc(const size_t &elements);

void pinnedFree(void *ptr);

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

class Allocator final : public common::AllocatorInterface {
   public:
    Allocator();
    ~Allocator() = default;
    void shutdown() override;
    int getActiveDeviceId() override;
    size_t getMaxMemorySize(int id) override;
    void *nativeAlloc(const size_t bytes) override;
    void nativeFree(void *ptr) override;
};

class AllocatorPinned final : public common::AllocatorInterface {
   public:
    AllocatorPinned();
    ~AllocatorPinned() = default;
    void shutdown() override;
    int getActiveDeviceId() override;
    size_t getMaxMemorySize(int id) override;
    void *nativeAlloc(const size_t bytes) override;
    void nativeFree(void *ptr) override;

   private:
    std::vector<std::map<void *, void *>> pinnedMaps;
};

}  // namespace oneapi
}  // namespace arrayfire

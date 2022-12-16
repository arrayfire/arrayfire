/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memory.hpp>

#include <common/DefaultMemoryManager.hpp>
#include <common/Logger.hpp>
#include <common/half.hpp>
#include <err_cpu.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <spdlog/spdlog.h>
#include <types.hpp>
#include <af/dim4.hpp>

#include <utility>

using af::dim4;
using arrayfire::common::bytesToString;
using arrayfire::common::half;
using std::function;
using std::move;
using std::unique_ptr;

namespace arrayfire {
namespace cpu {
float getMemoryPressure() { return memoryManager().getMemoryPressure(); }
float getMemoryPressureThreshold() {
    return memoryManager().getMemoryPressureThreshold();
}

bool jitTreeExceedsMemoryPressure(size_t bytes) {
    return memoryManager().jitTreeExceedsMemoryPressure(bytes);
}

void setMemStepSize(size_t step_bytes) {
    memoryManager().setMemStepSize(step_bytes);
}

size_t getMemStepSize() { return memoryManager().getMemStepSize(); }

void signalMemoryCleanup() { memoryManager().signalMemoryCleanup(); }

void shutdownMemoryManager() { memoryManager().shutdown(); }

void printMemInfo(const char *msg, const int device) {
    memoryManager().printInfo(msg, device);
}

template<typename T>
unique_ptr<T[], function<void(T *)>> memAlloc(const size_t &elements) {
    // TODO: make memAlloc aware of array shapes
    dim4 dims(elements);
    T *ptr = static_cast<T *>(
        memoryManager().alloc(false, 1, dims.get(), sizeof(T)));
    return unique_ptr<T[], function<void(T *)>>(ptr, memFree<T>);
}

void *memAllocUser(const size_t &bytes) {
    dim4 dims(bytes);
    void *ptr = memoryManager().alloc(true, 1, dims.get(), 1);
    return ptr;
}

template<typename T>
void memFree(T *ptr) {
    return memoryManager().unlock(static_cast<void *>(ptr), false);
}

void memFreeUser(void *ptr) { memoryManager().unlock(ptr, true); }

void memLock(const void *ptr) { memoryManager().userLock(ptr); }

bool isLocked(const void *ptr) { return memoryManager().isUserLocked(ptr); }

void memUnlock(const void *ptr) { memoryManager().userUnlock(ptr); }

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes, size_t *lock_buffers) {
    memoryManager().usageInfo(alloc_bytes, alloc_buffers, lock_bytes,
                              lock_buffers);
}

template<typename T>
T *pinnedAlloc(const size_t &elements) {
    // TODO: make pinnedAlloc aware of array shapes
    dim4 dims(elements);
    void *ptr = memoryManager().alloc(false, 1, dims.get(), sizeof(T));
    return static_cast<T *>(ptr);
}

template<typename T>
void pinnedFree(T *ptr) {
    memoryManager().unlock(static_cast<void *>(ptr), false);
}

#define INSTANTIATE(T)                                                \
    template std::unique_ptr<T[], std::function<void(T *)>> memAlloc( \
        const size_t &elements);                                      \
    template void memFree(T *ptr);                                    \
    template T *pinnedAlloc(const size_t &elements);                  \
    template void pinnedFree(T *ptr);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(char)
INSTANTIATE(uchar)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

Allocator::Allocator() { logger = common::loggerFactory("mem"); }

void Allocator::shutdown() {
    for (int n = 0; n < cpu::getDeviceCount(); n++) {
        try {
            cpu::setDevice(n);
            shutdownMemoryManager();
        } catch (const AfError &err) {
            continue;  // Do not throw any errors while shutting down
        }
    }
}

int Allocator::getActiveDeviceId() {
    return static_cast<int>(cpu::getActiveDeviceId());
}

size_t Allocator::getMaxMemorySize(int id) {
    return cpu::getDeviceMemorySize(id);
}

void *Allocator::nativeAlloc(const size_t bytes) {
    void *ptr = malloc(bytes);  // NOLINT(hicpp-no-malloc)
    AF_TRACE("nativeAlloc: {:>7} {}", bytesToString(bytes), ptr);
    if (!ptr) { AF_ERROR("Unable to allocate memory", AF_ERR_NO_MEM); }
    return ptr;
}

void Allocator::nativeFree(void *ptr) {
    AF_TRACE("nativeFree: {: >8} {}", " ", ptr);
    // Make sure this pointer is not being used on the queue before freeing the
    // memory.
    getQueue().sync();
    free(ptr);  // NOLINT(hicpp-no-malloc)
}
}  // namespace cpu
}  // namespace arrayfire

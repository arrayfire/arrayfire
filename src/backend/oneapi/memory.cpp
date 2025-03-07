/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/Logger.hpp>
#include <common/MemoryManagerBase.hpp>
#include <common/half.hpp>
#include <err_oneapi.hpp>
#include <errorcodes.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <spdlog/spdlog.h>
#include <types.hpp>
#include <af/dim4.hpp>

#include <sycl/sycl.hpp>

#include <utility>

using arrayfire::common::bytesToString;

using af::dim4;
using std::function;
using std::move;
using std::unique_ptr;

namespace arrayfire {
namespace oneapi {
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

void shutdownPinnedMemoryManager() { pinnedMemoryManager().shutdown(); }

void printMemInfo(const char *msg, const int device) {
    memoryManager().printInfo(msg, device);
}

template<typename T>
// unique_ptr<cl::Buffer, function<void(cl::Buffer *)>> memAlloc(
// unique_ptr<int, function<void(int *)>> memAlloc(
std::unique_ptr<sycl::buffer<T>, std::function<void(sycl::buffer<T> *)>>
memAlloc(const size_t &elements) {
    if (elements) {
        dim4 dims(elements);

        // The alloc function returns a pointer to a buffer<std::byte> object.
        // We need to reinterpret that object into buffer<T> while keeping the
        // same pointer value for memory accounting purposes. We acheive this
        // assigning the renterpreted buffer back into the original pointer.
        // This would delete the buffer<std::byte> object and replace it with
        // the buffer<T> object. We do the reverse in the memFree function
        auto *ptr = static_cast<sycl::buffer<std::byte> *>(
            memoryManager().alloc(false, 1, dims.get(), sizeof(T)));
        sycl::buffer<T> *optr = static_cast<sycl::buffer<T> *>((void *)ptr);
        size_t bytes          = ptr->byte_size();

        // TODO(umar): This could be a DANGEROUS function becasue we are calling
        // delete on the reniterpreted buffer<T> instead of the orignal
        // buffer<byte> object
        *optr = ptr->template reinterpret<T>(sycl::range(bytes / sizeof(T)));
        return unique_ptr<sycl::buffer<T>, function<void(sycl::buffer<T> *)>>(
            optr, memFree<T>);
    } else {
        return unique_ptr<sycl::buffer<T>, function<void(sycl::buffer<T> *)>>(
            nullptr, memFree<T>);
    }
}

void *memAllocUser(const size_t &bytes) {
    dim4 dims(bytes);
    void *ptr = memoryManager().alloc(true, 1, dims.get(), 1);
    return ptr;
}

template<typename T>
void memFree(sycl::buffer<T> *ptr) {
    if (ptr) {
        sycl::buffer<std::byte> *optr =
            static_cast<sycl::buffer<std::byte> *>((void *)ptr);
        size_t bytes = ptr->byte_size();
        *optr        = ptr->template reinterpret<std::byte>(sycl::range(bytes));
        memoryManager().unlock(optr, false);
    }
}

void memFreeUser(void *ptr) { memoryManager().unlock(ptr, true); }

template<typename T>
void memLock(const sycl::buffer<T> *ptr) {
    memoryManager().userLock(static_cast<const void *>(ptr));
}

template<typename T>
void memUnlock(const sycl::buffer<T> *ptr) {
    memoryManager().userUnlock(static_cast<const void *>(ptr));
}

bool isLocked(const void *ptr) {
    return memoryManager().isUserLocked(const_cast<void *>(ptr));
}

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes, size_t *lock_buffers) {
    memoryManager().usageInfo(alloc_bytes, alloc_buffers, lock_bytes,
                              lock_buffers);
}

template<typename T>
T *pinnedAlloc(const size_t &elements) {
    // TODO: make pinnedAlloc aware of array shapes
    dim4 dims(elements);
    void *ptr = pinnedMemoryManager().alloc(false, 1, dims.get(), sizeof(T));
    return static_cast<T *>(ptr);
}

void pinnedFree(void *ptr) { pinnedMemoryManager().unlock(ptr, false); }

// template unique_ptr<int, function<void(int *)>> memAlloc<T>(
#define INSTANTIATE(T)                                               \
    template std::unique_ptr<sycl::buffer<T>,                        \
                             std::function<void(sycl::buffer<T> *)>> \
    memAlloc(const size_t &elements);                                \
    template T *pinnedAlloc(const size_t &elements);                 \
    template void memLock(const sycl::buffer<T> *buf);               \
    template void memUnlock(const sycl::buffer<T> *buf);

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
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(arrayfire::common::half)
INSTANTIATE(int64_t)

template<>
void *pinnedAlloc<void>(const size_t &elements) {
    // TODO: make pinnedAlloc aware of array shapes
    dim4 dims(elements);
    void *ptr = pinnedMemoryManager().alloc(false, 1, dims.get(), 1);
    return ptr;
}

Allocator::Allocator() { logger = common::loggerFactory("mem"); }

void Allocator::shutdown() { shutdownMemoryManager(); }

int Allocator::getActiveDeviceId() { return oneapi::getActiveDeviceId(); }

size_t Allocator::getMaxMemorySize(int id) {
    return oneapi::getDeviceMemorySize(id);
}

void *Allocator::nativeAlloc(const size_t bytes) {
    auto *ptr = new sycl::buffer<std::byte>(sycl::range(bytes));
    AF_TRACE("nativeAlloc: {} {}", bytesToString(bytes),
             static_cast<void *>(ptr));
    return ptr;
}

void Allocator::nativeFree(void *ptr) {
    auto *buf = static_cast<sycl::buffer<std::byte> *>(ptr);
    AF_TRACE("nativeFree:          {}", ptr);
    delete buf;
}

AllocatorPinned::AllocatorPinned() { logger = common::loggerFactory("mem"); }

void AllocatorPinned::shutdown() { shutdownPinnedMemoryManager(); }

int AllocatorPinned::getActiveDeviceId() { return oneapi::getActiveDeviceId(); }

size_t AllocatorPinned::getMaxMemorySize(int id) {
    return oneapi::getDeviceMemorySize(id);
}

void *AllocatorPinned::nativeAlloc(const size_t bytes) {
    void *ptr = NULL;
    try {
        ptr = sycl::malloc_host<unsigned char>(bytes, getQueue());
    } catch (...) {
        auto str = fmt::format("Failed to allocate device memory of size {}",
                               bytesToString(bytes));
        AF_ERROR(str, AF_ERR_NO_MEM);
    }
    AF_TRACE("Pinned::nativeAlloc: {:>7} {}", bytesToString(bytes), ptr);
    return ptr;
}

void AllocatorPinned::nativeFree(void *ptr) {
    AF_TRACE("Pinned::nativeFree:          {}", ptr);
    try {
        sycl::free(ptr, getQueue());
    } catch (...) {
        AF_ERROR("Failed to release device memory.", AF_ERR_RUNTIME);
    }
}
}  // namespace oneapi
}  // namespace arrayfire

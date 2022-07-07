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
    return unique_ptr<sycl::buffer<T>, function<void(sycl::buffer<T> *)>>(
        new sycl::buffer<T>(sycl::range(elements)), bufferFree<T>);
}

void *memAllocUser(const size_t &bytes) {
    ONEAPI_NOT_SUPPORTED("memAllocUser Not supported");
    return nullptr;

    // dim4 dims(bytes);
    // void *ptr = memoryManager().alloc(true, 1, dims.get(), 1);
    // auto buf  = static_cast<cl_mem>(ptr);
    // return new cl::Buffer(buf, true);
}

template<typename T>
void memFree(T *ptr) {
    ONEAPI_NOT_SUPPORTED("memFree Not supported");

    // cl::Buffer *buf = reinterpret_cast<cl::Buffer *>(ptr);
    // cl_mem mem      = static_cast<cl_mem>((*buf)());
    // delete buf;
    // return memoryManager().unlock(static_cast<void *>(mem), false);
}

void memFreeUser(void *ptr) {
    ONEAPI_NOT_SUPPORTED("memFreeUser Not supported");

    // cl::Buffer *buf = static_cast<cl::Buffer *>(ptr);
    // cl_mem mem      = (*buf)();
    // delete buf;
    // memoryManager().unlock(mem, true);
}

template<typename T>
sycl::buffer<T> *bufferAlloc(const size_t &bytes) {
    ONEAPI_NOT_SUPPORTED("bufferAlloc Not supported");
    return nullptr;

    // dim4 dims(bytes);
    // if (bytes) {
    //     void *ptr       = memoryManager().alloc(false, 1, dims.get(), 1);
    //     cl_mem mem      = static_cast<cl_mem>(ptr);
    //     cl::Buffer *buf = new cl::Buffer(mem, true);
    //     return buf;
    // } else {
    //     return nullptr;
    // }
}

template<typename T>
void bufferFree(sycl::buffer<T> *buf) {
    if (buf) { delete buf; }
    // if (buf) {
    //     cl_mem mem = (*buf)();
    //     delete buf;
    //     memoryManager().unlock(static_cast<void *>(mem), false);
    // }
}

template<typename T>
void memLock(const sycl::buffer<T> *ptr) {
    ONEAPI_NOT_SUPPORTED("memLock Not supported");

    // cl_mem mem = static_cast<cl_mem>((*ptr)());
    // memoryManager().userLock(static_cast<void *>(mem));
}

template<typename T>
void memUnlock(const sycl::buffer<T> *ptr) {
    ONEAPI_NOT_SUPPORTED("memUnlock Not supported");

    // cl_mem mem = static_cast<cl_mem>((*ptr)());
    // memoryManager().userUnlock(static_cast<void *>(mem));
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

template<typename T>
void pinnedFree(T *ptr) {
    pinnedMemoryManager().unlock(static_cast<void *>(ptr), false);
}

// template unique_ptr<int, function<void(int *)>> memAlloc<T>(
#define INSTANTIATE(T)                                               \
    template std::unique_ptr<sycl::buffer<T>,                        \
                             std::function<void(sycl::buffer<T> *)>> \
    memAlloc(const size_t &elements);                                \
    template void memFree(T *ptr);                                   \
    template T *pinnedAlloc(const size_t &elements);                 \
    template void pinnedFree(T *ptr);                                \
    template void bufferFree(sycl::buffer<T> *buf);                  \
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

template<>
void *pinnedAlloc<void>(const size_t &elements) {
    ONEAPI_NOT_SUPPORTED("pinnedAlloc Not supported");

    // // TODO: make pinnedAlloc aware of array shapes
    // dim4 dims(elements);
    // void *ptr = pinnedMemoryManager().alloc(false, 1, dims.get(), sizeof(T));
    return static_cast<void *>(nullptr);
}
template<>
void pinnedFree(void *ptr) {
    // pinnedMemoryManager().unlock(ptr, false);
}

Allocator::Allocator() { logger = common::loggerFactory("mem"); }

void Allocator::shutdown() {
    ONEAPI_NOT_SUPPORTED("Allocator::shutdown Not supported");

    // for (int n = 0; n < opencl::getDeviceCount(); n++) {
    //     try {
    //         opencl::setDevice(n);
    //         shutdownMemoryManager();
    //     } catch (const AfError &err) {
    //         continue;  // Do not throw any errors while shutting down
    //     }
    // }
}

int Allocator::getActiveDeviceId() {
    ONEAPI_NOT_SUPPORTED("Allocator::getActiveDeviceId Not supported");

    return 0;
    // return opencl::getActiveDeviceId();
}

size_t Allocator::getMaxMemorySize(int id) {
    ONEAPI_NOT_SUPPORTED("Allocator::getMaxMemorySize Not supported");

    return 0;
    // return opencl::getDeviceMemorySize(id);
}

void *Allocator::nativeAlloc(const size_t bytes) {
    ONEAPI_NOT_SUPPORTED("Allocator::nativeAlloc Not supported");
    return nullptr;

    // cl_int err = CL_SUCCESS;
    // auto ptr   = static_cast<void *>(clCreateBuffer(
    //     getContext()(), CL_MEM_READ_WRITE,  // NOLINT(hicpp-signed-bitwise)
    //     bytes, nullptr, &err));

    // if (err != CL_SUCCESS) {
    //     auto str = fmt::format("Failed to allocate device memory of size {}",
    //                            bytesToString(bytes));
    //     AF_ERROR(str, AF_ERR_NO_MEM);
    // }

    // AF_TRACE("nativeAlloc: {} {}", bytesToString(bytes), ptr);
    // return ptr;
}

void Allocator::nativeFree(void *ptr) {
    ONEAPI_NOT_SUPPORTED("Allocator::nativeFree Not supported");

    // cl_mem buffer = static_cast<cl_mem>(ptr);
    // AF_TRACE("nativeFree:          {}", ptr);
    // cl_int err = clReleaseMemObject(buffer);
    // if (err != CL_SUCCESS) {
    //     AF_ERROR("Failed to release device memory.", AF_ERR_RUNTIME);
    // }
}

AllocatorPinned::AllocatorPinned() { logger = common::loggerFactory("mem"); }

void AllocatorPinned::shutdown() { shutdownPinnedMemoryManager(); }

int AllocatorPinned::getActiveDeviceId() { oneapi::getActiveDeviceId(); }

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

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/Logger.hpp>
#include <common/MemoryManagerBase.hpp>
#include <common/half.hpp>
#include <err_opencl.hpp>
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
namespace opencl {
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
unique_ptr<cl::Buffer, function<void(cl::Buffer *)>> memAlloc(
    const size_t &elements) {
    // TODO: make memAlloc aware of array shapes
    if (elements) {
        dim4 dims(elements);
        void *ptr = memoryManager().alloc(false, 1, dims.get(), sizeof(T));
        auto buf  = static_cast<cl_mem>(ptr);
        cl::Buffer *bptr = new cl::Buffer(buf, true);
        return unique_ptr<cl::Buffer, function<void(cl::Buffer *)>>(bptr,
                                                                    bufferFree);
    } else {
        return unique_ptr<cl::Buffer, function<void(cl::Buffer *)>>(nullptr,
                                                                    bufferFree);
    }
}

void *memAllocUser(const size_t &bytes) {
    dim4 dims(bytes);
    void *ptr = memoryManager().alloc(true, 1, dims.get(), 1);
    auto buf  = static_cast<cl_mem>(ptr);
    return new cl::Buffer(buf, true);
}

template<typename T>
void memFree(T *ptr) {
    cl::Buffer *buf = reinterpret_cast<cl::Buffer *>(ptr);
    cl_mem mem      = static_cast<cl_mem>((*buf)());
    delete buf;
    return memoryManager().unlock(static_cast<void *>(mem), false);
}

void memFreeUser(void *ptr) {
    cl::Buffer *buf = static_cast<cl::Buffer *>(ptr);
    cl_mem mem      = (*buf)();
    delete buf;
    memoryManager().unlock(mem, true);
}

cl::Buffer *bufferAlloc(const size_t &bytes) {
    dim4 dims(bytes);
    if (bytes) {
        void *ptr       = memoryManager().alloc(false, 1, dims.get(), 1);
        cl_mem mem      = static_cast<cl_mem>(ptr);
        cl::Buffer *buf = new cl::Buffer(mem, true);
        return buf;
    } else {
        return nullptr;
    }
}

void bufferFree(cl::Buffer *buf) {
    if (buf) {
        cl_mem mem = (*buf)();
        delete buf;
        memoryManager().unlock(static_cast<void *>(mem), false);
    }
}

void memLock(const cl::Buffer *ptr) {
    cl_mem mem = static_cast<cl_mem>((*ptr)());
    memoryManager().userLock(static_cast<void *>(mem));
}

void memUnlock(const cl::Buffer *ptr) {
    cl_mem mem = static_cast<cl_mem>((*ptr)());
    memoryManager().userUnlock(static_cast<void *>(mem));
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

#define INSTANTIATE(T)                                                         \
    template unique_ptr<cl::Buffer, function<void(cl::Buffer *)>> memAlloc<T>( \
        const size_t &elements);                                               \
    template void memFree(T *ptr);                                             \
    template T *pinnedAlloc(const size_t &elements);                           \
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
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(common::half)

Allocator::Allocator() { logger = common::loggerFactory("mem"); }

void Allocator::shutdown() {
    for (int n = 0; n < opencl::getDeviceCount(); n++) {
        try {
            opencl::setDevice(n);
            shutdownMemoryManager();
        } catch (const AfError &err) {
            continue;  // Do not throw any errors while shutting down
        }
    }
}

int Allocator::getActiveDeviceId() { return opencl::getActiveDeviceId(); }

size_t Allocator::getMaxMemorySize(int id) {
    return opencl::getDeviceMemorySize(id);
}

void *Allocator::nativeAlloc(const size_t bytes) {
    cl_int err = CL_SUCCESS;
    auto ptr   = static_cast<void *>(clCreateBuffer(
          getContext()(), CL_MEM_READ_WRITE,  // NOLINT(hicpp-signed-bitwise)
          bytes, nullptr, &err));

    if (err != CL_SUCCESS) {
        auto str = fmt::format("Failed to allocate device memory of size {}",
                               bytesToString(bytes));
        AF_ERROR(str, AF_ERR_NO_MEM);
    }

    AF_TRACE("nativeAlloc: {} {}", bytesToString(bytes), ptr);
    return ptr;
}

void Allocator::nativeFree(void *ptr) {
    cl_mem buffer = static_cast<cl_mem>(ptr);
    AF_TRACE("nativeFree:          {}", ptr);
    cl_int err = clReleaseMemObject(buffer);
    if (err != CL_SUCCESS) {
        AF_ERROR("Failed to release device memory.", AF_ERR_RUNTIME);
    }
}

AllocatorPinned::AllocatorPinned() : pinnedMaps(opencl::getDeviceCount()) {
    logger = common::loggerFactory("mem");
}

void AllocatorPinned::shutdown() {
    for (int n = 0; n < opencl::getDeviceCount(); n++) {
        opencl::setDevice(n);
        shutdownPinnedMemoryManager();
        auto currIterator = pinnedMaps[n].begin();
        auto endIterator  = pinnedMaps[n].end();
        while (currIterator != endIterator) {
            pinnedMaps[n].erase(currIterator++);
        }
    }
}

int AllocatorPinned::getActiveDeviceId() { return opencl::getActiveDeviceId(); }

size_t AllocatorPinned::getMaxMemorySize(int id) {
    return opencl::getDeviceMemorySize(id);
}

void *AllocatorPinned::nativeAlloc(const size_t bytes) {
    void *ptr = NULL;

    cl_int err = CL_SUCCESS;
    auto buf   = clCreateBuffer(getContext()(), CL_MEM_ALLOC_HOST_PTR, bytes,
                                nullptr, &err);
    if (err != CL_SUCCESS) {
        AF_ERROR("Failed to allocate pinned memory.", AF_ERR_NO_MEM);
    }

    ptr = clEnqueueMapBuffer(getQueue()(), buf, CL_TRUE,
                             CL_MAP_READ | CL_MAP_WRITE, 0, bytes, 0, nullptr,
                             nullptr, &err);
    if (err != CL_SUCCESS) {
        AF_ERROR("Failed to map pinned memory", AF_ERR_RUNTIME);
    }
    AF_TRACE("Pinned::nativeAlloc: {:>7} {}", bytesToString(bytes), ptr);
    pinnedMaps[opencl::getActiveDeviceId()].emplace(ptr, new cl::Buffer(buf));
    return ptr;
}

void AllocatorPinned::nativeFree(void *ptr) {
    AF_TRACE("Pinned::nativeFree:          {}", ptr);
    int n     = opencl::getActiveDeviceId();
    auto &map = pinnedMaps[n];
    auto iter = map.find(ptr);

    if (iter != map.end()) {
        cl::Buffer *buf = map[ptr];
        if (cl_int err = getQueue().enqueueUnmapMemObject(*buf, ptr)) {
            getLogger()->warn(
                "Pinned::nativeFree: Error unmapping pinned memory({}:{}). "
                "Ignoring",
                err, getErrorMessage(err));
        }
        delete buf;
        map.erase(iter);
    }
}
}  // namespace opencl
}  // namespace arrayfire

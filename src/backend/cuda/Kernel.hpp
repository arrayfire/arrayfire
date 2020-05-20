/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/KernelInterface.hpp>

#include <EnqueueArgs.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>

#include <cstdio>

#define CU_CHECK(fn)                                                      \
    do {                                                                  \
        CUresult res = fn;                                                \
        if (res == CUDA_SUCCESS) break;                                   \
        char cu_err_msg[1024];                                            \
        const char* cu_err_name;                                          \
        const char* cu_err_string;                                        \
        cuGetErrorName(res, &cu_err_name);                                \
        cuGetErrorString(res, &cu_err_string);                            \
        snprintf(cu_err_msg, sizeof(cu_err_msg), "CU Error %s(%d): %s\n", \
                 cu_err_name, (int)(res), cu_err_string);                 \
        AF_ERROR(cu_err_msg, AF_ERR_INTERNAL);                            \
    } while (0)

namespace cuda {

struct Enqueuer {
    template<typename... Args>
    void operator()(void* ker, const EnqueueArgs& qArgs, Args... args) {
        void* params[] = {reinterpret_cast<void*>(&args)...};
        for (auto& event : qArgs.mEvents) {
            CU_CHECK(cuStreamWaitEvent(qArgs.mStream, event, 0));
        }
        CU_CHECK(cuLaunchKernel(static_cast<CUfunction>(ker), qArgs.mBlocks.x,
                                qArgs.mBlocks.y, qArgs.mBlocks.z,
                                qArgs.mThreads.x, qArgs.mThreads.y,
                                qArgs.mThreads.z, qArgs.mSharedMemSize,
                                qArgs.mStream, params, NULL));
    }
};

class Kernel
    : public common::KernelInterface<CUmodule, CUfunction, Enqueuer,
                                     CUdeviceptr> {
   public:
    using ModuleType = CUmodule;
    using KernelType = CUfunction;
    using DevPtrType = CUdeviceptr;
    using BaseClass =
        common::KernelInterface<ModuleType, KernelType, Enqueuer, DevPtrType>;

    Kernel() : BaseClass(nullptr, nullptr) {}
    Kernel(ModuleType mod, KernelType ker) : BaseClass(mod, ker) {}

    DevPtrType getDevPtr(const char* name) final;

    void copyToReadOnly(DevPtrType dst, DevPtrType src, size_t bytes) final;

    void setScalar(DevPtrType dst, int value) final;

    int getScalar(DevPtrType src) final;
};

}  // namespace cuda

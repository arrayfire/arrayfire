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

#include <backend.hpp>
#include <cl2hpp.hpp>

namespace opencl {

struct Enqueuer {
    template<typename... Args>
    void operator()(void* ker, const cl::EnqueueArgs& qArgs, Args... args) {
        auto launchOp =
            cl::KernelFunctor<Args...>(*static_cast<const cl::Kernel*>(ker));
        launchOp(qArgs, std::forward<Args>(args)...);
    }
};

class Kernel
    : public common::KernelInterface<cl::Program*, cl::Kernel*, Enqueuer,
                                     cl::Buffer*> {
   public:
    using ModuleType = cl::Program*;
    using KernelType = cl::Kernel*;
    using DevPtrType = cl::Buffer*;
    using BaseClass =
        common::KernelInterface<ModuleType, KernelType, Enqueuer, DevPtrType>;

    Kernel() : BaseClass(nullptr, nullptr) {}
    Kernel(ModuleType mod, KernelType ker) : BaseClass(mod, ker) {}

    // clang-format off
    [[deprecated("OpenCL backend doesn't need Kernel::get method")]]
    DevPtrType get(const char* name) override;
    // clang-format on

    void copyToReadOnly(DevPtrType dst, DevPtrType src, size_t bytes) override;

    void setScalar(DevPtrType dst, int value) override;

    int getScalar(DevPtrType src) override;
};

}  // namespace opencl

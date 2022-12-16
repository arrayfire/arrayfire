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
#include <common/Logger.hpp>

#include <backend.hpp>
#include <cl2hpp.hpp>
#include <string>

namespace arrayfire {
namespace opencl {
namespace kernel_logger {
inline auto getLogger() -> spdlog::logger* {
    static auto logger = common::loggerFactory("kernel");
    return logger.get();
}
}  // namespace kernel_logger

struct Enqueuer {
    template<typename... Args>
    void operator()(std::string name, cl::Kernel ker,
                    const cl::EnqueueArgs& qArgs, Args&&... args) {
        auto launchOp = cl::KernelFunctor<Args...>(ker);
        using namespace kernel_logger;
        AF_TRACE("Launching {}", name);
        launchOp(qArgs, std::forward<Args>(args)...);
    }
};

class Kernel
    : public common::KernelInterface<const cl::Program*, cl::Kernel, Enqueuer,
                                     cl::Buffer*> {
   public:
    using ModuleType = const cl::Program*;
    using KernelType = cl::Kernel;
    using DevPtrType = cl::Buffer*;
    using BaseClass =
        common::KernelInterface<ModuleType, KernelType, Enqueuer, DevPtrType>;

    Kernel() : BaseClass("", nullptr, cl::Kernel{nullptr, false}) {}
    Kernel(std::string name, ModuleType mod, KernelType ker)
        : BaseClass(name, mod, ker) {}

    // clang-format off
    [[deprecated("OpenCL backend doesn't need Kernel::getDevPtr method")]]
    DevPtrType getDevPtr(const char* name) final;
    // clang-format on

    void copyToReadOnly(DevPtrType dst, DevPtrType src, size_t bytes) final;

    void setFlag(DevPtrType dst, int* scalarValPtr,
                 const bool syncCopy = false) final;

    int getFlag(DevPtrType src) final;
};

}  // namespace opencl
}  // namespace arrayfire

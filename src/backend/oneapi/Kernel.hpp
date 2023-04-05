/*******************************************************
 * Copyright (c) 2022, ArrayFire
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
#include <sycl/sycl.hpp>
#include <string>

namespace arrayfire {
namespace oneapi {
namespace kernel_logger {
inline auto getLogger() -> spdlog::logger* {
    static auto logger = common::loggerFactory("kernel");
    return logger.get();
}
}  // namespace kernel_logger

/*
 */
struct Enqueuer {
    template<typename... Args>
    void operator()(std::string name, sycl::kernel ker, const Enqueuer& qArgs,
                    Args&&... args) {
        // auto launchOp = cl::KernelFunctor<Args...>(ker);
        using namespace kernel_logger;
        AF_TRACE("Launching {}", name);
        // launchOp(qArgs, std::forward<Args>(args)...);
    }
};

class Kernel {
    //   public:
    //    using BaseClass =
    //      common::KernelInterface<ModuleType, KernelType, Enqueuer,
    //      sycl::buffer<float>*>;
    //
    //  Kernel() : {}
    //    Kernel(std::string name, ModuleType mod, KernelType ker)
    //        : BaseClass(name, mod, ker) {}
    //
    //    // clang-format off
    //    [[deprecated("OpenCL backend doesn't need Kernel::getDevPtr method")]]
    //    DevPtrType getDevPtr(const char* name) final;
    //    // clang-format on
    //
    //    void copyToReadOnly(DevPtrType dst, DevPtrType src, size_t bytes)
    // final;
    //
    //    void setFlag(DevPtrType dst, int* scalarValPtr,
    //                 const bool syncCopy = false) final;
    //
    //    int getFlag(DevPtrType src) final;
};

}  // namespace oneapi
}  // namespace arrayfire

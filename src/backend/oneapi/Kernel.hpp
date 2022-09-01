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
#include <CL/sycl.hpp>
#include <string>

namespace oneapi {
namespace kernel_logger {
inline auto getLogger() -> spdlog::logger* {
    static auto logger = common::loggerFactory("kernel");
    return logger.get();
}
}  // namespace kernel_logger

/*
struct Enqueuer {
    template<typename... Args>
    void operator()(std::string name, sycl::kernel ker,
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
    using ModuleType = const sycl::program*;
    using KernelType = sycl::kernel;
    using DevPtrType<T> = sycl::buffer<T>*;
    using BaseClass =
        common::KernelInterface<ModuleType, KernelType, Enqueuer, DevPtrType<T>>;

    Kernel() : BaseClass("", nullptr, cl::Kernel{nullptr, false}) {}
    Kernel(std::string name, ModuleType mod, KernelType ker)
        : BaseClass(name, mod, ker) {}

    // clang-format off
    [[deprecated("OpenCL backend doesn't need Kernel::getDevPtr method")]]
    DevPtrType<T> getDevPtr(const char* name) final;
    // clang-format on

    void copyToReadOnly(DevPtrType<T> dst, DevPtrType<T> src, size_t bytes) final;

    void setFlag(DevPtrType<T> dst, int* scalarValPtr,
                 const bool syncCopy = false) final;

    int getFlag(DevPtrType<T> src) final;
};
*/

class Kernel {
   public:
    using ModuleType = const sycl::kernel_bundle<sycl::bundle_state::executable> *;
    using KernelType = sycl::kernel;
  template<typename T>
    using DevPtrType = sycl::buffer<T>*;
    //using BaseClass =
        //common::KernelInterface<ModuleType, KernelType, Enqueuer, DevPtrType<T>>;

    Kernel() {}
    Kernel(std::string name, ModuleType mod, KernelType ker) {}

    template<typename T>
    void copyToReadOnly(DevPtrType<T> dst, DevPtrType<T> src, size_t bytes);

    template<typename T>
    void setFlag(DevPtrType<T> dst, int* scalarValPtr,
                 const bool syncCopy = false);

    template<typename T>
    int getFlag(DevPtrType<T> src);
};

}  // namespace oneapi

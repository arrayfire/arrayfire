/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cl2hpp.hpp>

namespace opencl {

using ModuleType  = cl::Program*;
using KernelType  = cl::Kernel*;
using DevPtrType  = cl::Buffer*;
using EnqueueArgs = cl::EnqueueArgs;

struct Enqueuer {
    template<typename... Args>
    void operator()(void* ker, const cl::EnqueueArgs& qArgs, Args... args) {
        auto launchOp =
            cl::KernelFunctor<Args...>(*static_cast<const cl::Kernel*>(ker));
        launchOp(qArgs, std::forward<Args>(args)...);
    }
};

}  // namespace opencl

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>  // This header has the declaration of structures
                      // that are passed onto kernel. Operator overloads
                      // for creating Param objects from opencl::Array<T>
                      // objects is automatic, no special work is needed.
                      // Hence, the OpenCL kernel wrapper function takes in
                      // Param instead of opencl::Array<T>

#include <kernel_headers/example.hpp>  // This is the header that gets auto-generated
// from the .cl file you will create. We pre-process
// cl files to obfuscate code.

#include <traits.hpp>

#include <common/dispatch.hpp>      // common utility header for CUDA & OpenCL
#include <common/kernel_cache.hpp>  // Has getKernel
                                    // backends has the divup macro

#include <debug_opencl.hpp>  // For Debug only related OpenCL validations

// Following c++ standard library headers are needed to create
// the lists of parameters for common::getKernel function call
#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;

template<typename T>
void exampleFunc(Param c, const Param a, const Param b, const af_someenum_t p) {
    // Compilation options for compiling OpenCL kernel.
    // Go to common/kernel_cache.hpp to find details on this.
    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
    };

    // Compilation options for compiling OpenCL kernel.
    // Go to common/kernel_cache.hpp to find details on this.
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };

    // The following templated function can take variable
    // number of template parameters and if one of them is double
    // precision, it will enable necessary constants, flags, ops
    // in opencl kernel compilation stage
    options.emplace_back(getTypeBuildDefinition<T>());

    // Fetch the Kernel functor, go to common/kernel_cache.hpp
    // to find details of this function
    auto exOp =
        common::getKernel("example", {{example_cl_src}}, targs, options);

    // configure work group parameters
    cl::NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(c.info.dims[0], THREADS_X);
    int blk_y = divup(c.info.dims[1], THREADS_Y);

    // configure global launch parameters
    cl::NDRange global(blk_x * THREADS_X, blk_y * THREADS_Y);

    // launch the kernel
    exOp(cl::EnqueueArgs(getQueue(), global, local), *c.data, c.info, *a.data,
         a.info, *b.data, b.info, (int)p);
    // Below Macro activates validations ONLY in DEBUG
    // mode as its name indicates
    CL_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/example.hpp>   // This is the header that gets auto-generated
                                        // from the .cl file you will create. We pre-process
                                        // cl files to obfuscate code.

#include <program.hpp>
#include <traits.hpp>

// Following c++ standard library headers are needed to maintain
// OpenCL cl::Kernel & cl::Program objects
#include <string>

#include <cache.hpp>                    // Has the definitions of functions such as the following
                                        // used in caching and fetching kernels.
                                        // * kernelCache - used to fetch existing kernel from cache
                                        // if any
                                        // * addKernelToCache - push new kernels into cache

#include <dispatch.hpp>                 // common utility header for CUDA & OpenCL backends
                                        // has the divup macro

#include <Param.hpp>                    // This header has the declaration of structures
                                        // that are passed onto kernel. Operator overloads
                                        // for creating Param objects from opencl::Array<T>
                                        // objects is automatic, no special work is needed.
                                        // Hence, the OpenCL kernel wrapper function takes in
                                        // Param instead of opencl::Array<T>

#include <debug_opencl.hpp>             // For Debug only related OpenCL validations

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
namespace kernel
{
static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename T>
void exampleFunc(Param c, const Param a, const Param b, const af_someenum_t p)
{
    std::string refName =
        std::string("example_") + //<kernel_function_name>_
        std::string(dtype_traits<T>::getName());
        // std::string("encode template parameters one after one");
        // If you have numericals, you can use std::to_string to convert
        // them into std::strings

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    // Make sure OpenCL kernel isn't already available before
    // compiling for given device and combination of template
    // parameters to this kernel wrapper function 'exampleFunc<T>'
    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        // You can pass any template parameters as compile options
        // to kernel the compilation step. This is equivalent of
        // having templated kernels in CUDA

        // The following option is passed to kernel compilation
        // if template parameter T is double or complex double
        // to enable FP64 extension
        if (std::is_same<T, double>::value ||
                std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        const char *ker_strs[] = {example_cl};
        const int   ker_lens[] = {example_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker = new Kernel(*entry.prog, "example");

        addKernelToCache(device, refName, entry);
    }

    // configure work group parameters
    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(c.info.dims[0], THREADS_X);
    int blk_y = divup(c.info.dims[1], THREADS_Y);

    // configure global launch parameters
    NDRange global(blk_x * THREADS_X, blk_y * THREADS_Y);

    // create a kernel functor from the cl::Kernel object
    // corresponding to the device on which current execution
    // is happending.
    auto exampleFuncOp = KernelFunctor< Buffer, KParam, Buffer, KParam,
                                        Buffer, KParam, int>(*entry.ker);

    // launch the kernel
    exampleFuncOp(EnqueueArgs(getQueue(), global, local),
                  *c.data, c.info, *a.data, a.info, *b.data, b.info, (int)p);

    // Below Macro activates validations ONLY in DEBUG
    // mode as its name indicates
    CL_DEBUG_FINISH(getQueue());
}
}
}

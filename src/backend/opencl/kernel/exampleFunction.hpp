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
#include <mutex>
#include <map>

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
using cl::make_kernel;
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
void exampleFunc(Param out, const Param in, const af_someenum_t p)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>  egProgs;
        static std::map<int, Kernel*> egKernels;

        int device = getActiveDeviceId();

        // std::call_once is used to ensure OpenCL kernels
        // are compiled only once for any given device and combination
        // of template parameters to this kernel wrapper function 'exampleFunc<T>'
        std::call_once( compileFlags[device], [device] () {

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

                Program prog;
                // below helper function 'buildProgram' uses the option string
                // we just created and compiles the kernel string
                // 'example_cl' which was created by our opencl kernel code obfuscation
                // stage
                buildProgram(prog, example_cl, example_cl_len, options.str());

                // create a cl::Program object on heap
                egProgs[device]   = new Program(prog);

                // create a cl::Kernel object on heap
                egKernels[device] = new Kernel(*egProgs[device], "example");
            });

        // configure work group parameters
        NDRange local(THREADS_X, THREADS_Y);

        int blk_x = divup(out.info.dims[0], THREADS_X);
        int blk_y = divup(out.info.dims[1], THREADS_Y);

        // configure global launch parameters
        NDRange global(blk_x * THREADS_X, blk_y * THREADS_Y);

        // create a kernel functor from the cl::Kernel object
        // corresponding to the device on which current execution
        // is happending.
        auto exampleFuncOp = make_kernel<Buffer, KParam,
                                     Buffer, KParam, int>(*egKernels[device]);

        // launch the kernel
        exampleFuncOp(EnqueueArgs(getQueue(), global, local),
                    *out.data, out.info, *in.data, in.info, (int)p);

        // Below Macro activates validations ONLY in DEBUG
        // mode as its name indicates
        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) { // Catch all cl::Errors and convert them
                              // to appropriate ArrayFire error codes
        CL_TO_AF_ERROR(err);
    }
}

}

}

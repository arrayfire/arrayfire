/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/moments.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <type_util.hpp>
#include <math.hpp>
#include "config.hpp"

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
        static const int THREADS = 128;

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename T>
        void moments(Param out, const Param in, af_moment_type moment)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>  momentsProgs;
                static std::map<int, Kernel*> momentsKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T="        << dtype_traits<T>::getName();

                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }


                    Program prog;
                    buildProgram(prog, moments_cl, moments_cl_len, options.str());
                    momentsProgs[device] = new Program(prog);

                    momentsKernels[device] = new Kernel(*momentsProgs[device], "moments_kernel");
                });


                auto momentsp = KernelFunctor<Buffer, const KParam, const Buffer, const KParam, const int, const int, const int>
                                      (*momentsKernels[device]);

                NDRange local(THREADS, 1, 1);
                dim_t blocksMatX = divup(in.info.dims[0], local[0]);
                NDRange global(in.info.dims[1] * local[0] ,
                               in.info.dims[2] * in.info.dims[3] * local[1] );

                bool pBatch = !(in.info.dims[2] == 1 && in.info.dims[3] == 1);

                momentsp(EnqueueArgs(getQueue(), global, local),
                          *out.data, out.info, *in.data, in.info,
                          (int)moment, blocksMatX, (int)pBatch);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

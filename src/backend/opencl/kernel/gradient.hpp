/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/gradient.hpp>
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
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const int TX = 32;
        static const int TY = 8;

        template<typename T>
        void gradient(Param grad0, Param grad1, const Param in)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>  gradProgs;
                static std::map<int, Kernel*> gradKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    ToNum<T> toNum;
                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D TX=" << TX
                            << " -D TY=" << TY
                            << " -D ZERO=" << toNum(scalar<T>(0));

                    if((af_dtype) dtype_traits<T>::af_type == c32 ||
                       (af_dtype) dtype_traits<T>::af_type == c64) {
                        options << " -D CPLX=1";
                    } else {
                        options << " -D CPLX=0";
                    }
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    Program prog;
                    buildProgram(prog, gradient_cl, gradient_cl_len, options.str());
                    gradProgs[device]   = new Program(prog);
                    gradKernels[device] = new Kernel(*gradProgs[device], "gradient_kernel");
                });

                auto gradOp = make_kernel<Buffer, const KParam, Buffer, const KParam,
                                    const Buffer, const KParam, const int, const int>
                                        (*gradKernels[device]);

                NDRange local(TX, TY, 1);

                int blocksPerMatX = divup(in.info.dims[0], TX);
                int blocksPerMatY = divup(in.info.dims[1], TY);
                NDRange global(local[0] * blocksPerMatX * in.info.dims[2],
                               local[1] * blocksPerMatY * in.info.dims[3],
                               1);

                gradOp(EnqueueArgs(getQueue(), global, local),
                        *grad0.data, grad0.info, *grad1.data, grad1.info,
                        *in.data, in.info, blocksPerMatX, blocksPerMatY);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

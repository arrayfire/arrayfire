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
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

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
        static const dim_type TX = 32;
        static const dim_type TY = 8;

        template<typename T>
        void gradient(Param grad0, Param grad1, const Param in)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static Program           gradProgs[DeviceManager::MAX_DEVICES];
                static Kernel          gradKernels[DeviceManager::MAX_DEVICES];

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D TX=" << TX
                            << " -D TY=" << TY;

                    if((af_dtype) dtype_traits<T>::af_type == c32 ||
                       (af_dtype) dtype_traits<T>::af_type == c64) {
                        options << " -D CPLX=1";
                    } else {
                        options << " -D CPLX=0";
                    }

                    buildProgram(gradProgs[device],
                                 gradient_cl,
                                 gradient_cl_len,
                                 options.str());

                    gradKernels[device] = Kernel(gradProgs[device], "gradient_kernel");
                });

                auto gradOp = make_kernel<Buffer, const KParam, Buffer, const KParam,
                                    const Buffer, const KParam, const dim_type, const dim_type>
                                        (gradKernels[device]);

                NDRange local(TX, TY, 1);

                dim_type blocksPerMatX = divup(in.info.dims[0], TX);
                dim_type blocksPerMatY = divup(in.info.dims[1], TY);
                NDRange global(local[0] * blocksPerMatX * in.info.dims[2],
                               local[1] * blocksPerMatY * in.info.dims[3],
                               1);

                gradOp(EnqueueArgs(getQueue(), global, local),
                        grad0.data, grad0.info, grad1.data, grad1.info,
                        in.data, in.info, blocksPerMatX, blocksPerMatY);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

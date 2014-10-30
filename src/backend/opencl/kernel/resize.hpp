/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/resize.hpp>
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
        static const dim_type TX = 16;
        static const dim_type TY = 16;

        template<typename T, af_interp_type method>
        void resize(Param out, const Param in)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static Program         resizeProgs[DeviceManager::MAX_DEVICES];
                static Kernel        resizeKernels[DeviceManager::MAX_DEVICES];

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T="        << dtype_traits<T>::getName();

                    switch(method) {
                        case AF_INTERP_NEAREST:  options<<" -D INTERP=NEAREST";  break;
                        case AF_INTERP_BILINEAR: options<<" -D INTERP=BILINEAR"; break;
                        default: break;
                    }

                    buildProgram(resizeProgs[device],
                                 resize_cl,
                                 resize_cl_len,
                                 options.str());

                    resizeKernels[device] = Kernel(resizeProgs[device], "resize_kernel");
                });

                auto resizeOp = make_kernel<Buffer, const KParam,
                                      const Buffer, const KParam,
                                      const dim_type, const float, const float>
                                      (resizeKernels[device]);

                NDRange local(TX, TY, 1);

                dim_type blocksPerMatX = divup(out.info.dims[0], local[0]);
                dim_type blocksPerMatY = divup(out.info.dims[1], local[1]);
                NDRange global(local[0] * blocksPerMatX * in.info.dims[2],
                               local[1] * blocksPerMatY,
                               1);

                double xd = (double)in.info.dims[0] / (double)out.info.dims[0];
                double yd = (double)in.info.dims[1] / (double)out.info.dims[1];

                float xf = (float)xd, yf = (float)yd;

                resizeOp(EnqueueArgs(getQueue(), global, local),
                         out.data, out.info, in.data, in.info, blocksPerMatX, xf, yf);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

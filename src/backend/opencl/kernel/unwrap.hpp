/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/unwrap.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <map>
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
        template<typename T, int TX>
        void unwrap(Param out, const Param in, const dim_t wx, const dim_t wy,
                    const dim_t sx, const dim_t sy, const dim_t px, const dim_t py, const dim_t nx)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>   unwrapProgs;
                static std::map<int, Kernel *> unwrapKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T="        << dtype_traits<T>::getName();
                    options << " -D TX="       << TX;

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
                    buildProgram(prog, unwrap_cl, unwrap_cl_len, options.str());
                    unwrapProgs[device] = new Program(prog);
                    unwrapKernels[device] = new Kernel(*unwrapProgs[device], "unwrap_kernel");
                });

                auto unwrapOp = make_kernel<Buffer, const KParam, const Buffer, const KParam,
                                      const dim_t, const dim_t, const dim_t, const dim_t,
                                      const dim_t, const dim_t, const dim_t, const dim_t>
                                      (*unwrapKernels[device]);

                const dim_t TY = 256 / TX;
                dim_t repsPerColumn = 1;
                if(TX == 256 && wx * wy > 256) {
                    repsPerColumn = divup((wx * wy), 256);
                }

                NDRange local(TX, TY, 1);

                NDRange global(local[0] * divup(out.info.dims[1], TY),
                               local[1] * out.info.dims[2] * out.info.dims[3],
                               1);

                unwrapOp(EnqueueArgs(getQueue(), global, local),
                       *out.data, out.info, *in.data, in.info, wx, wy, sx, sy, px, py, nx, repsPerColumn);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

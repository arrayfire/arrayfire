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
        template<typename T, bool is_column>
        void unwrap(Param out, const Param in, const dim_t wx, const dim_t wy,
                    const dim_t sx, const dim_t sy, const dim_t px, const dim_t py, const dim_t nx)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>   unwrapProgs;
                static std::map<int, Kernel *> unwrapKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {

                        ToNum<T> toNum;
                        std::ostringstream options;
                        options << " -D is_column=" << is_column
                                << " -D ZERO=" << toNum(scalar<T>(0))
                                << " -D T="    << dtype_traits<T>::getName();

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

                dim_t TX = 1, TY = 1;
                dim_t BX = 1;
                const dim_t BY = out.info.dims[2] * out.info.dims[3];
                dim_t reps = 1;

                if (is_column) {
                    TX = std::min(THREADS_PER_GROUP, nextpow2(out.info.dims[0]));
                    TY = THREADS_PER_GROUP / TX;
                    BX = divup(out.info.dims[1], TY);
                    reps = divup((wx * wy), TX);
                } else {
                    TX = THREADS_X;
                    TY = THREADS_Y;
                    BX = divup(out.info.dims[0], TX);
                    reps = divup((wx * wy), TY);
                }

                NDRange local(TX, TY);
                NDRange global(local[0] * BX,
                               local[1] * BY);

                unwrapOp(EnqueueArgs(getQueue(), global, local),
                       *out.data, out.info, *in.data, in.info, wx, wy, sx, sy, px, py, nx, reps);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

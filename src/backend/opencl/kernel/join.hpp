/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/join.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
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
        static const int TX = 32;
        static const int TY = 8;
        static const int TILEX = 256;
        static const int TILEY = 32;

        template<typename To, typename Ti, int dim>
        void join(Param out, const Param in, const af::dim4 offset)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>   joinProgs;
                static std::map<int, Kernel *> joinKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D To=" << dtype_traits<To>::getName()
                            << " -D Ti=" << dtype_traits<Ti>::getName()
                            << " -D dim=" << dim;

                    if (std::is_same<To, double>::value ||
                        std::is_same<To, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    } else if (std::is_same<Ti, double>::value ||
                               std::is_same<Ti, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    Program prog;
                    buildProgram(prog, join_cl, join_cl_len, options.str());
                    joinProgs[device] = new Program(prog);
                    joinKernels[device] = new Kernel(*joinProgs[device], "join_kernel");
                });

                auto joinOp = make_kernel<Buffer, const KParam, const Buffer, const KParam,
                              const int, const int, const int, const int,
                              const int, const int> (*joinKernels[device]);

                NDRange local(TX, TY, 1);

                int blocksPerMatX = divup(in.info.dims[0], TILEX);
                int blocksPerMatY = divup(in.info.dims[1], TILEY);
                NDRange global(local[0] * blocksPerMatX * in.info.dims[2],
                               local[1] * blocksPerMatY * in.info.dims[3],
                               1);

                joinOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info, *in.data, in.info,
                        offset[0], offset[1], offset[2], offset[3], blocksPerMatX, blocksPerMatY);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

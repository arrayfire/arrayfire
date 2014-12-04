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
        static const dim_type TX = 32;
        static const dim_type TY = 8;
        static const dim_type TILEX = 256;
        static const dim_type TILEY = 32;

        template<typename Tx, typename Ty, int dim>
        void join(Param out, const Param X, const Param Y)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>   joinProgs;
                static std::map<int, Kernel *> joinKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D Tx=" << dtype_traits<Tx>::getName()
                            << " -D Ty=" << dtype_traits<Ty>::getName()
                            << " -D dim=" << dim;

                    if (std::is_same<Tx, double>::value ||
                        std::is_same<Tx, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    } else if (std::is_same<Tx, double>::value ||
                        std::is_same<Tx, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    Program prog;
                    buildProgram(prog, join_cl, join_cl_len, options.str());
                    joinProgs[device] = new Program(prog);
                    joinKernels[device] = new Kernel(*joinProgs[device], "join_kernel");
                });

                auto joinOp = make_kernel<Buffer, const KParam, const Buffer, const KParam,
                              const Buffer, const KParam, const dim_type, const dim_type> (*joinKernels[device]);

                NDRange local(TX, TY, 1);

                dim_type blocksPerMatX = divup(out.info.dims[0], TILEX);
                dim_type blocksPerMatY = divup(out.info.dims[1], TILEY);
                NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                               local[1] * blocksPerMatY * out.info.dims[3],
                               1);

                joinOp(EnqueueArgs(getQueue(), global, local), out.data, out.info,
                        X.data, X.info, Y.data, Y.info, blocksPerMatX, blocksPerMatY);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

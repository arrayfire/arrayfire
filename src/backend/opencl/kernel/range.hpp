/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/range.hpp>
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
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const int RANGE_TX = 32;
        static const int RANGE_TY = 8;
        static const int RANGE_TILEX = 512;
        static const int RANGE_TILEY = 32;

        template<typename T>
        void range(Param out, const int dim)
        {
            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static std::map<int, Program*>  rangeProgs;
            static std::map<int, Kernel*> rangeKernels;

            int device = getActiveDeviceId();

            std::call_once( compileFlags[device], [device] () {
                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName();
                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }
                Program prog;
                buildProgram(prog, range_cl, range_cl_len, options.str());
                rangeProgs[device]   = new Program(prog);
                rangeKernels[device] = new Kernel(*rangeProgs[device], "range_kernel");
            });

            auto rangeOp = KernelFunctor<Buffer, const KParam, const int,
                                        const int, const int> (*rangeKernels[device]);

            NDRange local(RANGE_TX, RANGE_TY, 1);

            int blocksPerMatX = divup(out.info.dims[0], RANGE_TILEX);
            int blocksPerMatY = divup(out.info.dims[1], RANGE_TILEY);
            NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                            local[1] * blocksPerMatY * out.info.dims[3],
                            1);

            rangeOp(EnqueueArgs(getQueue(), global, local),
                    *out.data, out.info, dim, blocksPerMatX, blocksPerMatY);

            CL_DEBUG_FINISH(getQueue());
        }
    }
}

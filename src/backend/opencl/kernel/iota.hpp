/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/iota.hpp>
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
        static const int TILEX = 512;
        static const int TILEY = 32;

        template<typename T>
        void iota(Param out, const dim4 &sdims, const dim4 &tdims)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>  iotaProgs;
                static std::map<int, Kernel*> iotaKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName();
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    Program prog;
                    buildProgram(prog, iota_cl, iota_cl_len, options.str());
                    iotaProgs[device]   = new Program(prog);
                    iotaKernels[device] = new Kernel(*iotaProgs[device], "iota_kernel");
                });

                auto iotaOp = make_kernel<Buffer, const KParam,
                                          const int, const int, const int, const int,
                                          const int, const int, const int, const int,
                                          const int, const int> (*iotaKernels[device]);

                NDRange local(TX, TY, 1);

                int blocksPerMatX = divup(out.info.dims[0], TILEX);
                int blocksPerMatY = divup(out.info.dims[1], TILEY);
                NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                               local[1] * blocksPerMatY * out.info.dims[3],
                               1);

                iotaOp(EnqueueArgs(getQueue(), global, local),
                       *out.data, out.info, sdims[0], sdims[1], sdims[2], sdims[3],
                       tdims[0], tdims[1], tdims[2], tdims[3], blocksPerMatX, blocksPerMatY);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

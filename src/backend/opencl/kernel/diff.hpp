/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/diff.hpp>
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
        static const int TX = 16;
        static const int TY = 16;

        template<typename T, unsigned dim, bool isDiff2>
        void diff(Param out, const Param in, const unsigned indims)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>   diffProgs;
                static std::map<int, Kernel*>  diffKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T="        << dtype_traits<T>::getName()
                            << " -D DIM="      << dim
                            << " -D isDiff2=" << isDiff2;
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    Program prog;
                    buildProgram(prog, diff_cl, diff_cl_len, options.str());
                    diffProgs[device]   = new Program(prog);
                    diffKernels[device] = new Kernel(*diffProgs[device], "diff_kernel");
                });

                auto diffOp = make_kernel<Buffer, const Buffer, const KParam, const KParam,
                                          const int, const int, const int>
                                          (*diffKernels[device]);

                NDRange local(TX, TY, 1);
                if(dim == 0 && indims == 1) {
                    local = NDRange(TX * TY, 1, 1);
                }

                int blocksPerMatX = divup(out.info.dims[0], local[0]);
                int blocksPerMatY = divup(out.info.dims[1], local[1]);
                NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                               local[1] * blocksPerMatY * out.info.dims[3],
                               1);

                const int oElem = out.info.dims[0] * out.info.dims[1]
                                     * out.info.dims[2] * out.info.dims[3];

                diffOp(EnqueueArgs(getQueue(), global, local),
                       *out.data, *in.data, out.info, in.info,
                       oElem, blocksPerMatX, blocksPerMatY);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

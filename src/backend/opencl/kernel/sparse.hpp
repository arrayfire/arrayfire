/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/sparse.hpp>
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
        static const int TX = 16;
        static const int TY = 16;
        static const int THREADS = 256;
        static const int reps = 4;

        template<typename T>
        void coo2dense(Param out, const Param values, const Param rowIdx, const Param colIdx)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>   coo2denseProgs;
                static std::map<int, Kernel*>  coo2denseKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T="        << dtype_traits<T>::getName()
                            << " -D reps="     << reps
                            ;

                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    Program prog;
                    buildProgram(prog, sparse_cl, sparse_cl_len, options.str());
                    coo2denseProgs[device]   = new Program(prog);
                    coo2denseKernels[device] = new Kernel(*coo2denseProgs[device], "coo2dense_kernel");
                });

                auto coo2denseOp = KernelFunctor<Buffer, const KParam,
                                           const Buffer, const KParam,
                                           const Buffer, const KParam,
                                           const Buffer, const KParam>
                                          (*coo2denseKernels[device]);

                NDRange local(THREADS, 1, 1);

                NDRange global(divup(out.info.dims[0], local[0] * reps) * THREADS, 1, 1);

                coo2denseOp(EnqueueArgs(getQueue(), global, local),
                       *out.data, out.info,
                       *values.data, values.info,
                       *rowIdx.data, rowIdx.info,
                       *colIdx.data, colIdx.info);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

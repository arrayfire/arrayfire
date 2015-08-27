/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/approx1.hpp>
#include <kernel_headers/approx2.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
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
        static const int TX = 16;
        static const int TY = 16;

        static const int THREADS = 256;

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename Ty, typename Tp, af_interp_type method>
        void approx1(Param out, const Param in, const Param pos, const float offGrid)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>  approxProgs;
                static std::map<int, Kernel*> approxKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    ToNum<Ty> toNum;
                    std::ostringstream options;
                    options << " -D Ty="        << dtype_traits<Ty>::getName()
                            << " -D Tp="        << dtype_traits<Tp>::getName()
                            << " -D ZERO="      << toNum(scalar<Ty>(0));

                    if((af_dtype) dtype_traits<Ty>::af_type == c32 ||
                       (af_dtype) dtype_traits<Ty>::af_type == c64) {
                        options << " -D CPLX=1";
                    } else {
                        options << " -D CPLX=0";
                    }
                    if (std::is_same<Ty, double>::value ||
                        std::is_same<Ty, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    switch(method) {
                        case AF_INTERP_NEAREST: options << " -D INTERP=NEAREST";
                            break;
                        case AF_INTERP_LINEAR:  options << " -D INTERP=LINEAR";
                            break;
                        default:
                            break;
                    }
                    Program prog;
                    buildProgram(prog, approx1_cl, approx1_cl_len, options.str());
                    approxProgs[device] = new Program(prog);

                    approxKernels[device] = new Kernel(*approxProgs[device], "approx1_kernel");
                });


                auto approx1Op = make_kernel<Buffer, const KParam, const Buffer, const KParam,
                                       const Buffer, const KParam, const float, const int>
                                      (*approxKernels[device]);

                NDRange local(THREADS, 1, 1);
                int blocksPerMat = divup(out.info.dims[0], local[0]);
                NDRange global(blocksPerMat * local[0] * out.info.dims[1],
                               out.info.dims[2] * out.info.dims[3] * local[0],
                               1);

                approx1Op(EnqueueArgs(getQueue(), global, local),
                          *out.data, out.info, *in.data, in.info,
                          *pos.data, pos.info, offGrid, blocksPerMat);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }

        template <typename Ty, typename Tp, af_interp_type method>
        void approx2(Param out, const Param in, const Param pos, const Param qos, const float offGrid)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>       approxProgs;
                static std::map<int, Kernel*>      approxKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    ToNum<Ty> toNum;
                    std::ostringstream options;
                    options << " -D Ty="        << dtype_traits<Ty>::getName()
                            << " -D Tp="        << dtype_traits<Tp>::getName()
                            << " -D ZERO="      << toNum(scalar<Ty>(0));

                    if((af_dtype) dtype_traits<Ty>::af_type == c32 ||
                       (af_dtype) dtype_traits<Ty>::af_type == c64) {
                        options << " -D CPLX=1";
                    } else {
                        options << " -D CPLX=0";
                    }
                    if (std::is_same<Ty, double>::value ||
                        std::is_same<Ty, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    switch(method) {
                        case AF_INTERP_NEAREST: options << " -D INTERP=NEAREST";
                            break;
                        case AF_INTERP_LINEAR:  options << " -D INTERP=LINEAR";
                            break;
                        default:
                            break;
                    }
                    Program prog;
                    buildProgram(prog, approx2_cl, approx2_cl_len, options.str());
                    approxProgs[device] = new Program(prog);

                    approxKernels[device] = new Kernel(*approxProgs[device], "approx2_kernel");
                });

                auto approx2Op = make_kernel<Buffer, const KParam, const Buffer, const KParam,
                                       const Buffer, const KParam, const Buffer, const KParam,
                                       const float, const int, const int>
                                       (*approxKernels[device]);

                NDRange local(TX, TY, 1);
                int blocksPerMatX = divup(out.info.dims[0], local[0]);
                int blocksPerMatY = divup(out.info.dims[1], local[1]);
                NDRange global(blocksPerMatX * local[0] * out.info.dims[2],
                               blocksPerMatY * local[1] * out.info.dims[3],
                               1);


                approx2Op(EnqueueArgs(getQueue(), global, local),
                          *out.data, out.info,
                          *in.data, in.info,
                          *pos.data, pos.info,
                          *qos.data, qos.info,
                          offGrid, blocksPerMatX, blocksPerMatY);
                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

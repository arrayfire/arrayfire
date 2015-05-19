/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel_headers/diag_create.hpp>
#include <kernel_headers/diag_extract.hpp>
#include <program.hpp>
#include "../traits.hpp"
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <map>
#include <mutex>
#include <math.hpp>
#include "config.hpp"

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;
using af::scalar_to_option;

namespace opencl
{

namespace kernel
{

    template<typename T>
    static void diagCreate(Param out, Param in, int num)
    {
        try {
            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static std::map<int, Program*>   diagCreateProgs;
            static std::map<int, Kernel*>  diagCreateKernels;

            int device = getActiveDeviceId();

            std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T="    << dtype_traits<T>::getName()
                            << " -D ZERO=(T)(" << scalar_to_option(scalar<T>(0)) << ")";
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    Program prog;
                    buildProgram(prog, diag_create_cl, diag_create_cl_len, options.str());
                    diagCreateProgs[device]   = new Program(prog);
                    diagCreateKernels[device] = new Kernel(*diagCreateProgs[device],
                                                           "diagCreateKernel");
                });

            NDRange local(32, 8);
            int groups_x = divup(out.info.dims[0], local[0]);
            int groups_y = divup(out.info.dims[1], local[1]);
            NDRange global(groups_x * local[0] * out.info.dims[2],
                           groups_y * local[1]);

            auto diagCreateOp = make_kernel<Buffer, const KParam,
                                            Buffer, const KParam,
                                            int, int> (*diagCreateKernels[device]);

            diagCreateOp(EnqueueArgs(getQueue(), global, local),
                         *(out.data), out.info, *(in.data), in.info, num, groups_x);
            CL_DEBUG_FINISH(getQueue());

        } catch (cl::Error err) {
            CL_TO_AF_ERROR(err);
        }
    }

    template<typename T>
    static void diagExtract(Param out, Param in, int num)
    {
        try {
            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static std::map<int, Program*>   diagExtractProgs;
            static std::map<int, Kernel*>  diagExtractKernels;

            int device = getActiveDeviceId();

            std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T="    << dtype_traits<T>::getName()
                            << " -D ZERO=(T)(" << scalar_to_option(scalar<T>(0)) << ")";
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    Program prog;
                    buildProgram(prog, diag_extract_cl, diag_extract_cl_len, options.str());
                    diagExtractProgs[device]   = new Program(prog);
                    diagExtractKernels[device] = new Kernel(*diagExtractProgs[device],
                                                           "diagExtractKernel");
                });

            NDRange local(256, 1);
            int groups_x = divup(out.info.dims[0], local[0]);
            int groups_z = out.info.dims[2];
            NDRange global(groups_x * local[0],
                           groups_z * local[1] * out.info.dims[3]);

            auto diagExtractOp = make_kernel<Buffer, const KParam,
                                             Buffer, const KParam,
                                             int, int> (*diagExtractKernels[device]);

            diagExtractOp(EnqueueArgs(getQueue(), global, local),
                          *(out.data), out.info, *(in.data), in.info, num, groups_z);
            CL_DEBUG_FINISH(getQueue());

        } catch (cl::Error err) {
            CL_TO_AF_ERROR(err);
        }
    }

}

}

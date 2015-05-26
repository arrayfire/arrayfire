/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/triangle.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <types.hpp>
#include <math.hpp>

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

// Kernel Launch Config Values
static const unsigned TX = 32;
static const unsigned TY = 8;
static const unsigned TILEX = 128;
static const unsigned TILEY = 32;

template<typename T, bool is_upper, bool is_unit_diag>
void triangle(Param out, const Param in)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>  trgProgs;
        static std::map<int, Kernel*> trgKernels;

        int device = getActiveDeviceId();

        std::call_once(compileFlags[device], [device] () {

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D is_upper=" << is_upper
                        << " -D is_unit_diag=" << is_unit_diag
                        << " -D ZERO=(T)(" << scalar_to_option(scalar<T>(0)) << ")"
                        << " -D ONE=(T)(" << scalar_to_option(scalar<T>(1)) << ")";

                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                cl::Program prog;
                buildProgram(prog, triangle_cl, triangle_cl_len, options.str());
                trgProgs[device] = new Program(prog);

                trgKernels[device] = new Kernel(*trgProgs[device], "triangle_kernel");
            });

        NDRange local(TX, TY);

        int groups_x = divup(out.info.dims[0], TILEX);
        int groups_y = divup(out.info.dims[1], TILEY);

        NDRange global(groups_x * out.info.dims[2] * local[0],
                       groups_y * out.info.dims[3] * local[1]);

        auto triangleOp = make_kernel<Buffer, KParam,
                                      const Buffer, KParam,
                                      const int, const int> (*trgKernels[device]);

        triangleOp(EnqueueArgs(getQueue(), global, local),
                    *out.data, out.info, *in.data, in.info, groups_x, groups_y);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}

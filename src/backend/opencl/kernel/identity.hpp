/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel_headers/identity.hpp>
#include <program.hpp>
#include <traits.hpp>
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
using std::ostringstream;
using af::scalar_to_option;

namespace opencl
{

namespace kernel
{

    template<typename T>
    static void identity(Param out)
    {
        try {
            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static std::map<int, Program*>   identityProgs;
            static std::map<int, Kernel*>  identityKernels;

            int device = getActiveDeviceId();

            std::call_once( compileFlags[device], [device] () {
                    ostringstream options;
                    options << " -D T="    << dtype_traits<T>::getName()
                            << " -D ONE=(T)("  << scalar_to_option(scalar<T>(1)) << ")"
                            << " -D ZERO=(T)(" << scalar_to_option(scalar<T>(0)) << ")";
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    Program prog;
                    buildProgram(prog, identity_cl, identity_cl_len, options.str());
                    identityProgs[device]   = new Program(prog);
                    identityKernels[device] = new Kernel(*identityProgs[device], "identity_kernel");
                });

            NDRange local(32, 8);
            int groups_x = divup(out.info.dims[0], local[0]);
            int groups_y = divup(out.info.dims[1], local[1]);
            NDRange global(groups_x * out.info.dims[2] * local[0],
                           groups_y * out.info.dims[3] * local[1]);

            auto identityOp = make_kernel<Buffer, const KParam,
                                          int, int> (*identityKernels[device]);

            identityOp(EnqueueArgs(getQueue(), global, local),
                       *(out.data), out.info, groups_x, groups_y);
            CL_DEBUG_FINISH(getQueue());

        } catch (cl::Error err) {
            CL_TO_AF_ERROR(err);
        }
    }

}

}

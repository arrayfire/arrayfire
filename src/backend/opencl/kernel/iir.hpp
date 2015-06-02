/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/iir.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <types.hpp>

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
        template<typename T, bool batch_a>
        void iir(Param y, Param c, Param a)
        {

            //FIXME: This is a temporary fix. Ideally the local memory should be allocted outside
            static const int MAX_A_SIZE = (1024 * sizeof(double)) / sizeof(T);

            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static std::map<int, Program*>  iirProgs;
            static std::map<int, Kernel*> iirKernels;

            int device = getActiveDeviceId();

            std::call_once(compileFlags[device], [device] () {

                    std::ostringstream options;
                    options << " -D MAX_A_SIZE=" << MAX_A_SIZE
                            << " -D BATCH_A=" << batch_a
                            << " -D ZERO=(T)(" << scalar_to_option(scalar<T>(0)) << ")"
                            << " -D T=" << dtype_traits<T>::getName();

                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    cl::Program prog;
                    buildProgram(prog, iir_cl, iir_cl_len, options.str());
                    iirProgs[device] = new Program(prog);

                    iirKernels[device] = new Kernel(*iirProgs[device], "iir_kernel");
                });


            const int groups_y = y.info.dims[1];
            const int groups_x = y.info.dims[2];

            int threads = 256;
            while (threads > (int)y.info.dims[0] && threads > 32) threads /= 2;


            NDRange local(threads, 1);
            NDRange global(groups_x * local[0],
                           groups_y * y.info.dims[3] * local[1]);

            auto iirOp = make_kernel<Buffer, KParam,
                                     Buffer, KParam,
                                     Buffer, KParam,
                                     int>(*iirKernels[device]);

            try {
                iirOp(EnqueueArgs(getQueue(), global, local),
                      *y.data, y.info, *c.data, c.info, *a.data, a.info, groups_y);
            } catch(cl::Error &clerr) {
                AF_ERROR("Size of a too big for this datatype",
                         AF_ERR_SIZE);
            }

            CL_DEBUG_FINISH(getQueue());
        }

    }
}

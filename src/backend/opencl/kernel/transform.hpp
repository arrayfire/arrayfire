/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/transform_interp.hpp>
#include <kernel_headers/transform.hpp>
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
        static const dim_type TX = 16;
        static const dim_type TY = 16;

        template<typename T, bool isInverse, af_interp_type method>
        void transform(Param out, const Param in, const Param tf)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>   transformProgs;
                static std::map<int, Kernel *> transformKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T="        << dtype_traits<T>::getName()
                            << " -D INVERSE="  << (isInverse ? 1 : 0);

                    if((af_dtype) dtype_traits<T>::af_type == c32 ||
                       (af_dtype) dtype_traits<T>::af_type == c64) {
                        options << " -D CPLX=1";
                    } else {
                        options << " -D CPLX=0";
                    }
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    switch(method) {
                        case AF_INTERP_NEAREST: options << " -D INTERP=NEAREST";
                            break;
                        case AF_INTERP_BILINEAR:  options << " -D INTERP=BILINEAR";
                            break;
                        default:
                            break;
                    }

                    const char *ker_strs[] = {transform_interp_cl, transform_cl};
                    const int   ker_lens[] = {transform_interp_cl_len, transform_cl_len};
                    Program prog;
                    buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                    transformProgs[device] = new Program(prog);
                    transformKernels[device] = new Kernel(*transformProgs[device], "transform_kernel");
                });

                auto transformOp = make_kernel<Buffer, const KParam,
                                         const Buffer, const KParam, const Buffer, const KParam,
                                         const dim_type, const dim_type>
                                         (*transformKernels[device]);

                const dim_type nimages = in.info.dims[2];
                // Multiplied in src/backend/transform.cpp
                const dim_type ntransforms = out.info.dims[2] / in.info.dims[2];
                NDRange local(TX, TY, 1);

                NDRange global(local[0] * divup(out.info.dims[0], local[0]),
                               local[1] * divup(out.info.dims[1], local[1]) * ntransforms,
                               1);

                transformOp(EnqueueArgs(getQueue(), global, local),
                         out.data, out.info, in.data, in.info, tf.data, tf.info, nimages, ntransforms);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

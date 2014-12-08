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
#include <kernel_headers/rotate.hpp>
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

        typedef struct {
            float tmat[6];
        } tmat_t;

        template<typename T, af_interp_type method>
        void rotate(Param out, const Param in, const float theta)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>   rotateProgs;
                static std::map<int, Kernel *> rotateKernels;

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T="        << dtype_traits<T>::getName();

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

                    const char *ker_strs[] = {transform_interp_cl, rotate_cl};
                    const int   ker_lens[] = {transform_interp_cl_len, rotate_cl_len};
                    Program prog;
                    buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                    rotateProgs[device] = new Program(prog);
                    rotateKernels[device] = new Kernel(*rotateProgs[device], "rotate_kernel");
                });

                auto rotateOp = make_kernel<Buffer, const KParam, const Buffer, const KParam,
                                             const tmat_t, const dim_type>
                                           (*rotateKernels[device]);

                const dim_type nimages = in.info.dims[2];

                const float c = cos(-theta), s = sin(-theta);
                float tx, ty;
                {
                    const float nx = 0.5 * (in.info.dims[0] - 1);
                    const float ny = 0.5 * (in.info.dims[1] - 1);
                    const float mx = 0.5 * (out.info.dims[0] - 1);
                    const float my = 0.5 * (out.info.dims[1] - 1);
                    const float sx = (mx * c + my *-s);
                    const float sy = (mx * s + my * c);
                    tx = -(sx - nx);
                    ty = -(sy - ny);
                }

                tmat_t t;
                t.tmat[0] =  c;
                t.tmat[1] = -s;
                t.tmat[2] = tx;
                t.tmat[3] =  s;
                t.tmat[4] =  c;
                t.tmat[5] = ty;

                NDRange local(TX, TY, 1);
                NDRange global(local[0] * divup(out.info.dims[0], local[0]),
                               local[1] * divup(out.info.dims[1], local[1]),
                               1);

                rotateOp(EnqueueArgs(getQueue(), global, local),
                         *out.data, out.info, *in.data, in.info, t, nimages);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

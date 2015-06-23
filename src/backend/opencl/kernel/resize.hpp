/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/resize.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <map>
#include <mutex>
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
        static const int RESIZE_TX = 16;
        static const int RESIZE_TY = 16;

        using std::conditional;
        using std::is_same;
        template<typename T>
        using wtype_t = typename conditional<is_same<T, double>::value, double, float>::type;

        template<typename T>
        using vtype_t = typename conditional<is_complex<T>::value,
                                             T, wtype_t<T>
                                            >::type;

        template<typename T, af_interp_type method>
        void resize(Param out, const Param in)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>   resizeProgs;
                static std::map<int, Kernel *> resizeKernels;

                int device = getActiveDeviceId();

                typedef typename dtype_traits<T>::base_type BT;

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T="        << dtype_traits<T>::getName();
                    options << " -D VT="       << dtype_traits<vtype_t<T>>::getName();
                    options << " -D WT="       << dtype_traits<wtype_t<BT>>::getName();

                    switch(method) {
                        case AF_INTERP_NEAREST:  options <<" -D INTERP=NEAREST" ; break;
                        case AF_INTERP_BILINEAR: options <<" -D INTERP=BILINEAR"; break;
                        case AF_INTERP_LOWER:    options <<" -D INTERP=LOWER"   ; break;
                        default: break;
                    }

                    if((af_dtype) dtype_traits<T>::af_type == c32 ||
                       (af_dtype) dtype_traits<T>::af_type == c64) {
                        options << " -D CPLX=1";
                        options << " -D TB=" << dtype_traits<BT>::getName();
                    } else {
                        options << " -D CPLX=0";
                    }

                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    Program prog;
                    buildProgram(prog, resize_cl, resize_cl_len, options.str());
                    resizeProgs[device] = new Program(prog);
                    resizeKernels[device] = new Kernel(*resizeProgs[device], "resize_kernel");
                });

                auto resizeOp = make_kernel<Buffer, const KParam,
                                      const Buffer, const KParam,
                                      const int, const int, const float, const float>
                                      (*resizeKernels[device]);

                NDRange local(RESIZE_TX, RESIZE_TY, 1);

                int blocksPerMatX = divup(out.info.dims[0], local[0]);
                int blocksPerMatY = divup(out.info.dims[1], local[1]);
                NDRange global(local[0] * blocksPerMatX * in.info.dims[2],
                               local[1] * blocksPerMatY * in.info.dims[3],
                               1);

                double xd = (double)in.info.dims[0] / (double)out.info.dims[0];
                double yd = (double)in.info.dims[1] / (double)out.info.dims[1];

                float xf = (float)xd, yf = (float)yd;

                resizeOp(EnqueueArgs(getQueue(), global, local),
                         *out.data, out.info, *in.data, in.info, blocksPerMatX, blocksPerMatY, xf, yf);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

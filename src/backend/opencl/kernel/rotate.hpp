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
        static const int TX = 16;
        static const int TY = 16;
        // Used for batching images
        static const int TI = 4;

        typedef struct {
            float tmat[6];
        } tmat_t;

        using std::conditional;
        using std::is_same;
        template<typename T>
        using wtype_t = typename conditional<is_same<T, double>::value, double, float>::type;

        template<typename T>
        using vtype_t = typename conditional<is_complex<T>::value,
                                             T, wtype_t<T>
                                            >::type;

        template<typename T, af_interp_type method>
        void rotate(Param out, const Param in, const float theta)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>   rotateProgs;
                static std::map<int, Kernel *> rotateKernels;

                int device = getActiveDeviceId();
                typedef typename dtype_traits<T>::base_type BT;

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T="        << dtype_traits<T>::getName();
                    options << " -D VT="       << dtype_traits<vtype_t<T>>::getName();
                    options << " -D WT="       << dtype_traits<wtype_t<BT>>::getName();

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

                auto rotateOp = make_kernel<Buffer, const KParam, const Buffer, const KParam, const tmat_t,
                                            const int, const int, const int, const int>
                                           (*rotateKernels[device]);

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

                // Rounding error. Anything more than 3 decimal points wont make a diff
                tmat_t t;
                t.tmat[0] = round( c * 1000) / 1000.0f;
                t.tmat[1] = round(-s * 1000) / 1000.0f;
                t.tmat[2] = round(tx * 1000) / 1000.0f;
                t.tmat[3] = round( s * 1000) / 1000.0f;
                t.tmat[4] = round( c * 1000) / 1000.0f;
                t.tmat[5] = round(ty * 1000) / 1000.0f;


                NDRange local(TX, TY, 1);

                int nimages  = in.info.dims[2];
                int nbatches = in.info.dims[3];
                int global_x = local[0] * divup(out.info.dims[0], local[0]);
                int global_y = local[1] * divup(out.info.dims[1], local[1]);
                const int blocksXPerImage = global_x / local[0];
                const int blocksYPerImage = global_y / local[1];

                if(nimages > TI) {
                    int tile_images = divup(nimages, TI);
                    nimages = TI;
                    global_x = global_x * tile_images;
                }
                global_y *= nbatches;

                NDRange global(global_x, global_y, 1);

                rotateOp(EnqueueArgs(getQueue(), global, local),
                         *out.data, out.info, *in.data, in.info, t, nimages, nbatches,
                         blocksXPerImage, blocksYPerImage);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

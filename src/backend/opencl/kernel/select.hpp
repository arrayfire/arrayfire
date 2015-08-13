/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/select.hpp>
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

namespace opencl
{
    namespace kernel
    {
        static const uint DIMX = 32;
        static const uint DIMY =  8;

        template<typename T, bool is_same>
        void select_launcher(Param out, Param cond, Param a, Param b, int ndims)
        {
            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static std::map<int, Program*>  selProgs;
            static std::map<int, Kernel*> selKernels;

            int device = getActiveDeviceId();

            std::call_once(compileFlags[device], [device] () {

                    std::ostringstream options;
                    options << " -D is_same=" << is_same
                            << " -D T=" << dtype_traits<T>::getName();

                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    cl::Program prog;
                    buildProgram(prog, select_cl, select_cl_len, options.str());
                    selProgs[device] = new Program(prog);

                    selKernels[device] = new Kernel(*selProgs[device], "select_kernel");
                });


            int threads[] = {DIMX, DIMY};

            if (ndims == 1) {
                threads[0] *= threads[1];
                threads[1] = 1;
            }

            NDRange local(threads[0],
                          threads[1]);


            int groups_0 = divup(out.info.dims[0], local[0]);
            int groups_1 = divup(out.info.dims[1], local[1]);

            NDRange global(groups_0 * out.info.dims[2] * local[0],
                           groups_1 * out.info.dims[3] * local[1]);

            auto selectOp = make_kernel<Buffer, KParam,
                                        Buffer, KParam,
                                        Buffer, KParam,
                                        Buffer, KParam,
                                        int, int>(*selKernels[device]);

            selectOp(EnqueueArgs(getQueue(), global, local),
                     *out.data, out.info,
                     *cond.data, cond.info,
                     *a.data, a.info,
                     *b.data, b.info,
                     groups_0, groups_1);

        }

        template<typename T>
        void select(Param out, Param cond, Param a, Param b, int ndims)
        {
            try {
                bool is_same = true;
                for (int i = 0; i < 4; i++) {
                    is_same &= (a.info.dims[i] == b.info.dims[i]);
                }

                if (is_same) {
                    select_launcher<T, true >(out, cond, a, b, ndims);
                } else {
                    select_launcher<T, false>(out, cond, a, b, ndims);
                }
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
            }
        }

        template<typename T, bool flip>
        void select_scalar(Param out, Param cond, Param a, const double b, int ndims)
        {
            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static std::map<int, Program*>  selProgs;
            static std::map<int, Kernel*> selKernels;

            int device = getActiveDeviceId();

            std::call_once(compileFlags[device], [device] () {

                    std::ostringstream options;
                    options << " -D flip=" << flip
                            << " -D T=" << dtype_traits<T>::getName();

                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    cl::Program prog;
                    buildProgram(prog, select_cl, select_cl_len, options.str());
                    selProgs[device] = new Program(prog);

                    selKernels[device] = new Kernel(*selProgs[device], "select_scalar_kernel");
                });


            int threads[] = {DIMX, DIMY};

            if (ndims == 1) {
                threads[0] *= threads[1];
                threads[1] = 1;
            }

            NDRange local(threads[0],
                          threads[1]);

            int groups_0 = divup(out.info.dims[0], local[0]);
            int groups_1 = divup(out.info.dims[1], local[1]);

            NDRange global(groups_0 * out.info.dims[2] * local[0],
                           groups_1 * out.info.dims[3] * local[1]);

            auto selectOp = make_kernel<Buffer, KParam,
                                        Buffer, KParam,
                                        Buffer, KParam,
                                        T,
                                        int, int>(*selKernels[device]);

            selectOp(EnqueueArgs(getQueue(), global, local),
                     *out.data, out.info,
                     *cond.data, cond.info,
                     *a.data, a.info,
                     scalar<T>(b),
                     groups_0, groups_1);
        }
    }
}

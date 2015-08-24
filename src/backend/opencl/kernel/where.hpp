/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <string>
#include <mutex>
#include <map>
#include <kernel_headers/where.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <type_util.hpp>
#include "names.hpp"
#include "config.hpp"
#include "scan_first.hpp"
#include <memory.hpp>

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

    template<typename T>
    static void get_out_idx(Buffer *out_data,
                            Param &otmp, Param &rtmp,
                            Param &in, uint threads_x,
                            uint groups_x, uint groups_y)
    {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> whereProgs;
        static std::map<int, Kernel *> whereKerns;

        int device= getActiveDeviceId();

        std::call_once(compileFlags[device], [device] () {

                ToNum<T> toNum;

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D zero=" << toNum(scalar<T>(0))
                        << " -D CPLX=" << af::iscplx<T>();
                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }
                Program prog;
                buildProgram(prog, where_cl, where_cl_len, options.str());
                whereProgs[device] = new Program(prog);
                whereKerns[device] = new Kernel(*whereProgs[device], "get_out_idx_kernel");
            });

        NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
        NDRange global(local[0] * groups_x * in.info.dims[2],
                       local[1] * groups_y * in.info.dims[3]);

        uint lim = divup(otmp.info.dims[0], (threads_x * groups_x));

        auto whereOp = make_kernel<Buffer,
                                   Buffer, KParam,
                                   Buffer, KParam,
                                   Buffer, KParam,
                                   uint, uint, uint>(*whereKerns[device]);

        whereOp(EnqueueArgs(getQueue(), global, local),
                *out_data,
                *otmp.data, otmp.info,
                *rtmp.data, rtmp.info,
                *in.data, in.info,
                groups_x, groups_y, lim);

        CL_DEBUG_FINISH(getQueue());

    }

    template<typename T>
    static void where(Param &out, Param &in)
    {
        try {
            uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
            threads_x = std::min(threads_x, THREADS_PER_GROUP);
            uint threads_y = THREADS_PER_GROUP / threads_x;

            uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
            uint groups_y = divup(in.info.dims[1], threads_y);

            Param rtmp;
            Param otmp;

            rtmp.info.dims[0] = groups_x;
            otmp.info.dims[0] = in.info.dims[0];

            rtmp.info.strides[0] = 1;
            otmp.info.strides[0] = 1;

            rtmp.info.offset = 0;
            otmp.info.offset = 0;

            for (int k = 1; k < 4; k++) {
                rtmp.info.dims[k] = in.info.dims[k];
                rtmp.info.strides[k] = rtmp.info.strides[k - 1] * rtmp.info.dims[k - 1];

                otmp.info.dims[k] = in.info.dims[k];
                otmp.info.strides[k] = otmp.info.strides[k - 1] * otmp.info.dims[k - 1];
            }

            int rtmp_elements = rtmp.info.strides[3] * rtmp.info.dims[3];
            rtmp.data = bufferAlloc(rtmp_elements * sizeof(uint));

            int otmp_elements = otmp.info.strides[3] * otmp.info.dims[3];
            otmp.data = bufferAlloc(otmp_elements * sizeof(uint));

            scan_first_launcher<T, uint, af_notzero_t>(otmp, rtmp, in,
                                                       false,
                                                       groups_x, groups_y,
                                                       threads_x);

            // Linearize the dimensions and perform scan
            Param ltmp = rtmp;
            ltmp.info.offset = 0;
            ltmp.info.dims[0] = rtmp_elements;
            for (int k = 1; k < 4; k++) {
                ltmp.info.dims[k] = 1;
                ltmp.info.strides[k] = rtmp_elements;
            }

            scan_first<uint, uint, af_add_t>(ltmp, ltmp);

            // Get output size and allocate output
            uint total;
            getQueue().enqueueReadBuffer(*rtmp.data, CL_TRUE,
                                         sizeof(uint) * (rtmp_elements - 1),
                                         sizeof(uint),
                                         &total);


            out.data = bufferAlloc(total * sizeof(uint));

            out.info.dims[0] = total;
            out.info.strides[0] = 1;
            for (int k = 1; k < 4; k++) {
                out.info.dims[k] = 1;
                out.info.strides[k] = total;
            }

            get_out_idx<T>(out.data, otmp, rtmp, in, threads_x, groups_x, groups_y);

            bufferFree(rtmp.data);
            bufferFree(otmp.data);
        } catch (cl::Error err) {
            CL_TO_AF_ERROR(err);
        }
    }
}
}

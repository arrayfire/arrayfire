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
#include <kernel_headers/scan_dim.hpp>
#include <kernel_headers/ops.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <type_util.hpp>
#include "names.hpp"
#include "config.hpp"

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
    template<typename Ti, typename To, af_op_t op, int dim, bool isFinalPass, uint threads_y>
    static Kernel* get_scan_dim_kernels(int kerIdx)
    {
        try {
            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static std::map<int, Program*> scanProgs;
            static std::map<int, Kernel*>  scanKerns;
            static std::map<int, Kernel*>  bcastKerns;

            int device= getActiveDeviceId();

            std::call_once(compileFlags[device], [device] () {

                    Binary<To, op> scan;
                    ToNum<To> toNum;

                    std::ostringstream options;
                    options << " -D To=" << dtype_traits<To>::getName()
                            << " -D Ti=" << dtype_traits<Ti>::getName()
                            << " -D T=To"
                            << " -D dim=" << dim
                            << " -D DIMY=" << threads_y
                            << " -D THREADS_X=" << THREADS_X
                            << " -D init=" << toNum(scan.init())
                            << " -D " << binOpName<op>()
                            << " -D CPLX=" << af::iscplx<Ti>()
                            << " -D isFinalPass=" << (int)(isFinalPass);
                    if (std::is_same<Ti, double>::value ||
                        std::is_same<Ti, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    const char *ker_strs[] = {ops_cl, scan_dim_cl};
                    const int   ker_lens[] = {ops_cl_len, scan_dim_cl_len};
                    cl::Program prog;
                    buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                    scanProgs[device] = new Program(prog);

                    scanKerns[device] = new Kernel(*scanProgs[device],  "scan_dim_kernel");
                    bcastKerns[device] = new Kernel(*scanProgs[device],  "bcast_dim_kernel");

                });

            return (kerIdx == 0) ? scanKerns[device] : bcastKerns[device];
        } catch (cl::Error err) {
            CL_TO_AF_ERROR(err);
            throw;
        }
    }

    template<typename Ti, typename To, af_op_t op, int dim, bool isFinalPass, uint threads_y>
    static void scan_dim_launcher(Param &out,
                                  Param &tmp,
                                  const Param &in,
                                  const uint groups_all[4])
    {
        try {
            Kernel* ker = get_scan_dim_kernels<Ti, To, op, dim, isFinalPass, threads_y>(0);

            NDRange local(THREADS_X, threads_y);
            NDRange global(groups_all[0] * groups_all[2] * local[0],
                           groups_all[1] * groups_all[3] * local[1]);

            uint lim = divup(out.info.dims[dim], (threads_y * groups_all[dim]));

            auto scanOp = make_kernel<Buffer, KParam,
                                      Buffer, KParam,
                                      Buffer, KParam,
                                      uint, uint,
                                      uint, uint>(*ker);


            scanOp(EnqueueArgs(getQueue(), global, local),
                   *out.data, out.info, *tmp.data, tmp.info, *in.data, in.info,
                   groups_all[0], groups_all[1], groups_all[dim], lim);

            CL_DEBUG_FINISH(getQueue());
        } catch (cl::Error err) {
            CL_TO_AF_ERROR(err);
            throw;
        }
    }

    template<typename Ti, typename To, af_op_t op, int dim, bool isFinalPass, uint threads_y>
    static void bcast_dim_launcher(Param &out,
                                   Param &tmp,
                                   const uint groups_all[4])
    {
        try {
            Kernel* ker = get_scan_dim_kernels<Ti, To, op, dim, isFinalPass, threads_y>(1);

            NDRange local(THREADS_X, threads_y);
            NDRange global(groups_all[0] * groups_all[2] * local[0],
                           groups_all[1] * groups_all[3] * local[1]);

            uint lim = divup(out.info.dims[dim], (threads_y * groups_all[dim]));

            auto bcastOp = make_kernel<Buffer, KParam,
                                       Buffer, KParam,
                                       uint, uint,
                                       uint, uint>(*ker);

            bcastOp(EnqueueArgs(getQueue(), global, local),
                    *out.data, out.info, *tmp.data, tmp.info,
                    groups_all[0], groups_all[1], groups_all[dim], lim);

            CL_DEBUG_FINISH(getQueue());
        } catch (cl::Error err) {
            CL_TO_AF_ERROR(err);
            throw;
        }
    }


    template<typename Ti, typename To, af_op_t op, int dim, bool isFinalPass>
    static void scan_dim_fn(Param &out,
                            Param &tmp,
                            const Param &in,
                            const uint threads_y,
                            const uint groups_all[4])
    {

        switch (threads_y) {
        case 8:
            (scan_dim_launcher<Ti, To, op, dim, isFinalPass, 8>)(
                out, tmp, in, groups_all); break;
        case 4:
            (scan_dim_launcher<Ti, To, op, dim, isFinalPass, 4>)(
                out, tmp, in, groups_all); break;
        case 2:
            (scan_dim_launcher<Ti, To, op, dim, isFinalPass, 2>)(
                out, tmp, in, groups_all); break;
        case 1:
            (scan_dim_launcher<Ti, To, op, dim, isFinalPass, 1>)(
                out, tmp, in, groups_all); break;
        }

    }

    template<typename Ti, typename To, af_op_t op, int dim, bool isFinalPass>
    static void bcast_dim_fn(Param &out,
                             Param &tmp,
                             const uint threads_y,
                             const uint groups_all[4])
    {

        switch (threads_y) {
        case 8:
            (bcast_dim_launcher<Ti, To, op, dim, isFinalPass, 8>)(
                out, tmp, groups_all); break;
        case 4:
            (bcast_dim_launcher<Ti, To, op, dim, isFinalPass, 4>)(
                out, tmp, groups_all); break;
        case 2:
            (bcast_dim_launcher<Ti, To, op, dim, isFinalPass, 2>)(
                out, tmp, groups_all); break;
        case 1:
            (bcast_dim_launcher<Ti, To, op, dim, isFinalPass, 1>)(
                out, tmp, groups_all); break;
        }
    }

    template<typename Ti, typename To, af_op_t op, int dim>
    static void scan_dim(Param &out, const Param &in)
    {
        try {
            uint threads_y = std::min(THREADS_Y, nextpow2(out.info.dims[dim]));
            uint threads_x = THREADS_X;

            uint groups_all[] = {divup((uint)out.info.dims[0], threads_x),
                                 (uint)out.info.dims[1],
                                 (uint)out.info.dims[2],
                                 (uint)out.info.dims[3]};

            groups_all[dim] = divup(out.info.dims[dim], threads_y * REPEAT);

            if (groups_all[dim] == 1) {

                scan_dim_fn<Ti, To, op, dim, true>(out, out, in,
                                                   threads_y,
                                                   groups_all);
            } else {

                Param tmp = out;

                tmp.info.dims[dim] = groups_all[dim];
                tmp.info.strides[0] = 1;
                for (int k = 1; k < 4; k++) {
                    tmp.info.strides[k] = tmp.info.strides[k - 1] * tmp.info.dims[k - 1];
                }

                int tmp_elements = tmp.info.strides[3] * tmp.info.dims[3];
                // FIXME: Do I need to free this ?
                tmp.data = bufferAlloc(tmp_elements * sizeof(To));

                scan_dim_fn<Ti, To, op, dim, false>(out, tmp, in,
                                                    threads_y,
                                                    groups_all);

                int gdim = groups_all[dim];
                groups_all[dim] = 1;

                if (op == af_notzero_t) {
                    scan_dim_fn<To, To, af_add_t, dim, true>(tmp, tmp, tmp,
                                                             threads_y,
                                                             groups_all);
                } else {
                    scan_dim_fn<To, To,       op, dim, true>(tmp, tmp, tmp,
                                                             threads_y,
                                                             groups_all);
                }

                groups_all[dim] = gdim;
                bcast_dim_fn<To, To, op, dim, true>(out, tmp,
                                                    threads_y,
                                                    groups_all);
                bufferFree(tmp.data);
            }
        } catch (cl::Error err) {
            CL_TO_AF_ERROR(err);
            throw;
        }
    }
}
}

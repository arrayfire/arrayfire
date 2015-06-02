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
#include <kernel_headers/reduce_first.hpp>
#include <kernel_headers/reduce_dim.hpp>
#include <kernel_headers/ops.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <type_util.hpp>
#include "names.hpp"
#include "config.hpp"
#include <memory.hpp>
#include <memory>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;
using std::unique_ptr;

namespace opencl
{

namespace kernel
{

    template<typename Ti, typename To, af_op_t op, int dim, int threads_y>
    void reduce_dim_launcher(Param out, Param in,
                       const uint groups_all[4])
    {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> reduceProgs;
        static std::map<int, Kernel*> reduceKerns;

        int device= getActiveDeviceId();
        std::call_once(compileFlags[device], [device] () {

                Binary<To, op> reduce;
                ToNum<To> toNum;

                std::ostringstream options;
                options << " -D To=" << dtype_traits<To>::getName()
                        << " -D Ti=" << dtype_traits<Ti>::getName()
                        << " -D T=To"
                        << " -D dim=" << dim
                        << " -D DIMY=" << threads_y
                        << " -D THREADS_X=" << THREADS_X
                        << " -D init=" << toNum(reduce.init())
                        << " -D " << binOpName<op>()
                        << " -D CPLX=" << af::iscplx<Ti>();
                if (std::is_same<Ti, double>::value ||
                    std::is_same<Ti, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                const char *ker_strs[] = {ops_cl, reduce_dim_cl};
                const int   ker_lens[] = {ops_cl_len, reduce_dim_cl_len};
                Program prog;
                buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                reduceProgs[device] = new Program(prog);

                reduceKerns[device] = new Kernel(*reduceProgs[device], "reduce_dim_kernel");
            });

        NDRange local(THREADS_X, threads_y);
        NDRange global(groups_all[0] * groups_all[2] * local[0],
                       groups_all[1] * groups_all[3] * local[1]);

        auto reduceOp = make_kernel<Buffer, KParam,
                                    Buffer, KParam,
                                    uint, uint, uint>(*reduceKerns[device]);

        reduceOp(EnqueueArgs(getQueue(), global, local),
                 *out.data, out.info,
                 *in.data, in.info,
                 groups_all[0],
                 groups_all[1],
                 groups_all[dim]);

        CL_DEBUG_FINISH(getQueue());
    }

    template<typename Ti, typename To, af_op_t op, int dim>
    void reduce_dim_fn(Param out, Param in,
                       const uint threads_y, const uint groups_all[4])
    {
        switch(threads_y) {
        case 8: return reduce_dim_launcher<Ti, To, op, dim, 8>(out, in, groups_all);
        case 4: return reduce_dim_launcher<Ti, To, op, dim, 4>(out, in, groups_all);
        case 2: return reduce_dim_launcher<Ti, To, op, dim, 2>(out, in, groups_all);
        case 1: return reduce_dim_launcher<Ti, To, op, dim, 1>(out, in, groups_all);
        case 16: return reduce_dim_launcher<Ti, To, op, dim, 16>(out, in, groups_all);
        case 32: return reduce_dim_launcher<Ti, To, op, dim, 32>(out, in, groups_all);
        }
    }

    template<typename Ti, typename To, af_op_t op, int dim>
    void reduce_dim(Param out, Param in)
    {
        uint threads_y = std::min(THREADS_Y, nextpow2(in.info.dims[dim]));
        uint threads_x = THREADS_X;

        uint groups_all[] = {(uint)divup(in.info.dims[0], threads_x),
                             (uint)in.info.dims[1],
                             (uint)in.info.dims[2],
                             (uint)in.info.dims[3]};

        groups_all[dim] = divup(in.info.dims[dim], threads_y * REPEAT);

        Param tmp = out;

        int tmp_elements = 1;
        if (groups_all[dim] > 1) {
            tmp.info.dims[dim] = groups_all[dim];

            for (int k = 0; k < 4; k++) tmp_elements *= tmp.info.dims[k];

            tmp.data = bufferAlloc(tmp_elements * sizeof(To));

            for (int k = dim + 1; k < 4; k++) tmp.info.strides[k] *= groups_all[dim];
        }

        reduce_dim_fn<Ti, To, op, dim>(tmp, in, threads_y, groups_all);

        if (groups_all[dim] > 1) {
            groups_all[dim] = 1;

            if (op == af_notzero_t) {
                reduce_dim_fn<To, To, af_add_t, dim>(out, tmp, threads_y, groups_all);
            } else {
                reduce_dim_fn<To, To,       op, dim>(out, tmp, threads_y, groups_all);
            }
            bufferFree(tmp.data);
        }

    }

    template<typename Ti, typename To, af_op_t op, int threads_x>
    void reduce_first_launcher(Param out, Param in,
                               const uint groups_x,
                               const uint groups_y)
    {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> reduceProgs;
        static std::map<int, Kernel*>  reduceKerns;

        int device= getActiveDeviceId();
        std::call_once(compileFlags[device], [device] () {

                Binary<To, op> reduce;
                ToNum<To> toNum;

                std::ostringstream options;
                options << " -D To=" << dtype_traits<To>::getName()
                        << " -D Ti=" << dtype_traits<Ti>::getName()
                        << " -D T=To"
                        << " -D DIMX=" << threads_x
                        << " -D THREADS_PER_GROUP=" << THREADS_PER_GROUP
                        << " -D init=" << toNum(reduce.init())
                        << " -D " << binOpName<op>()
                        << " -D CPLX=" << af::iscplx<Ti>();
                if (std::is_same<Ti, double>::value ||
                    std::is_same<Ti, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                const char *ker_strs[] = {ops_cl, reduce_first_cl};
                const int   ker_lens[] = {ops_cl_len, reduce_first_cl_len};
                Program prog;
                buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                reduceProgs[device] = new Program(prog);

                reduceKerns[device] = new Kernel(*reduceProgs[device], "reduce_first_kernel");
            });

        NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
        NDRange global(groups_x * in.info.dims[2] * local[0],
                       groups_y * in.info.dims[3] * local[1]);

        uint repeat = divup(in.info.dims[0], (local[0] * groups_x));

        auto reduceOp = make_kernel<Buffer, KParam,
                                    Buffer, KParam,
                                    uint, uint, uint>(*reduceKerns[device]);

        reduceOp(EnqueueArgs(getQueue(), global, local),
                 *out.data, out.info,
                 *in.data, in.info, groups_x, groups_y, repeat);

        CL_DEBUG_FINISH(getQueue());
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce_first_fn(Param out, Param in,
                         const uint groups_x,
                         const uint groups_y,
                         const uint threads_x)
    {
        switch(threads_x) {
        case  32: return reduce_first_launcher<Ti, To, op,  32>(out, in, groups_x,
                                                                groups_y);
        case  64: return reduce_first_launcher<Ti, To, op,  64>(out, in, groups_x,
                                                                groups_y);
        case 128: return reduce_first_launcher<Ti, To, op, 128>(out, in, groups_x,
                                                                groups_y);
        case 256: return reduce_first_launcher<Ti, To, op, 256>(out, in, groups_x,
                                                                groups_y);
        case 512: return reduce_first_launcher<Ti, To, op, 512>(out, in, groups_x,
                                                                groups_y);
        }
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce_first(Param out, Param in)
    {
        uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
        threads_x = std::min(threads_x, THREADS_PER_GROUP);
        uint threads_y = THREADS_PER_GROUP / threads_x;

        uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
        uint groups_y = divup(in.info.dims[1], threads_y);

        Param tmp = out;

        if (groups_x > 1) {
            tmp.data = bufferAlloc(groups_x *
                                in.info.dims[1] *
                                in.info.dims[2] *
                                in.info.dims[3] *
                                sizeof(To));

            tmp.info.dims[0] = groups_x;
            for (int k = 1; k < 4; k++) tmp.info.strides[k] *= groups_x;
        }

        reduce_first_fn<Ti, To, op>(tmp, in, groups_x, groups_y, threads_x);

        if (groups_x > 1) {

            //FIXME: Is there an alternative to the if condition ?
            if (op == af_notzero_t) {
                reduce_first_fn<To, To, af_add_t>(out, tmp, 1, groups_y, threads_x);
            } else {
                reduce_first_fn<To, To,       op>(out, tmp, 1, groups_y, threads_x);
            }

            bufferFree(tmp.data);
        }
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce(Param out, Param in, int dim)
    {
        try {
            switch (dim) {
            case 0: return reduce_first<Ti, To, op   >(out, in);
            case 1: return reduce_dim  <Ti, To, op, 1>(out, in);
            case 2: return reduce_dim  <Ti, To, op, 2>(out, in);
            case 3: return reduce_dim  <Ti, To, op, 3>(out, in);
            }
        } catch(cl::Error ex) {
            CL_TO_AF_ERROR(ex);
        }
    }

    template<typename Ti, typename To, af_op_t op>
    To reduce_all(Param in)
    {
        try {
            int in_elements = in.info.dims[3] * in.info.strides[3];

            // FIXME: Use better heuristics to get to the optimum number
            if (in_elements > 4096) {

                bool is_linear = (in.info.strides[0] == 1);
                for (int k = 1; k < 4; k++) {
                    is_linear &= (in.info.strides[k] == (in.info.strides[k - 1] * in.info.dims[k - 1]));
                }

                if (is_linear) {
                    in.info.dims[0] = in_elements;
                    for (int k = 1; k < 4; k++) {
                        in.info.dims[k] = 1;
                        in.info.strides[k] = in_elements;
                    }
                }

                uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
                threads_x = std::min(threads_x, THREADS_PER_GROUP);
                uint threads_y = THREADS_PER_GROUP / threads_x;

                Param tmp;
                uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
                uint groups_y = divup(in.info.dims[1], threads_y);

                tmp.info.offset = 0;
                tmp.info.dims[0] = groups_x;
                tmp.info.strides[0] = 1;

                for (int k = 1; k < 4; k++) {
                    tmp.info.dims[k] = in.info.dims[k];
                    tmp.info.strides[k] = tmp.info.dims[k - 1] * tmp.info.strides[k - 1];
                }

                int tmp_elements = tmp.info.strides[3] * tmp.info.dims[3];
                tmp.data = bufferAlloc(tmp_elements * sizeof(To));

                reduce_first_fn<Ti, To, op>(tmp, in, groups_x, groups_y, threads_x);

                unique_ptr<To> h_ptr(new To[tmp_elements]);
                getQueue().enqueueReadBuffer(*tmp.data, CL_TRUE, 0, sizeof(To) * tmp_elements, h_ptr.get());

                Binary<To, op> reduce;
                To out = reduce.init();
                for (int i = 0; i < (int)tmp_elements; i++) {
                    out = reduce(out, h_ptr.get()[i]);
                }

                bufferFree(tmp.data);
                return out;

            } else {

                unique_ptr<Ti> h_ptr(new Ti[in_elements]);
                getQueue().enqueueReadBuffer(*in.data, CL_TRUE, 0, sizeof(Ti) * in_elements, h_ptr.get());

                Transform<Ti, To, op> transform;
                Binary<To, op> reduce;
                To out = reduce.init();

                for (int i = 0; i < (int)in_elements; i++) {
                    out = reduce(out, transform(h_ptr.get()[i]));
                }

                return out;
            }
        } catch(cl::Error ex) {
            CL_TO_AF_ERROR(ex);
        }
    }


}

}

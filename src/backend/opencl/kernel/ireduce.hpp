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
#include <memory>
#include <kernel_headers/ireduce_first.hpp>
#include <kernel_headers/ireduce_dim.hpp>
#include <kernel_headers/iops.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <cache.hpp>
#include <debug_opencl.hpp>
#include <type_util.hpp>
#include "names.hpp"
#include "config.hpp"
#include <memory.hpp>

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

    template<typename T, af_op_t op>
    void ireduce_dim_launcher(Param out, cl::Buffer *oidx,
                              Param in, cl::Buffer *iidx,
                              const int dim,
                              const int threads_y,
                              const bool is_first,
                              const uint groups_all[4])
    {
        std::string ref_name =
            std::string("ireduce_") +
            std::to_string(dim) +
            std::string("_") +
            std::string(dtype_traits<T>::getName()) +
            std::string("_") +
            std::to_string(op) +
            std::string("_") +
            std::to_string(is_first) +
            std::string("_") +
            std::to_string(threads_y);

        int device = getActiveDeviceId();
        kc_t::iterator idx = kernelCaches[device].find(ref_name);

        kc_entry_t entry;
        if (idx == kernelCaches[device].end()) {

                Binary<T, op> ireduce;
                ToNum<T> toNum;

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D dim=" << dim
                        << " -D DIMY=" << threads_y
                        << " -D THREADS_X=" << THREADS_X
                        << " -D init=" << toNum(ireduce.init())
                        << " -D " << binOpName<op>()
                        << " -D CPLX=" << af::iscplx<T>()
                        << " -D IS_FIRST=" << is_first;

                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                const char *ker_strs[] = {iops_cl, ireduce_dim_cl};
                const int   ker_lens[] = {iops_cl_len, ireduce_dim_cl_len};
                Program prog;
                buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                entry.prog = new Program(prog);
                entry.ker = new Kernel(*entry.prog, "ireduce_dim_kernel");

                kernelCaches[device][ref_name] = entry;
        } else {
            entry = idx->second;
        }

        NDRange local(THREADS_X, threads_y);
        NDRange global(groups_all[0] * groups_all[2] * local[0],
                       groups_all[1] * groups_all[3] * local[1]);

        auto ireduceOp = make_kernel<Buffer, KParam, Buffer,
                                     Buffer, KParam, Buffer,
                                     uint, uint, uint>(*entry.ker);

        ireduceOp(EnqueueArgs(getQueue(), global, local),
                  *out.data, out.info, *oidx,
                  *in.data, in.info, *iidx,
                  groups_all[0],
                  groups_all[1],
                  groups_all[dim]);

        CL_DEBUG_FINISH(getQueue());
    }

    template<typename T, af_op_t op>
    void ireduce_dim(Param out, cl::Buffer *oidx, Param in, int dim)
    {
        uint threads_y = std::min(THREADS_Y, nextpow2(in.info.dims[dim]));
        uint threads_x = THREADS_X;

        uint groups_all[] = {(uint)divup(in.info.dims[0], threads_x),
                             (uint)in.info.dims[1],
                             (uint)in.info.dims[2],
                             (uint)in.info.dims[3]};

        groups_all[dim] = divup(in.info.dims[dim], threads_y * REPEAT);

        Param tmp = out;
        cl::Buffer *tidx = oidx;

        int tmp_elements = 1;
        if (groups_all[dim] > 1) {
            tmp.info.dims[dim] = groups_all[dim];

            for (int k = 0; k < 4; k++) tmp_elements *= tmp.info.dims[k];

            tmp.data = bufferAlloc(tmp_elements * sizeof(T));
            tidx = bufferAlloc(tmp_elements * sizeof(uint));

            for (int k = dim + 1; k < 4; k++) tmp.info.strides[k] *= groups_all[dim];
        }

        ireduce_dim_launcher<T, op>(tmp, tidx, in, tidx, dim, threads_y, true, groups_all);

        if (groups_all[dim] > 1) {
            groups_all[dim] = 1;

            ireduce_dim_launcher<T, op>(out, oidx, tmp, tidx, dim, threads_y, false, groups_all);
            bufferFree(tmp.data);
            bufferFree(tidx);
        }

    }

    template<typename T, af_op_t op>
    void ireduce_first_launcher(Param out, cl::Buffer *oidx,
                                Param in, cl::Buffer *iidx,
                                const int threads_x,
                                const bool is_first,
                                const uint groups_x,
                                const uint groups_y)
    {
        std::string ref_name =
            std::string("ireduce_0_") +
            std::string(dtype_traits<T>::getName()) +
            std::string("_") +
            std::to_string(op) +
            std::string("_") +
            std::to_string(is_first);
            std::string("_") +
            std::to_string(threads_x);

        int device = getActiveDeviceId();
        kc_t::iterator idx = kernelCaches[device].find(ref_name);

        kc_entry_t entry;
        if (idx == kernelCaches[device].end()) {

            Binary<T, op> ireduce;
            ToNum<T> toNum;

            std::ostringstream options;
            options << " -D T=" << dtype_traits<T>::getName()
                    << " -D DIMX=" << threads_x
                    << " -D THREADS_PER_GROUP=" << THREADS_PER_GROUP
                    << " -D init=" << toNum(ireduce.init())
                    << " -D " << binOpName<op>()
                    << " -D CPLX=" << af::iscplx<T>()
                    << " -D IS_FIRST=" << is_first;

            if (std::is_same<T, double>::value ||
                std::is_same<T, cdouble>::value) {
                options << " -D USE_DOUBLE";
            }

            const char *ker_strs[] = {iops_cl, ireduce_first_cl};
            const int   ker_lens[] = {iops_cl_len, ireduce_first_cl_len};
            Program prog;
            buildProgram(prog, 2, ker_strs, ker_lens, options.str());
            entry.prog = new Program(prog);
            entry.ker = new Kernel(*entry.prog, "ireduce_first_kernel");

            kernelCaches[device][ref_name] = entry;
        } else {
            entry = idx->second;
        }

        NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
        NDRange global(groups_x * in.info.dims[2] * local[0],
                       groups_y * in.info.dims[3] * local[1]);

        uint repeat = divup(in.info.dims[0], (local[0] * groups_x));

        auto ireduceOp = make_kernel<Buffer, KParam, Buffer,
                                     Buffer, KParam, Buffer,
                                     uint, uint, uint>(*entry.ker);

        ireduceOp(EnqueueArgs(getQueue(), global, local),
                  *out.data, out.info, *oidx,
                  *in.data, in.info, *iidx,
                  groups_x, groups_y, repeat);

        CL_DEBUG_FINISH(getQueue());
    }

    template<typename T, af_op_t op>
    void ireduce_first(Param out, cl::Buffer *oidx, Param in)
    {
        uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
        threads_x = std::min(threads_x, THREADS_PER_GROUP);
        uint threads_y = THREADS_PER_GROUP / threads_x;

        uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
        uint groups_y = divup(in.info.dims[1], threads_y);

        Param tmp = out;
        cl::Buffer *tidx = oidx;

        if (groups_x > 1) {

            tmp.data = bufferAlloc(groups_x *
                                in.info.dims[1] *
                                in.info.dims[2] *
                                in.info.dims[3] *
                                sizeof(T));

            tidx = bufferAlloc(groups_x *
                               in.info.dims[1] *
                               in.info.dims[2] *
                               in.info.dims[3] *
                               sizeof(uint));


            tmp.info.dims[0] = groups_x;
            for (int k = 1; k < 4; k++) tmp.info.strides[k] *= groups_x;
        }

        ireduce_first_launcher<T, op>(tmp, tidx, in, tidx, threads_x, true, groups_x, groups_y);

        if (groups_x > 1) {
            ireduce_first_launcher<T, op>(out, oidx, tmp, tidx, threads_x, false, 1, groups_y);

            bufferFree(tmp.data);
            bufferFree(tidx);
        }
    }

    template<typename T, af_op_t op>
    void ireduce(Param out, cl::Buffer *oidx, Param in, int dim)
    {
        try {
            if (dim == 0)
                return ireduce_first<T, op>(out, oidx, in);
            else
                return ireduce_dim  <T, op>(out, oidx, in, dim);
        } catch(cl::Error ex) {
            CL_TO_AF_ERROR(ex);
        }
    }

    template<typename T> double cabs(const T in) { return (double)in; }
    static double cabs(const cfloat in) { return (double)abs(in); }
    static double cabs(const cdouble in) { return (double)abs(in); }

    template<af_op_t op, typename T>
    struct MinMaxOp
    {
        T m_val;
        uint m_idx;
        MinMaxOp(T val, uint idx) :
            m_val(val), m_idx(idx)
        {
        }

        void operator()(T val, uint idx)
        {
            if (cabs(val) < cabs(m_val) ||
                (cabs(val) == cabs(m_val) &&
                 idx > m_idx)) {
                m_val = val;
                m_idx = idx;
            }
        }
    };

    template<typename T>
    struct MinMaxOp<af_max_t, T>
    {
        T m_val;
        uint m_idx;
        MinMaxOp(T val, uint idx) :
            m_val(val), m_idx(idx)
        {
        }

        void operator()(T val, uint idx)
        {
            if (cabs(val) > cabs(m_val) ||
                (cabs(val) == cabs(m_val) &&
                 idx <= m_idx)) {
                m_val = val;
                m_idx = idx;
            }
        }
    };


    template<typename T, af_op_t op>
    T ireduce_all(uint *loc, Param in)
    {
        try {
            int in_elements = in.info.dims[0] * in.info.dims[1] * in.info.dims[2] * in.info.dims[3];

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
                tmp.data = bufferAlloc(tmp_elements * sizeof(T));
                cl::Buffer *tidx = bufferAlloc(tmp_elements * sizeof(uint));

                ireduce_first_launcher<T, op>(tmp, tidx, in, tidx, threads_x, true, groups_x, groups_y);

                unique_ptr<T> h_ptr(new T[tmp_elements]);
                unique_ptr<uint> h_iptr(new uint[tmp_elements]);

                getQueue().enqueueReadBuffer(*tmp.data, CL_TRUE, 0, sizeof(T) * tmp_elements, h_ptr.get());
                getQueue().enqueueReadBuffer(*tidx, CL_TRUE, 0, sizeof(uint) * tmp_elements, h_iptr.get());

                T* h_ptr_raw = h_ptr.get();
                uint* h_iptr_raw = h_iptr.get();
                MinMaxOp<op, T> Op(h_ptr_raw[0], h_iptr_raw[0]);

                for (int i = 1; i < (int)tmp_elements; i++) {
                    Op(h_ptr_raw[i], h_iptr_raw[i]);
                }

                bufferFree(tmp.data);
                bufferFree(tidx);

                *loc = Op.m_idx;
                return Op.m_val;

            } else {

                unique_ptr<T> h_ptr(new T[in_elements]);
                T* h_ptr_raw = h_ptr.get();

                getQueue().enqueueReadBuffer(*in.data, CL_TRUE, sizeof(T) * in.info.offset,
                                             sizeof(T) * in_elements, h_ptr_raw);


                MinMaxOp<op, T> Op(h_ptr_raw[0], 0);
                for (int i = 1; i < (int)in_elements; i++) {
                    Op(h_ptr_raw[i], i);
                }

                *loc = Op.m_idx;
                return Op.m_val;
            }
        } catch(cl::Error ex) {
            CL_TO_AF_ERROR(ex);
        }
    }


}

}

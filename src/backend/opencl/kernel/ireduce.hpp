/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/iops.hpp>
#include <kernel_headers/ireduce_dim.hpp>
#include <kernel_headers/ireduce_first.hpp>
#include <memory.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <type_util.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include "config.hpp"
#include "names.hpp"

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;
using std::unique_ptr;

namespace opencl {

namespace kernel {

template<typename T, af_op_t op>
void ireduce_dim_launcher(Param out, cl::Buffer *oidx, Param in,
                          cl::Buffer *iidx, const int dim, const int threads_y,
                          const bool is_first, const uint groups_all[4], Param rlen) {
    std::string ref_name =
        std::string("ireduce_") + std::to_string(dim) + std::string("_") +
        std::string(dtype_traits<T>::getName()) + std::string("_") +
        std::to_string(op) + std::string("_") + std::to_string(is_first) +
        std::string("_") + std::to_string(threads_y);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        ToNumStr<T> toNumStr;

        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName() << " -D kDim=" << dim
                << " -D DIMY=" << threads_y << " -D THREADS_X=" << THREADS_X
                << " -D init=" << toNumStr(Binary<T, op>::init()) << " -D "
                << binOpName<op>() << " -D CPLX=" << af::iscplx<T>()
                << " -D IS_FIRST=" << is_first;

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        const char *ker_strs[] = {iops_cl, ireduce_dim_cl};
        const int ker_lens[]   = {iops_cl_len, ireduce_dim_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "ireduce_dim_kernel");

        addKernelToCache(device, ref_name, entry);
    }

    NDRange local(THREADS_X, threads_y);
    NDRange global(groups_all[0] * groups_all[2] * local[0],
                   groups_all[1] * groups_all[3] * local[1]);

    auto ireduceOp = KernelFunctor<Buffer, KParam, Buffer, Buffer, KParam,
                                   Buffer, uint, uint, uint, Buffer, KParam>(*entry.ker);

    ireduceOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *oidx, *in.data, in.info, *iidx, groups_all[0], groups_all[1],
              groups_all[dim], *rlen.data, rlen.info);

    CL_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op>
void ireduce_dim(Param out, cl::Buffer *oidx, Param in, int dim, Param rlen) {
    uint threads_y = std::min(THREADS_Y, nextpow2(in.info.dims[dim]));
    uint threads_x = THREADS_X;

    uint groups_all[] = {(uint)divup(in.info.dims[0], threads_x),
                         (uint)in.info.dims[1], (uint)in.info.dims[2],
                         (uint)in.info.dims[3]};

    groups_all[dim] = divup(in.info.dims[dim], threads_y * REPEAT);

    Param tmp        = out;
    cl::Buffer *tidx = oidx;

    int tmp_elements = 1;
    if (groups_all[dim] > 1) {
        tmp.info.dims[dim] = groups_all[dim];

        for (int k = 0; k < 4; k++) tmp_elements *= tmp.info.dims[k];

        tmp.data = bufferAlloc(tmp_elements * sizeof(T));
        tidx     = bufferAlloc(tmp_elements * sizeof(uint));

        for (int k = dim + 1; k < 4; k++)
            tmp.info.strides[k] *= groups_all[dim];
    }

    ireduce_dim_launcher<T, op>(tmp, tidx, in, tidx, dim, threads_y, true,
                                groups_all, rlen);

    if (groups_all[dim] > 1) {
        groups_all[dim] = 1;

        ireduce_dim_launcher<T, op>(out, oidx, tmp, tidx, dim, threads_y, false,
                                    groups_all, rlen);
        bufferFree(tmp.data);
        bufferFree(tidx);
    }
}

template<typename T, af_op_t op>
void ireduce_first_launcher(Param out, cl::Buffer *oidx, Param in,
                            cl::Buffer *iidx, const int threads_x,
                            const bool is_first, const uint groups_x,
                            const uint groups_y, Param rlen) {
    std::string ref_name =
        std::string("ireduce_0_") + std::string(dtype_traits<T>::getName()) +
        std::string("_") + std::to_string(op) + std::string("_") +
        std::to_string(is_first) + std::string("_") + std::to_string(threads_x);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        ToNumStr<T> toNumStr;

        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D DIMX=" << threads_x
                << " -D THREADS_PER_GROUP=" << THREADS_PER_GROUP
                << " -D init=" << toNumStr(Binary<T, op>::init()) << " -D "
                << binOpName<op>() << " -D CPLX=" << af::iscplx<T>()
                << " -D IS_FIRST=" << is_first;

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        const char *ker_strs[] = {iops_cl, ireduce_first_cl};
        const int ker_lens[]   = {iops_cl_len, ireduce_first_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "ireduce_first_kernel");

        addKernelToCache(device, ref_name, entry);
    }

    NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    NDRange global(groups_x * in.info.dims[2] * local[0],
                   groups_y * in.info.dims[3] * local[1]);

    uint repeat = divup(in.info.dims[0], (local[0] * groups_x));

    auto ireduceOp = KernelFunctor<Buffer, KParam, Buffer, Buffer, KParam,
                                   Buffer, uint, uint, uint, Buffer, KParam>(*entry.ker);

    ireduceOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *oidx, *in.data, in.info, *iidx, groups_x, groups_y, repeat, *rlen.data, rlen.info);

    CL_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op>
void ireduce_first(Param out, cl::Buffer *oidx, Param in, Param rlen) {
    uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_GROUP);
    uint threads_y = THREADS_PER_GROUP / threads_x;

    uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
    uint groups_y = divup(in.info.dims[1], threads_y);

    Param tmp        = out;
    cl::Buffer *tidx = oidx;

    if (groups_x > 1) {
        tmp.data = bufferAlloc(groups_x * in.info.dims[1] * in.info.dims[2] *
                               in.info.dims[3] * sizeof(T));

        tidx = bufferAlloc(groups_x * in.info.dims[1] * in.info.dims[2] *
                           in.info.dims[3] * sizeof(uint));

        tmp.info.dims[0] = groups_x;
        for (int k = 1; k < 4; k++) tmp.info.strides[k] *= groups_x;
    }

    ireduce_first_launcher<T, op>(tmp, tidx, in, tidx, threads_x, true,
                                  groups_x, groups_y, rlen);

    if (groups_x > 1) {
        ireduce_first_launcher<T, op>(out, oidx, tmp, tidx, threads_x, false, 1,
                                      groups_y, rlen);

        bufferFree(tmp.data);
        bufferFree(tidx);
    }
}

template<typename T, af_op_t op>
void ireduce(Param out, cl::Buffer *oidx, Param in, int dim, Param rlen) {
    if (dim == 0)
        return ireduce_first<T, op>(out, oidx, in, rlen);
    else
        return ireduce_dim<T, op>(out, oidx, in, dim, rlen);
}

#if defined(__GNUC__) || defined(__GNUG__)
/* GCC/G++, Clang/LLVM, Intel ICC */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#else
/* Other */
#endif

template<typename T>
double cabs(const T in) {
    return (double)in;
}
static double cabs(const cfloat in) { return (double)abs(in); }
static double cabs(const cdouble in) { return (double)abs(in); }

template<af_op_t op, typename T>
struct MinMaxOp {
    T m_val;
    uint m_idx;
    MinMaxOp(T val, uint idx) : m_val(val), m_idx(idx) {}

    void operator()(T val, uint idx) {
        if (cabs(val) < cabs(m_val) ||
            (cabs(val) == cabs(m_val) && idx > m_idx)) {
            m_val = val;
            m_idx = idx;
        }
    }
};

template<typename T>
struct MinMaxOp<af_max_t, T> {
    T m_val;
    uint m_idx;
    MinMaxOp(T val, uint idx) : m_val(val), m_idx(idx) {}

    void operator()(T val, uint idx) {
        if (cabs(val) > cabs(m_val) ||
            (cabs(val) == cabs(m_val) && idx <= m_idx)) {
            m_val = val;
            m_idx = idx;
        }
    }
};

#if defined(__GNUC__) || defined(__GNUG__)
/* GCC/G++, Clang/LLVM, Intel ICC */
#pragma GCC diagnostic pop
#else
/* Other */
#endif

template<typename T, af_op_t op>
T ireduce_all(uint *loc, Param in) {
    int in_elements =
        in.info.dims[0] * in.info.dims[1] * in.info.dims[2] * in.info.dims[3];

    // FIXME: Use better heuristics to get to the optimum number
    if (in_elements > 4096) {
        bool is_linear = (in.info.strides[0] == 1);
        for (int k = 1; k < 4; k++) {
            is_linear &= (in.info.strides[k] ==
                          (in.info.strides[k - 1] * in.info.dims[k - 1]));
        }

        if (is_linear) {
            in.info.dims[0] = in_elements;
            for (int k = 1; k < 4; k++) {
                in.info.dims[k]    = 1;
                in.info.strides[k] = in_elements;
            }
        }

        uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
        threads_x      = std::min(threads_x, THREADS_PER_GROUP);
        uint threads_y = THREADS_PER_GROUP / threads_x;

        uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
        uint groups_y = divup(in.info.dims[1], threads_y);
        Array<T> tmp  = createEmptyArray<T>(
            {groups_x, in.info.dims[1], in.info.dims[2], in.info.dims[3]});

        int tmp_elements = tmp.elements();
        cl::Buffer *tidx = bufferAlloc(tmp_elements * sizeof(uint));

        Param rlen;
        rlen.data = nullptr;
        ireduce_first_launcher<T, op>(tmp, tidx, in, tidx, threads_x, true,
                                      groups_x, groups_y, rlen);

        unique_ptr<T[]> h_ptr(new T[tmp_elements]);
        unique_ptr<uint[]> h_iptr(new uint[tmp_elements]);

        getQueue().enqueueReadBuffer(*tmp.get(), CL_TRUE, 0,
                                     sizeof(T) * tmp_elements, h_ptr.get());
        getQueue().enqueueReadBuffer(*tidx, CL_TRUE, 0,
                                     sizeof(uint) * tmp_elements, h_iptr.get());

        T *h_ptr_raw     = h_ptr.get();
        uint *h_iptr_raw = h_iptr.get();

        if (!is_linear) {
            // Converting n-d index into a linear index
            // in is of size   [   dims0, dims1, dims2, dims3]
            // tidx is of size [groups_x, dims1, dims2, dims3]
            // i / groups_x gives you the batch number "N"
            // "N * dims0 + i" gives the linear index
            for (int i = 0; i < tmp_elements; i++) {
                h_iptr_raw[i] += (i / groups_x) * in.info.dims[0];
            }
        }

        MinMaxOp<op, T> Op(h_ptr_raw[0], h_iptr_raw[0]);
        for (int i = 1; i < (int)tmp_elements; i++) {
            Op(h_ptr_raw[i], h_iptr_raw[i]);
        }

        bufferFree(tidx);

        *loc = Op.m_idx;
        return Op.m_val;

    } else {
        unique_ptr<T[]> h_ptr(new T[in_elements]);
        T *h_ptr_raw = h_ptr.get();

        getQueue().enqueueReadBuffer(*in.data, CL_TRUE,
                                     sizeof(T) * in.info.offset,
                                     sizeof(T) * in_elements, h_ptr_raw);

        MinMaxOp<op, T> Op(h_ptr_raw[0], 0);
        for (int i = 1; i < (int)in_elements; i++) { Op(h_ptr_raw[i], i); }

        *loc = Op.m_idx;
        return Op.m_val;
    }
}

}  // namespace kernel

}  // namespace opencl

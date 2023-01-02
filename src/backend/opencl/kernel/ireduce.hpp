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
#include <common/Binary.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel/config.hpp>
#include <kernel/names.hpp>
#include <kernel_headers/iops.hpp>
#include <kernel_headers/ireduce_dim.hpp>
#include <kernel_headers/ireduce_first.hpp>
#include <memory.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T, af_op_t op>
void ireduceDimLauncher(Param out, cl::Buffer *oidx, Param in, cl::Buffer *iidx,
                        const int dim, const int threads_y, const bool is_first,
                        const uint groups_all[4], Param rlen) {
    ToNumStr<T> toNumStr;
    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(), TemplateArg(dim),       TemplateArg(op),
        TemplateArg(is_first), TemplateArg(threads_y),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(kDim, dim),
        DefineKeyValue(DIMY, threads_y),
        DefineValue(THREADS_X),
        DefineKeyValue(init, toNumStr(common::Binary<T, op>::init())),
        DefineKeyFromStr(binOpName<op>()),
        DefineKeyValue(CPLX, iscplx<T>()),
        DefineKeyValue(IS_FIRST, is_first),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto ireduceDim =
        common::getKernel("ireduce_dim_kernel",
                          {{iops_cl_src, ireduce_dim_cl_src}}, targs, options);

    cl::NDRange local(THREADS_X, threads_y);
    cl::NDRange global(groups_all[0] * groups_all[2] * local[0],
                       groups_all[1] * groups_all[3] * local[1]);

    ireduceDim(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *oidx, *in.data, in.info, *iidx, groups_all[0], groups_all[1],
               groups_all[dim], *rlen.data, rlen.info);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op>
void ireduceDim(Param out, cl::Buffer *oidx, Param in, int dim, Param rlen) {
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

    ireduceDimLauncher<T, op>(tmp, tidx, in, tidx, dim, threads_y, true,
                              groups_all, rlen);

    if (groups_all[dim] > 1) {
        groups_all[dim] = 1;

        ireduceDimLauncher<T, op>(out, oidx, tmp, tidx, dim, threads_y, false,
                                  groups_all, rlen);
        bufferFree(tmp.data);
        bufferFree(tidx);
    }
}

template<typename T, af_op_t op>
void ireduceFirstLauncher(Param out, cl::Buffer *oidx, Param in,
                          cl::Buffer *iidx, const int threads_x,
                          const bool is_first, const uint groups_x,
                          const uint groups_y, Param rlen) {
    ToNumStr<T> toNumStr;
    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(op),
        TemplateArg(is_first),
        TemplateArg(threads_x),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(DIMX, threads_x),
        DefineValue(THREADS_PER_GROUP),
        DefineKeyValue(init, toNumStr(common::Binary<T, op>::init())),
        DefineKeyFromStr(binOpName<op>()),
        DefineKeyValue(CPLX, iscplx<T>()),
        DefineKeyValue(IS_FIRST, is_first),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto ireduceFirst = common::getKernel("ireduce_first_kernel",
                                          {{iops_cl_src, ireduce_first_cl_src}},
                                          targs, options);

    cl::NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    cl::NDRange global(groups_x * in.info.dims[2] * local[0],
                       groups_y * in.info.dims[3] * local[1]);

    uint repeat = divup(in.info.dims[0], (local[0] * groups_x));

    ireduceFirst(cl::EnqueueArgs(getQueue(), global, local), *out.data,
                 out.info, *oidx, *in.data, in.info, *iidx, groups_x, groups_y,
                 repeat, *rlen.data, rlen.info);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op>
void ireduceFirst(Param out, cl::Buffer *oidx, Param in, Param rlen) {
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

    ireduceFirstLauncher<T, op>(tmp, tidx, in, tidx, threads_x, true, groups_x,
                                groups_y, rlen);

    if (groups_x > 1) {
        ireduceFirstLauncher<T, op>(out, oidx, tmp, tidx, threads_x, false, 1,
                                    groups_y, rlen);

        bufferFree(tmp.data);
        bufferFree(tidx);
    }
}

template<typename T, af_op_t op>
void ireduce(Param out, cl::Buffer *oidx, Param in, int dim, Param rlen) {
    cl::Buffer buf;
    if (rlen.info.dims[0] * rlen.info.dims[1] * rlen.info.dims[2] *
            rlen.info.dims[3] ==
        0) {
        // empty opencl::Param() does not have nullptr by default
        // set to nullptr explicitly here for consequent kernel calls
        // through cl::Buffer's constructor
        rlen.data = &buf;
    }
    if (dim == 0) {
        ireduceFirst<T, op>(out, oidx, in, rlen);
    } else {
        ireduceDim<T, op>(out, oidx, in, dim, rlen);
    }
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
T ireduceAll(uint *loc, Param in) {
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
        auto buff = std::make_unique<cl::Buffer>();
        rlen.data = buff.get();
        ireduceFirstLauncher<T, op>(tmp, tidx, in, tidx, threads_x, true,
                                    groups_x, groups_y, rlen);

        std::vector<T> h_ptr(tmp_elements);
        std::vector<uint> h_iptr(tmp_elements);

        getQueue().enqueueReadBuffer(*tmp.get(), CL_TRUE, 0,
                                     sizeof(T) * tmp_elements, h_ptr.data());
        getQueue().enqueueReadBuffer(
            *tidx, CL_TRUE, 0, sizeof(uint) * tmp_elements, h_iptr.data());

        T *h_ptr_raw     = h_ptr.data();
        uint *h_iptr_raw = h_iptr.data();

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
        std::unique_ptr<T[]> h_ptr(new T[in_elements]);
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
}  // namespace arrayfire

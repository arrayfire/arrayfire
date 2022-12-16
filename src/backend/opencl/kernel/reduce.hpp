/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <Param.hpp>
#include <common/Binary.hpp>
#include <common/Transform.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel/config.hpp>
#include <kernel/names.hpp>
#include <kernel_headers/ops.hpp>
#include <kernel_headers/reduce_all.hpp>
#include <kernel_headers/reduce_dim.hpp>
#include <kernel_headers/reduce_first.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename Ti, typename To, af_op_t op>
void reduceDimLauncher(Param out, Param in, const int dim, const uint threads_y,
                       const uint groups_all[4], int change_nan,
                       double nanval) {
    ToNumStr<To> toNumStr;
    std::array<TemplateArg, 5> targs = {
        TemplateTypename<Ti>(), TemplateTypename<To>(), TemplateArg(dim),
        TemplateArg(op),        TemplateArg(threads_y),
    };
    std::array<std::string, 10> options = {
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(kDim, dim),
        DefineKeyValue(DIMY, threads_y),
        DefineValue(THREADS_X),
        DefineKeyValue(init, toNumStr(common::Binary<To, op>::init())),
        DefineKeyFromStr(binOpName<op>()),
        DefineKeyValue(CPLX, iscplx<Ti>()),
        getTypeBuildDefinition<Ti, To>()};

    auto reduceDim = common::getKernel(
        "reduce_dim_kernel", std::array{ops_cl_src, reduce_dim_cl_src}, targs,
        options);

    cl::NDRange local(THREADS_X, threads_y);
    cl::NDRange global(groups_all[0] * groups_all[2] * local[0],
                       groups_all[1] * groups_all[3] * local[1]);

    reduceDim(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *in.data, in.info, groups_all[0], groups_all[1], groups_all[dim],
              change_nan, scalar<To>(nanval));
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename To, af_op_t op>
void reduceDim(Param out, Param in, int change_nan, double nanval, int dim) {
    uint threads_y = std::min(THREADS_Y, nextpow2(in.info.dims[dim]));
    uint threads_x = THREADS_X;

    uint groups_all[] = {(uint)divup(in.info.dims[0], threads_x),
                         (uint)in.info.dims[1], (uint)in.info.dims[2],
                         (uint)in.info.dims[3]};

    groups_all[dim] = divup(in.info.dims[dim], threads_y * REPEAT);

    Param tmp = out;

    int tmp_elements = 1;
    if (groups_all[dim] > 1) {
        tmp.info.dims[dim] = groups_all[dim];

        for (int k = 0; k < 4; k++) tmp_elements *= tmp.info.dims[k];

        tmp.data = bufferAlloc(tmp_elements * sizeof(To));

        for (int k = dim + 1; k < 4; k++)
            tmp.info.strides[k] *= groups_all[dim];
    }

    reduceDimLauncher<Ti, To, op>(tmp, in, dim, threads_y, groups_all,
                                  change_nan, nanval);

    if (groups_all[dim] > 1) {
        groups_all[dim] = 1;

        if (op == af_notzero_t) {
            reduceDimLauncher<To, To, af_add_t>(out, tmp, dim, threads_y,
                                                groups_all, change_nan, nanval);
        } else {
            reduceDimLauncher<To, To, op>(out, tmp, dim, threads_y, groups_all,
                                          change_nan, nanval);
        }
        bufferFree(tmp.data);
    }
}

template<typename Ti, typename To, af_op_t op>
void reduceAllLauncher(Param out, Param in, const uint groups_x,
                       const uint groups_y, const uint threads_x,
                       int change_nan, double nanval) {
    ToNumStr<To> toNumStr;
    std::array<TemplateArg, 4> targs = {
        TemplateTypename<Ti>(),
        TemplateTypename<To>(),
        TemplateArg(op),
        TemplateArg(threads_x),
    };
    std::array<std::string, 9> options = {
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(DIMX, threads_x),
        DefineValue(THREADS_PER_GROUP),
        DefineKeyValue(init, toNumStr(common::Binary<To, op>::init())),
        DefineKeyFromStr(binOpName<op>()),
        DefineKeyValue(CPLX, iscplx<Ti>()),
        getTypeBuildDefinition<Ti, To>()};

    auto reduceAll = common::getKernel(
        "reduce_all_kernel", std::array{ops_cl_src, reduce_all_cl_src}, targs,
        options);

    cl::NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    cl::NDRange global(groups_x * in.info.dims[2] * local[0],
                       groups_y * in.info.dims[3] * local[1]);

    uint repeat = divup(in.info.dims[0], (local[0] * groups_x));

    long tmp_elements = groups_x * in.info.dims[2] * groups_y * in.info.dims[3];
    if (tmp_elements > UINT_MAX) {
        AF_ERROR("Too many blocks requested (retirementCount == unsigned)",
                 AF_ERR_RUNTIME);
    }
    Array<To> tmp                   = createEmptyArray<To>(tmp_elements);
    Array<unsigned> retirementCount = createValueArray<unsigned>(1, 0);
    Param p_tmp(tmp);
    Param p_Count(retirementCount);

    reduceAll(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *p_Count.data, *p_tmp.data, p_tmp.info, *in.data, in.info,
              groups_x, groups_y, repeat, change_nan, scalar<To>(nanval));
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename To, af_op_t op>
void reduceFirstLauncher(Param out, Param in, const uint groups_x,
                         const uint groups_y, const uint threads_x,
                         int change_nan, double nanval) {
    ToNumStr<To> toNumStr;
    std::array<TemplateArg, 4> targs = {
        TemplateTypename<Ti>(),
        TemplateTypename<To>(),
        TemplateArg(op),
        TemplateArg(threads_x),
    };
    std::array<std::string, 9> options = {
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(DIMX, threads_x),
        DefineValue(THREADS_PER_GROUP),
        DefineKeyValue(init, toNumStr(common::Binary<To, op>::init())),
        DefineKeyFromStr(binOpName<op>()),
        DefineKeyValue(CPLX, iscplx<Ti>()),
        getTypeBuildDefinition<Ti, To>()};

    auto reduceFirst = common::getKernel(
        "reduce_first_kernel", std::array{ops_cl_src, reduce_first_cl_src},
        targs, options);

    cl::NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    cl::NDRange global(groups_x * in.info.dims[2] * local[0],
                       groups_y * in.info.dims[3] * local[1]);

    uint repeat = divup(in.info.dims[0], (local[0] * groups_x));

    reduceFirst(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
                *in.data, in.info, groups_x, groups_y, repeat, change_nan,
                scalar<To>(nanval));
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename To, af_op_t op>
void reduceFirst(Param out, Param in, int change_nan, double nanval) {
    uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_GROUP);
    uint threads_y = THREADS_PER_GROUP / threads_x;

    uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
    uint groups_y = divup(in.info.dims[1], threads_y);

    Param tmp = out;

    if (groups_x > 1) {
        tmp.data = bufferAlloc(groups_x * in.info.dims[1] * in.info.dims[2] *
                               in.info.dims[3] * sizeof(To));

        tmp.info.dims[0] = groups_x;
        for (int k = 1; k < 4; k++) tmp.info.strides[k] *= groups_x;
    }

    reduceFirstLauncher<Ti, To, op>(tmp, in, groups_x, groups_y, threads_x,
                                    change_nan, nanval);

    if (groups_x > 1) {
        // FIXME: Is there an alternative to the if condition ?
        if (op == af_notzero_t) {
            reduceFirstLauncher<To, To, af_add_t>(
                out, tmp, 1, groups_y, threads_x, change_nan, nanval);
        } else {
            reduceFirstLauncher<To, To, op>(out, tmp, 1, groups_y, threads_x,
                                            change_nan, nanval);
        }
        bufferFree(tmp.data);
    }
}

template<typename Ti, typename To, af_op_t op>
void reduce(Param out, Param in, int dim, int change_nan, double nanval) {
    if (dim == 0)
        return reduceFirst<Ti, To, op>(out, in, change_nan, nanval);
    else
        return reduceDim<Ti, To, op>(out, in, change_nan, nanval, dim);
}

template<typename Ti, typename To, af_op_t op>
void reduceAll(Param out, Param in, int change_nan, double nanval) {
    int in_elements =
        in.info.dims[0] * in.info.dims[1] * in.info.dims[2] * in.info.dims[3];

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
    reduceAllLauncher<Ti, To, op>(out, in, groups_x, groups_y, threads_x,
                                  change_nan, nanval);
}

}  // namespace kernel

}  // namespace opencl
}  // namespace arrayfire

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
#include <common/Transform.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel/config.hpp>
#include <kernel/names.hpp>
#include <kernel_headers/mean_dim.hpp>
#include <kernel_headers/mean_first.hpp>
#include <kernel_headers/mean_ops.hpp>
#include <memory.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename T, typename Tw>
struct MeanOp {
    T runningMean;
    Tw runningCount;
    MeanOp(T mean, Tw count) : runningMean(mean), runningCount(count) {}

    void operator()(T newMean, Tw newCount) {
        if ((newCount != 0) || (runningCount != 0)) {
            Tw runningScale = runningCount;
            Tw newScale     = newCount;
            runningCount += newCount;
            runningScale = runningScale / runningCount;
            newScale     = newScale / (Tw)runningCount;
            runningMean  = (runningScale * runningMean) + (newScale * newMean);
        }
    }
};

template<>
struct MeanOp<cfloat, float> {
    cfloat runningMean;
    float runningCount;
    MeanOp(cfloat mean, float count) : runningMean(mean), runningCount(count) {}

    void operator()(cfloat newMean, float newCount) {
        if ((newCount != 0) || (runningCount != 0)) {
            float runningScale = runningCount;
            float newScale     = newCount;
            runningCount += newCount;
            runningScale = runningScale / runningCount;
            newScale     = newScale / (float)runningCount;
            runningMean.s[0] =
                (runningScale * runningMean.s[0]) + (newScale * newMean.s[0]);
            runningMean.s[1] =
                (runningScale * runningMean.s[1]) + (newScale * newMean.s[1]);
        }
    }
};

template<>
struct MeanOp<cdouble, double> {
    cdouble runningMean;
    double runningCount;
    MeanOp(cdouble mean, double count)
        : runningMean(mean), runningCount(count) {}

    void operator()(cdouble newMean, double newCount) {
        if ((newCount != 0) || (runningCount != 0)) {
            double runningScale = runningCount;
            double newScale     = newCount;
            runningCount += newCount;
            runningScale = runningScale / runningCount;
            newScale     = newScale / (double)runningCount;
            runningMean.s[0] =
                (runningScale * runningMean.s[0]) + (newScale * newMean.s[0]);
            runningMean.s[1] =
                (runningScale * runningMean.s[1]) + (newScale * newMean.s[1]);
        }
    }
};

template<typename Ti, typename Tw, typename To>
void meanDimLauncher(Param out, Param owt, Param in, Param inWeight,
                     const int dim, const int threads_y,
                     const uint groups_all[4]) {
    using cl::EnqueueArgs;
    using cl::NDRange;

    bool input_weight = ((inWeight.info.dims[0] * inWeight.info.dims[1] *
                          inWeight.info.dims[2] * inWeight.info.dims[3]) != 0);

    bool output_weight = ((owt.info.dims[0] * owt.info.dims[1] *
                           owt.info.dims[2] * owt.info.dims[3]) != 0);

    ToNumStr<To> toNumStr;
    ToNumStr<Tw> twNumStr;
    common::Transform<uint, Tw, af_add_t> transform_weight;

    std::vector<TemplateArg> targs = {
        TemplateTypename<Ti>(),     TemplateTypename<To>(),
        TemplateTypename<Tw>(),     TemplateArg(dim),
        TemplateArg(threads_y),     TemplateArg(input_weight),
        TemplateArg(output_weight),
    };
    std::vector<std::string> options = {
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(Tw, dtype_traits<Tw>::getName()),
        DefineKeyValue(kDim, dim),
        DefineKeyValue(DIMY, threads_y),
        DefineValue(THREADS_X),
        DefineKeyValue(init_To, toNumStr(common::Binary<To, af_add_t>::init())),
        DefineKeyValue(init_Tw, twNumStr(transform_weight(0))),
        DefineKeyValue(one_Tw, twNumStr(transform_weight(1))),
    };
    options.emplace_back(getTypeBuildDefinition<Ti, To>());
    if (input_weight) { options.emplace_back(DefineKey(INPUT_WEIGHT)); }
    if (output_weight) { options.emplace_back(DefineKey(OUTPUT_WEIGHT)); }

    auto meanOp = common::getKernel(
        "meanDim", {{mean_ops_cl_src, mean_dim_cl_src}}, targs, options);

    NDRange local(THREADS_X, threads_y);
    NDRange global(groups_all[0] * groups_all[2] * local[0],
                   groups_all[1] * groups_all[3] * local[1]);

    if (input_weight && output_weight) {
        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *owt.data, owt.info, *in.data, in.info, *inWeight.data,
               inWeight.info, groups_all[0], groups_all[1], groups_all[dim]);
    } else if (!input_weight && !output_weight) {
        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *in.data, in.info, groups_all[0], groups_all[1],
               groups_all[dim]);
    } else if (input_weight && !output_weight) {
        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *in.data, in.info, *inWeight.data, inWeight.info, groups_all[0],
               groups_all[1], groups_all[dim]);
    } else if (!input_weight && output_weight) {
        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *owt.data, owt.info, *in.data, in.info, groups_all[0],
               groups_all[1], groups_all[dim]);
    }
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tw, typename To>
void meanDim(Param out, Param in, Param inWeight, int dim) {
    uint threads_y = std::min(THREADS_Y, nextpow2(in.info.dims[dim]));
    uint threads_x = THREADS_X;

    uint groups_all[] = {(uint)divup(in.info.dims[0], threads_x),
                         (uint)in.info.dims[1], (uint)in.info.dims[2],
                         (uint)in.info.dims[3]};

    groups_all[dim] = divup(in.info.dims[dim], threads_y * REPEAT);

    if (groups_all[dim] > 1) {
        dim4 d(4, out.info.dims);
        d[dim]              = groups_all[dim];
        Array<To> tmpOut    = createEmptyArray<To>(d);
        Array<Tw> tmpWeight = createEmptyArray<Tw>(d);
        meanDimLauncher<Ti, Tw, To>(tmpOut, tmpWeight, in, inWeight, dim,
                                    threads_y, groups_all);

        Param owt;
        groups_all[dim] = 1;
        meanDimLauncher<Ti, Tw, To>(out, owt, tmpOut, tmpWeight, dim, threads_y,
                                    groups_all);
    } else {
        Param tmpWeight;
        meanDimLauncher<Ti, Tw, To>(out, tmpWeight, in, inWeight, dim,
                                    threads_y, groups_all);
    }
}

template<typename Ti, typename Tw, typename To>
void meanFirstLauncher(Param out, Param owt, Param in, Param inWeight,
                       const int threads_x, const uint groups_x,
                       const uint groups_y) {
    using cl::EnqueueArgs;
    using cl::NDRange;

    bool input_weight = ((inWeight.info.dims[0] * inWeight.info.dims[1] *
                          inWeight.info.dims[2] * inWeight.info.dims[3]) != 0);

    bool output_weight = ((owt.info.dims[0] * owt.info.dims[1] *
                           owt.info.dims[2] * owt.info.dims[3]) != 0);
    ToNumStr<To> toNumStr;
    ToNumStr<Tw> twNumStr;
    common::Transform<uint, Tw, af_add_t> transform_weight;

    std::vector<TemplateArg> targs = {
        TemplateTypename<Ti>(),    TemplateTypename<To>(),
        TemplateTypename<Tw>(),    TemplateArg(threads_x),
        TemplateArg(input_weight), TemplateArg(output_weight),
    };
    std::vector<std::string> options = {
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(Tw, dtype_traits<Tw>::getName()),
        DefineKeyValue(DIMX, threads_x),
        DefineValue(THREADS_PER_GROUP),
        DefineKeyValue(init_To, toNumStr(common::Binary<To, af_add_t>::init())),
        DefineKeyValue(init_Tw, twNumStr(transform_weight(0))),
        DefineKeyValue(one_Tw, twNumStr(transform_weight(1))),
    };
    options.emplace_back(getTypeBuildDefinition<Ti, To>());
    if (input_weight) { options.emplace_back(DefineKey(INPUT_WEIGHT)); }
    if (output_weight) { options.emplace_back(DefineKey(OUTPUT_WEIGHT)); }

    auto meanOp = common::getKernel(
        "meanFirst", {{mean_ops_cl_src, mean_first_cl_src}}, targs, options);

    NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    NDRange global(groups_x * in.info.dims[2] * local[0],
                   groups_y * in.info.dims[3] * local[1]);

    uint repeat = divup(in.info.dims[0], (local[0] * groups_x));

    if (input_weight && output_weight) {
        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *owt.data, owt.info, *in.data, in.info, *inWeight.data,
               inWeight.info, groups_x, groups_y, repeat);
    } else if (!input_weight && !output_weight) {
        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *in.data, in.info, groups_x, groups_y, repeat);
    } else if (input_weight && !output_weight) {
        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *in.data, in.info, *inWeight.data, inWeight.info, groups_x,
               groups_y, repeat);
    } else if (!input_weight && output_weight) {
        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *owt.data, owt.info, *in.data, in.info, groups_x, groups_y,
               repeat);
    }
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tw, typename To>
void meanFirst(Param out, Param in, Param inWeight) {
    uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_GROUP);
    uint threads_y = THREADS_PER_GROUP / threads_x;

    uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
    uint groups_y = divup(in.info.dims[1], threads_y);

    Param tmpOut = out;
    Param noWeight;
    noWeight.info.offset = 0;
    for (int k = 0; k < 4; ++k) {
        noWeight.info.dims[k]    = 0;
        noWeight.info.strides[k] = 0;
    }
    // Does not matter what the value is it will not be used. Just needs to be
    // valid.
    noWeight.data = inWeight.data;

    Param tmpWeight = noWeight;

    if (groups_x > 1) {
        tmpOut.data = bufferAlloc(groups_x * in.info.dims[1] * in.info.dims[2] *
                                  in.info.dims[3] * sizeof(To));

        tmpWeight.data =
            bufferAlloc(groups_x * in.info.dims[1] * in.info.dims[2] *
                        in.info.dims[3] * sizeof(Tw));

        tmpOut.info.dims[0] = groups_x;
        for (int k = 1; k < 4; k++) tmpOut.info.strides[k] *= groups_x;
        tmpWeight.info = tmpOut.info;
    }

    meanFirstLauncher<Ti, Tw, To>(tmpOut, tmpWeight, in, inWeight, threads_x,
                                  groups_x, groups_y);

    if (groups_x > 1) {
        // No Weight is needed when writing out the output.
        meanFirstLauncher<Ti, Tw, To>(out, noWeight, tmpOut, tmpWeight,
                                      threads_x, 1, groups_y);

        bufferFree(tmpOut.data);
        bufferFree(tmpWeight.data);
    }
}

template<typename Ti, typename Tw, typename To>
void meanWeighted(Param out, Param in, Param inWeight, int dim) {
    if (dim == 0)
        return meanFirst<Ti, Tw, To>(out, in, inWeight);
    else
        return meanDim<Ti, Tw, To>(out, in, inWeight, dim);
}

template<typename Ti, typename Tw, typename To>
void mean(Param out, Param in, int dim) {
    Param noWeight;
    meanWeighted<Ti, Tw, To>(out, in, noWeight, dim);
}

template<typename T, typename Tw>
T meanAllWeighted(Param in, Param inWeight) {
    int in_elements =
        in.info.dims[0] * in.info.dims[1] * in.info.dims[2] * in.info.dims[3];

    // FIXME: Use better heuristics to get to the optimum number
    if (in_elements > 4096) {
        bool in_is_linear = (in.info.strides[0] == 1);
        bool wt_is_linear = (in.info.strides[0] == 1);
        for (int k = 1; k < 4; k++) {
            in_is_linear &= (in.info.strides[k] ==
                             (in.info.strides[k - 1] * in.info.dims[k - 1]));
            wt_is_linear &=
                (inWeight.info.strides[k] ==
                 (inWeight.info.strides[k - 1] * inWeight.info.dims[k - 1]));
        }

        if (in_is_linear && wt_is_linear) {
            in.info.dims[0] = in_elements;
            for (int k = 1; k < 4; k++) {
                in.info.dims[k]    = 1;
                in.info.strides[k] = in_elements;
            }
            inWeight.info = in.info;
        }

        uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
        threads_x      = std::min(threads_x, THREADS_PER_GROUP);
        uint threads_y = THREADS_PER_GROUP / threads_x;

        uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
        uint groups_y = divup(in.info.dims[1], threads_y);

        Array<T> tmpOut     = createEmptyArray<T>(groups_x);
        Array<Tw> tmpWeight = createEmptyArray<Tw>(groups_x);

        meanFirstLauncher<T, Tw, T>(tmpOut, tmpWeight, in, inWeight, threads_x,
                                    groups_x, groups_y);

        std::vector<T> h_ptr(tmpOut.elements());
        std::vector<Tw> h_wptr(tmpWeight.elements());

        getQueue().enqueueReadBuffer(*tmpOut.get(), CL_TRUE, 0,
                                     sizeof(T) * tmpOut.elements(),
                                     h_ptr.data());
        getQueue().enqueueReadBuffer(*tmpWeight.get(), CL_TRUE, 0,
                                     sizeof(Tw) * tmpWeight.elements(),
                                     h_wptr.data());

        compute_t<T> initial = static_cast<compute_t<T>>(h_ptr[0]);
        compute_t<Tw> w      = static_cast<compute_t<Tw>>(h_wptr[0]);
        MeanOp<compute_t<T>, compute_t<Tw>> Op(initial, w);
        for (int i = 1; i < (int)tmpOut.elements(); i++) {
            Op(compute_t<T>(h_ptr[i]), compute_t<Tw>(h_wptr[i]));
        }

        return static_cast<T>(Op.runningMean);
    } else {
        std::vector<T> h_ptr(in_elements);
        std::vector<Tw> h_wptr(in_elements);

        getQueue().enqueueReadBuffer(*in.data, CL_TRUE,
                                     sizeof(T) * in.info.offset,
                                     sizeof(T) * in_elements, h_ptr.data());
        getQueue().enqueueReadBuffer(*inWeight.data, CL_TRUE,
                                     sizeof(Tw) * inWeight.info.offset,
                                     sizeof(Tw) * in_elements, h_wptr.data());

        compute_t<T> initial = static_cast<compute_t<T>>(h_ptr[0]);
        compute_t<Tw> w      = static_cast<compute_t<Tw>>(h_wptr[0]);
        MeanOp<compute_t<T>, compute_t<Tw>> Op(initial, w);
        for (int i = 1; i < (int)in_elements; i++) {
            Op(compute_t<T>(h_ptr[i]), compute_t<Tw>(h_wptr[i]));
        }

        return static_cast<T>(Op.runningMean);
    }
}

template<typename Ti, typename Tw, typename To>
To meanAll(Param in) {
    int in_elements =
        in.info.dims[0] * in.info.dims[1] * in.info.dims[2] * in.info.dims[3];
    bool is_linear = (in.info.strides[0] == 1);
    for (int k = 1; k < 4; k++) {
        is_linear &= (in.info.strides[k] ==
                      (in.info.strides[k - 1] * in.info.dims[k - 1]));
    }

    // FIXME: Use better heuristics to get to the optimum number
    if (in_elements > 4096 || !is_linear) {
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

        dim4 outDims(groups_x, in.info.dims[1], in.info.dims[2],
                     in.info.dims[3]);
        Array<To> tmpOut = createEmptyArray<To>(outDims);
        Array<Tw> tmpCt  = createEmptyArray<Tw>(outDims);

        Param iWt;
        meanFirstLauncher<Ti, Tw, To>(tmpOut, tmpCt, in, iWt, threads_x,
                                      groups_x, groups_y);

        std::vector<To> h_ptr(tmpOut.elements());
        std::vector<Tw> h_cptr(tmpOut.elements());

        getQueue().enqueueReadBuffer(*tmpOut.get(), CL_TRUE, 0,
                                     sizeof(To) * tmpOut.elements(),
                                     h_ptr.data());
        getQueue().enqueueReadBuffer(*tmpCt.get(), CL_TRUE, 0,
                                     sizeof(Tw) * tmpCt.elements(),
                                     h_cptr.data());

        compute_t<To> initial = static_cast<compute_t<To>>(h_ptr[0]);
        compute_t<Tw> w       = static_cast<compute_t<Tw>>(h_cptr[0]);
        MeanOp<compute_t<To>, compute_t<Tw>> Op(initial, w);
        for (int i = 1; i < (int)h_ptr.size(); i++) {
            Op(compute_t<To>(h_ptr[i]), compute_t<Tw>(h_cptr[i]));
        }

        return static_cast<To>(Op.runningMean);
    } else {
        std::vector<Ti> h_ptr(in_elements);

        getQueue().enqueueReadBuffer(*in.data, CL_TRUE,
                                     sizeof(Ti) * in.info.offset,
                                     sizeof(Ti) * in_elements, h_ptr.data());

        // TODO : MeanOp with (Tw)1
        common::Transform<Ti, compute_t<To>, af_add_t> transform;
        common::Transform<uint, compute_t<Tw>, af_add_t> transform_weight;
        MeanOp<compute_t<To>, compute_t<Tw>> Op(transform(h_ptr[0]),
                                                transform_weight(1));
        for (int i = 1; i < (int)in_elements; i++) {
            Op(transform(h_ptr[i]), transform_weight(1));
        }

        return static_cast<To>(Op.runningMean);
    }
}

}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire

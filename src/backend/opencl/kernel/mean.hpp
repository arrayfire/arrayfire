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
#include <kernel_headers/mean_dim.hpp>
#include <kernel_headers/mean_first.hpp>
#include <kernel_headers/mean_ops.hpp>
#include <memory.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <type_util.hpp>
#include "config.hpp"
#include "names.hpp"

#include <map>
#include <mutex>
#include <string>
#include <vector>

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;
using std::vector;

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
void mean_dim_launcher(Param out, Param owt, Param in, Param inWeight,
                       const int dim, const int threads_y,
                       const uint groups_all[4]) {
    bool input_weight = ((inWeight.info.dims[0] * inWeight.info.dims[1] *
                          inWeight.info.dims[2] * inWeight.info.dims[3]) != 0);

    bool output_weight = ((owt.info.dims[0] * owt.info.dims[1] *
                           owt.info.dims[2] * owt.info.dims[3]) != 0);

    std::string ref_name =
        std::string("mean_") + std::to_string(dim) + std::string("_") +
        std::string(dtype_traits<Ti>::getName()) + std::string("_") +
        std::string(dtype_traits<Tw>::getName()) + std::string("_") +
        std::string(dtype_traits<To>::getName()) + std::string("_") +
        std::to_string(threads_y) + std::string("_") +
        std::to_string(input_weight) + std::string("_") +
        std::to_string(output_weight);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        ToNumStr<To> toNumStr;
        ToNumStr<Tw> twNumStr;
        Transform<uint, Tw, af_add_t> transform_weight;

        std::ostringstream options;
        options << " -D Ti=" << dtype_traits<Ti>::getName()
                << " -D Tw=" << dtype_traits<Tw>::getName()
                << " -D To=" << dtype_traits<To>::getName() << " -D dim=" << dim
                << " -D DIMY=" << threads_y << " -D THREADS_X=" << THREADS_X
                << " -D init_To=" << toNumStr(Binary<To, af_add_t>::init())
                << " -D init_Tw=" << twNumStr(transform_weight(0))
                << " -D one_Tw=" << twNumStr(transform_weight(1));

        if (input_weight) { options << " -D INPUT_WEIGHT"; }
        if (output_weight) { options << " -D OUTPUT_WEIGHT"; }

        if (std::is_same<Ti, double>::value ||
            std::is_same<Ti, cdouble>::value ||
            std::is_same<To, double>::value) {
            options << " -D USE_DOUBLE";
        }

        const char *ker_strs[] = {mean_ops_cl, mean_dim_cl};
        const int ker_lens[]   = {mean_ops_cl_len, mean_dim_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "mean_dim_kernel");

        addKernelToCache(device, ref_name, entry);
    }

    NDRange local(THREADS_X, threads_y);
    NDRange global(groups_all[0] * groups_all[2] * local[0],
                   groups_all[1] * groups_all[3] * local[1]);

    if (input_weight && output_weight) {
        auto meanOp =
            KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer, KParam,
                          Buffer, KParam, uint, uint, uint>(*entry.ker);

        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *owt.data, owt.info, *in.data, in.info, *inWeight.data,
               inWeight.info, groups_all[0], groups_all[1], groups_all[dim]);
    } else if (!input_weight && !output_weight) {
        auto meanOp =
            KernelFunctor<Buffer, KParam, Buffer, KParam, uint, uint, uint>(
                *entry.ker);

        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *in.data, in.info, groups_all[0], groups_all[1],
               groups_all[dim]);
    } else if (input_weight && !output_weight) {
        auto meanOp = KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer,
                                    KParam, uint, uint, uint>(*entry.ker);

        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *in.data, in.info, *inWeight.data, inWeight.info, groups_all[0],
               groups_all[1], groups_all[dim]);
    } else if (!input_weight && output_weight) {
        auto meanOp = KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer,
                                    KParam, uint, uint, uint>(*entry.ker);

        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *owt.data, owt.info, *in.data, in.info, groups_all[0],
               groups_all[1], groups_all[dim]);
    }

    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tw, typename To>
void mean_dim(Param out, Param in, Param inWeight, int dim) {
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
        mean_dim_launcher<Ti, Tw, To>(tmpOut, tmpWeight, in, inWeight, dim,
                                      threads_y, groups_all);

        Param owt;
        groups_all[dim] = 1;
        mean_dim_launcher<Ti, Tw, To>(out, owt, tmpOut, tmpWeight, dim,
                                      threads_y, groups_all);
    } else {
        Param tmpWeight;
        mean_dim_launcher<Ti, Tw, To>(out, tmpWeight, in, inWeight, dim,
                                      threads_y, groups_all);
    }
}

template<typename Ti, typename Tw, typename To>
void mean_first_launcher(Param out, Param owt, Param in, Param inWeight,
                         const int threads_x, const uint groups_x,
                         const uint groups_y) {
    bool input_weight = ((inWeight.info.dims[0] * inWeight.info.dims[1] *
                          inWeight.info.dims[2] * inWeight.info.dims[3]) != 0);

    bool output_weight = ((owt.info.dims[0] * owt.info.dims[1] *
                           owt.info.dims[2] * owt.info.dims[3]) != 0);

    std::string ref_name =
        std::string("mean_0_") + std::string(dtype_traits<Ti>::getName()) +
        std::string("_") + std::string(dtype_traits<Tw>::getName()) +
        std::string("_") + std::string(dtype_traits<To>::getName()) +
        std::string("_") + std::to_string(threads_x) + std::string("_") +
        std::to_string(input_weight) + std::string("_") +
        std::to_string(output_weight);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        ToNumStr<To> toNumStr;
        ToNumStr<Tw> twNumStr;
        Transform<uint, Tw, af_add_t> transform_weight;

        std::ostringstream options;
        options << " -D Ti=" << dtype_traits<Ti>::getName()
                << " -D Tw=" << dtype_traits<Tw>::getName()
                << " -D To=" << dtype_traits<To>::getName()
                << " -D DIMX=" << threads_x
                << " -D THREADS_PER_GROUP=" << THREADS_PER_GROUP
                << " -D init_To=" << toNumStr(Binary<To, af_add_t>::init())
                << " -D init_Tw=" << twNumStr(transform_weight(0))
                << " -D one_Tw=" << twNumStr(transform_weight(1));

        if (input_weight) { options << " -D INPUT_WEIGHT"; }
        if (output_weight) { options << " -D OUTPUT_WEIGHT"; }

        if (std::is_same<Ti, double>::value ||
            std::is_same<Ti, cdouble>::value ||
            std::is_same<To, double>::value) {
            options << " -D USE_DOUBLE";
        }

        const char *ker_strs[] = {mean_ops_cl, mean_first_cl};
        const int ker_lens[]   = {mean_ops_cl_len, mean_first_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "mean_first_kernel");

        addKernelToCache(device, ref_name, entry);
    }

    NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    NDRange global(groups_x * in.info.dims[2] * local[0],
                   groups_y * in.info.dims[3] * local[1]);

    uint repeat = divup(in.info.dims[0], (local[0] * groups_x));

    if (input_weight && output_weight) {
        auto meanOp =
            KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer, KParam,
                          Buffer, KParam, uint, uint, uint>(*entry.ker);
        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *owt.data, owt.info, *in.data, in.info, *inWeight.data,
               inWeight.info, groups_x, groups_y, repeat);
    } else if (!input_weight && !output_weight) {
        auto meanOp =
            KernelFunctor<Buffer, KParam, Buffer, KParam, uint, uint, uint>(
                *entry.ker);
        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *in.data, in.info, groups_x, groups_y, repeat);
    } else if (input_weight && !output_weight) {
        auto meanOp = KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer,
                                    KParam, uint, uint, uint>(*entry.ker);
        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *in.data, in.info, *inWeight.data, inWeight.info, groups_x,
               groups_y, repeat);
    } else if (!input_weight && output_weight) {
        auto meanOp = KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer,
                                    KParam, uint, uint, uint>(*entry.ker);
        meanOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
               *owt.data, owt.info, *in.data, in.info, groups_x, groups_y,
               repeat);
    }

    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tw, typename To>
void mean_first(Param out, Param in, Param inWeight) {
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

    mean_first_launcher<Ti, Tw, To>(tmpOut, tmpWeight, in, inWeight, threads_x,
                                    groups_x, groups_y);

    if (groups_x > 1) {
        // No Weight is needed when writing out the output.
        mean_first_launcher<Ti, Tw, To>(out, noWeight, tmpOut, tmpWeight,
                                        threads_x, 1, groups_y);

        bufferFree(tmpOut.data);
        bufferFree(tmpWeight.data);
    }
}

template<typename Ti, typename Tw, typename To>
void mean_weighted(Param out, Param in, Param inWeight, int dim) {
    if (dim == 0)
        return mean_first<Ti, Tw, To>(out, in, inWeight);
    else
        return mean_dim<Ti, Tw, To>(out, in, inWeight, dim);
}

template<typename Ti, typename Tw, typename To>
void mean(Param out, Param in, int dim) {
    Param noWeight;
    mean_weighted<Ti, Tw, To>(out, in, noWeight, dim);
}

template<typename T, typename Tw>
T mean_all_weighted(Param in, Param inWeight) {
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

        mean_first_launcher<T, Tw, T>(tmpOut, tmpWeight, in, inWeight,
                                      threads_x, groups_x, groups_y);

        vector<T> h_ptr(tmpOut.elements());
        vector<Tw> h_wptr(tmpWeight.elements());

        getQueue().enqueueReadBuffer(*tmpOut.get(), CL_TRUE, 0,
                                     sizeof(T) * tmpOut.elements(),
                                     h_ptr.data());
        getQueue().enqueueReadBuffer(*tmpWeight.get(), CL_TRUE, 0,
                                     sizeof(Tw) * tmpWeight.elements(),
                                     h_wptr.data());

        MeanOp<T, Tw> Op(h_ptr[0], h_wptr[0]);
        for (int i = 1; i < (int)tmpOut.elements(); i++) {
            Op(h_ptr[i], h_wptr[i]);
        }

        return Op.runningMean;

    } else {
        vector<T> h_ptr(in_elements);
        vector<Tw> h_wptr(in_elements);

        getQueue().enqueueReadBuffer(*in.data, CL_TRUE,
                                     sizeof(T) * in.info.offset,
                                     sizeof(T) * in_elements, h_ptr.data());
        getQueue().enqueueReadBuffer(*inWeight.data, CL_TRUE,
                                     sizeof(Tw) * inWeight.info.offset,
                                     sizeof(Tw) * in_elements, h_wptr.data());

        MeanOp<T, Tw> Op(h_ptr[0], h_wptr[0]);
        for (int i = 1; i < (int)in_elements; i++) { Op(h_ptr[i], h_wptr[i]); }

        return Op.runningMean;
    }
}

template<typename Ti, typename Tw, typename To>
To mean_all(Param in) {
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

        dim4 outDims(groups_x, in.info.dims[1],
                     in.info.dims[2], in.info.dims[3]);
        Array<To> tmpOut = createEmptyArray<To>(outDims);
        Array<Tw> tmpCt  = createEmptyArray<Tw>(outDims);

        Param iWt;
        mean_first_launcher<Ti, Tw, To>(tmpOut, tmpCt, in, iWt, threads_x,
                                        groups_x, groups_y);

        vector<To> h_ptr(tmpOut.elements());
        vector<Tw> h_cptr(tmpOut.elements());

        getQueue().enqueueReadBuffer(*tmpOut.get(), CL_TRUE, 0,
                                     sizeof(To) * tmpOut.elements(),
                                     h_ptr.data());
        getQueue().enqueueReadBuffer(*tmpCt.get(), CL_TRUE, 0,
                                     sizeof(Tw) * tmpCt.elements(),
                                     h_cptr.data());

        MeanOp<To, Tw> Op(h_ptr[0], h_cptr[0]);
        for (int i = 1; i < (int)h_ptr.size(); i++) { Op(h_ptr[i], h_cptr[i]); }

        return Op.runningMean;
    } else {
        vector<Ti> h_ptr(in_elements);

        getQueue().enqueueReadBuffer(*in.data, CL_TRUE,
                                     sizeof(Ti) * in.info.offset,
                                     sizeof(Ti) * in_elements, h_ptr.data());

        // TODO : MeanOp with (Tw)1
        Transform<Ti, To, af_add_t> transform;
        Transform<uint, Tw, af_add_t> transform_weight;
        MeanOp<To, Tw> Op(transform(h_ptr[0]), transform_weight(1));
        for (int i = 1; i < (int)in_elements; i++) {
            Op(transform(h_ptr[i]), transform_weight(1));
        }

        return Op.runningMean;
    }
}
}  // namespace kernel

}  // namespace opencl

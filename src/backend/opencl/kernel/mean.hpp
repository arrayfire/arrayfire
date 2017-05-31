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
#include <kernel_headers/mean_first.hpp>
#include <kernel_headers/mean_dim.hpp>
#include <kernel_headers/mops.hpp>
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
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;
using std::unique_ptr;

namespace opencl
{

namespace kernel
{

template<typename T, typename Tw>
struct MeanOp
{
    T runningMean;
    Tw runningCount;
    MeanOp(T mean, Tw count) :
        runningMean(mean), runningCount(count)
    {
    }

    void operator()(T newMean, Tw newCount)
    {
        if ((newCount != 0) || (runningCount != 0)) {
            Tw runningScale = runningCount;
            Tw newScale = newCount;
            runningCount += newCount;
            runningScale = runningScale/runningCount;
            newScale = newScale/(Tw)runningCount;
            runningMean = (runningScale*runningMean) + (newScale*newMean);
        }
    }
};

template<>
struct MeanOp<cfloat, float>
{
    cfloat runningMean;
    float runningCount;
    MeanOp(cfloat mean, float count) :
        runningMean(mean), runningCount(count)
    {
    }

    void operator()(cfloat newMean, float newCount)
    {
        if ((newCount != 0) || (runningCount != 0)) {
            float runningScale = runningCount;
            float newScale = newCount;
            runningCount += newCount;
            runningScale = runningScale/runningCount;
            newScale = newScale/(float)runningCount;
            runningMean.s[0] = (runningScale*runningMean.s[0]) + (newScale*newMean.s[0]);
            runningMean.s[1] = (runningScale*runningMean.s[1]) + (newScale*newMean.s[1]);
        }
    }
};

template<>
struct MeanOp<cdouble, double>
{
    cdouble runningMean;
    double runningCount;
    MeanOp(cdouble mean, double count) :
        runningMean(mean), runningCount(count)
    {
    }

    void operator()(cdouble newMean, double newCount)
    {
        if ((newCount != 0) || (runningCount != 0)) {
            double runningScale = runningCount;
            double newScale = newCount;
            runningCount += newCount;
            runningScale = runningScale/runningCount;
            newScale = newScale/(double)runningCount;
            runningMean.s[0] = (runningScale*runningMean.s[0]) + (newScale*newMean.s[0]);
            runningMean.s[1] = (runningScale*runningMean.s[1]) + (newScale*newMean.s[1]);
        }
    }
};

template<typename Ti, typename Tw, typename To>
void mean_dim_launcher(Param out, Param owt,
        Param in, Param iwt,
        const int dim,
        const int threads_y,
        const uint groups_all[4])
{
    bool input_weight = ((
            iwt.info.dims[0] *
            iwt.info.dims[1] *
            iwt.info.dims[2] *
            iwt.info.dims[3]) != 0);

    bool output_weight = ((
            owt.info.dims[0] *
            owt.info.dims[1] *
            owt.info.dims[2] *
            owt.info.dims[3]) != 0);

    std::string ref_name =
        std::string("mean_") +
        std::to_string(dim) +
        std::string("_") +
        std::string(dtype_traits<Ti>::getName()) +
        std::string("_") +
        std::string(dtype_traits<Tw>::getName()) +
        std::string("_") +
        std::string(dtype_traits<To>::getName()) +
        std::string("_") +
        std::to_string(threads_y) +
        std::string("_") +
        std::to_string(input_weight) +
        std::string("_") +
        std::to_string(output_weight);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog==0 && entry.ker==0) {

        Binary<To, af_add_t> mean;
        ToNumStr<To> toNumStr;
        ToNumStr<Tw> twNumStr;
        Transform<uint, Tw, af_add_t> transform_weight;

        std::ostringstream options;
        options << " -D Ti=" << dtype_traits<Ti>::getName()
            << " -D Tw=" << dtype_traits<Tw>::getName()
            << " -D To=" << dtype_traits<To>::getName()
            << " -D dim=" << dim
            << " -D DIMY=" << threads_y
            << " -D THREADS_X=" << THREADS_X
            << " -D init_To=" << toNumStr(mean.init())
            << " -D init_Tw=" << twNumStr(transform_weight(0))
            << " -D one_Tw=" << twNumStr(transform_weight(1));

        if (input_weight) { options << " -D INPUT_WEIGHT"; }
        if (output_weight) { options << " -D OUTPUT_WEIGHT"; }

        if (std::is_same<Ti, double>::value ||
                std::is_same<Ti, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        const char *ker_strs[] = {mops_cl, mean_dim_cl};
        const int   ker_lens[] = {mops_cl_len, mean_dim_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker = new Kernel(*entry.prog, "mean_dim_kernel");

        addKernelToCache(device, ref_name, entry);
    }

    NDRange local(THREADS_X, threads_y);
    NDRange global(groups_all[0] * groups_all[2] * local[0],
            groups_all[1] * groups_all[3] * local[1]);

    if (input_weight && output_weight) {
        auto meanOp = KernelFunctor<
            Buffer, KParam,
            Buffer, KParam,
            Buffer, KParam,
            Buffer, KParam,
            uint, uint, uint>(*entry.ker);

        meanOp(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info,
                *owt.data, owt.info,
                *in.data, in.info,
                *iwt.data, iwt.info,
                groups_all[0],
                groups_all[1],
                groups_all[dim]);
    } else if (!input_weight && !output_weight) {
        auto meanOp = KernelFunctor<
            Buffer, KParam,
            Buffer, KParam,
            uint, uint, uint>(*entry.ker);

        meanOp(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info,
                *in.data, in.info,
                groups_all[0],
                groups_all[1],
                groups_all[dim]);
    } else if ( input_weight && !output_weight) {
        auto meanOp = KernelFunctor<
            Buffer, KParam,
            Buffer, KParam,
            Buffer, KParam,
            uint, uint, uint>(*entry.ker);

        meanOp(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info,
                *in.data, in.info,
                *iwt.data, iwt.info,
                groups_all[0],
                groups_all[1],
                groups_all[dim]);
    } else if (!input_weight &&  output_weight) {
        auto meanOp = KernelFunctor<
            Buffer, KParam,
            Buffer, KParam,
            Buffer, KParam,
            uint, uint, uint>(*entry.ker);

        meanOp(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info,
                *owt.data, owt.info,
                *in.data, in.info,
                groups_all[0],
                groups_all[1],
                groups_all[dim]);
    }

    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tw, typename To>
void mean_dim(Param out, Param in, Param iwt, int dim)
{
    uint threads_y = std::min(THREADS_Y, nextpow2(in.info.dims[dim]));
    uint threads_x = THREADS_X;

    uint groups_all[] = {(uint)divup(in.info.dims[0], threads_x),
        (uint)in.info.dims[1],
        (uint)in.info.dims[2],
        (uint)in.info.dims[3]};

    groups_all[dim] = divup(in.info.dims[dim], threads_y * REPEAT);

    Param tmpOut = out;
    Param tmpWt;
    tmpWt.info.offset = 0;
    for (int k = 0; k < 4; ++k) {
        tmpWt.info.dims[k] = 0;
        tmpWt.info.strides[k] = 0;
    }

    int tmp_elements = 1;
    if (groups_all[dim] > 1) {
        tmpOut.info.dims[dim] = groups_all[dim];

        for (int k = 0; k < 4; k++) tmp_elements *= tmpOut.info.dims[k];

        tmpOut.data = bufferAlloc(tmp_elements * sizeof(To));
        tmpWt.data = bufferAlloc(tmp_elements * sizeof(Tw));

        for (int k = dim + 1; k < 4; k++) tmpOut.info.strides[k] *= groups_all[dim];
    }

    mean_dim_launcher<Ti, Tw, To>(tmpOut, tmpWt, in, iwt, dim, threads_y, groups_all);

    if (groups_all[dim] > 1) {
        groups_all[dim] = 1;

        Param owt;
        mean_dim_launcher<Ti, Tw, To>(out, owt, tmpOut, tmpWt, dim, threads_y, groups_all);
        bufferFree(tmpOut.data);
        bufferFree(tmpWt.data);
    }

}

template<typename Ti, typename Tw, typename To>
void mean_first_launcher(Param out, Param owt,
        Param in, Param iwt,
        const int threads_x,
        const uint groups_x,
        const uint groups_y)
{

    bool input_weight = ((
            iwt.info.dims[0] *
            iwt.info.dims[1] *
            iwt.info.dims[2] *
            iwt.info.dims[3]) != 0);

    bool output_weight = ((
            owt.info.dims[0] *
            owt.info.dims[1] *
            owt.info.dims[2] *
            owt.info.dims[3]) != 0);

    std::string ref_name =
        std::string("mean_0_") +
        std::string(dtype_traits<Ti>::getName()) +
        std::string("_") +
        std::string(dtype_traits<Tw>::getName()) +
        std::string("_") +
        std::string(dtype_traits<To>::getName()) +
        std::string("_") +
        std::to_string(threads_x) +
        std::string("_") +
        std::to_string(input_weight) +
        std::string("_") +
        std::to_string(output_weight);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog==0 && entry.ker==0) {

        Binary<To, af_add_t> mean;
        ToNumStr<To> toNumStr;
        ToNumStr<Tw> twNumStr;
        Transform<uint, Tw, af_add_t> transform_weight;

        std::ostringstream options;
        options << " -D Ti=" << dtype_traits<Ti>::getName()
            << " -D Tw=" << dtype_traits<Tw>::getName()
            << " -D To=" << dtype_traits<To>::getName()
            << " -D DIMX=" << threads_x
            << " -D THREADS_PER_GROUP=" << THREADS_PER_GROUP
            << " -D init_To=" << toNumStr(mean.init())
            << " -D init_Tw=" << twNumStr(transform_weight(0))
            << " -D one_Tw=" << twNumStr(transform_weight(1));

        if (input_weight) { options << " -D INPUT_WEIGHT"; }
        if (output_weight) { options << " -D OUTPUT_WEIGHT"; }

        if (std::is_same<Ti, double>::value ||
                std::is_same<Ti, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        const char *ker_strs[] = {mops_cl, mean_first_cl};
        const int   ker_lens[] = {mops_cl_len, mean_first_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker = new Kernel(*entry.prog, "mean_first_kernel");

        addKernelToCache(device, ref_name, entry);
    }

    NDRange local(threads_x, THREADS_PER_GROUP / threads_x);
    NDRange global(groups_x * in.info.dims[2] * local[0],
            groups_y * in.info.dims[3] * local[1]);

    uint repeat = divup(in.info.dims[0], (local[0] * groups_x));

    if (input_weight && output_weight) {
        auto meanOp = KernelFunctor<
            Buffer, KParam,
            Buffer, KParam,
            Buffer, KParam,
            Buffer, KParam,
            uint, uint, uint>(*entry.ker);
        meanOp(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info,
                *owt.data, owt.info,
                *in.data, in.info,
                *iwt.data, iwt.info,
                groups_x, groups_y, repeat);
    } else if (!input_weight && !output_weight) {
        auto meanOp = KernelFunctor<
            Buffer, KParam,
            Buffer, KParam,
            uint, uint, uint>(*entry.ker);
        meanOp(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info,
                *in.data, in.info,
                groups_x, groups_y, repeat);
    } else if ( input_weight && !output_weight) {
        auto meanOp = KernelFunctor<
            Buffer, KParam,
            Buffer, KParam,
            Buffer, KParam,
            uint, uint, uint>(*entry.ker);
        meanOp(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info,
                *in.data, in.info,
                *iwt.data, iwt.info,
                groups_x, groups_y, repeat);
    } else if (!input_weight &&  output_weight) {
        auto meanOp = KernelFunctor<
            Buffer, KParam,
            Buffer, KParam,
            Buffer, KParam,
            uint, uint, uint>(*entry.ker);
        meanOp(EnqueueArgs(getQueue(), global, local),
                *out.data, out.info,
                *owt.data, owt.info,
                *in.data, in.info,
                groups_x, groups_y, repeat);
    }

    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tw, typename To>
void mean_first(Param out, Param in, Param iwt)
{
    uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
    threads_x = std::min(threads_x, THREADS_PER_GROUP);
    uint threads_y = THREADS_PER_GROUP / threads_x;

    uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
    uint groups_y = divup(in.info.dims[1], threads_y);

    Param tmpOut = out;
    Param tmpWt;
    tmpWt.info.offset = 0;
    for (int k = 0; k < 4; ++k) {
        tmpWt.info.dims[k] = 0;
        tmpWt.info.strides[k] = 0;
    }

    if (groups_x > 1) {

        tmpOut.data = bufferAlloc(groups_x *
                in.info.dims[1] *
                in.info.dims[2] *
                in.info.dims[3] *
                sizeof(To));

        tmpWt.data = bufferAlloc(groups_x *
                in.info.dims[1] *
                in.info.dims[2] *
                in.info.dims[3] *
                sizeof(Tw));


        tmpOut.info.dims[0] = groups_x;
        for (int k = 1; k < 4; k++) tmpOut.info.strides[k] *= groups_x;
        tmpWt.info = tmpOut.info;
    }

    mean_first_launcher<Ti, Tw, To>(tmpOut, tmpWt, in, iwt, threads_x, groups_x, groups_y);

    if (groups_x > 1) {
        Param owt;
        mean_first_launcher<Ti, Tw, To>(out, owt, tmpOut, tmpWt, threads_x, 1, groups_y);

        bufferFree(tmpOut.data);
        bufferFree(tmpWt.data);
    }
}

template<typename Ti, typename Tw, typename To>
void mean_weighted(Param out, Param in, Param iwt, int dim)
{
    if (dim == 0)
        return mean_first<Ti, Tw, To>(out, in, iwt);
    else
        return mean_dim  <Ti, Tw, To>(out, in, iwt, dim);
}

template<typename Ti, typename Tw, typename To>
void mean(Param out, Param in, int dim)
{
    Param dummy_weight;
    dummy_weight.info.offset = 0;
    for (int k = 0; k < 4; ++k) {
        dummy_weight.info.dims[k] = 0;
        dummy_weight.info.strides[k] = 0;
    }
    mean_weighted<Ti, Tw, To>(out, in, dummy_weight, dim);
}

#if defined(__GNUC__) || defined(__GNUG__)
/* GCC/G++, Clang/LLVM, Intel ICC */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#else
/* Other */
#endif

#if defined(__GNUC__) || defined(__GNUG__)
/* GCC/G++, Clang/LLVM, Intel ICC */
#pragma GCC diagnostic pop
#else
/* Other */
#endif

template<typename T, typename Tw>
T mean_all_weighted(Param in, Param iwt)
{
    int in_elements = in.info.dims[0] * in.info.dims[1] * in.info.dims[2] * in.info.dims[3];

    // FIXME: Use better heuristics to get to the optimum number
    if (in_elements > 4096) {

        bool in_is_linear = (in.info.strides[0] == 1);
        bool wt_is_linear = (in.info.strides[0] == 1);
        for (int k = 1; k < 4; k++) {
            in_is_linear &= ( in.info.strides[k] == ( in.info.strides[k - 1] *  in.info.dims[k - 1]));
            wt_is_linear &= (iwt.info.strides[k] == (iwt.info.strides[k - 1] * iwt.info.dims[k - 1]));
        }

        if (in_is_linear && wt_is_linear) {
            in.info.dims[0] = in_elements;
            for (int k = 1; k < 4; k++) {
                in.info.dims[k] = 1;
                in.info.strides[k] = in_elements;
            }
            iwt.info = in.info;
        }

        uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
        threads_x = std::min(threads_x, THREADS_PER_GROUP);
        uint threads_y = THREADS_PER_GROUP / threads_x;

        Param tmpOut;
        uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
        uint groups_y = divup(in.info.dims[1], threads_y);

        tmpOut.info.offset = 0;
        tmpOut.info.dims[0] = groups_x;
        tmpOut.info.strides[0] = 1;

        for (int k = 1; k < 4; k++) {
            tmpOut.info.dims[k] = in.info.dims[k];
            tmpOut.info.strides[k] = tmpOut.info.dims[k - 1] * tmpOut.info.strides[k - 1];
        }

        Param tmpWt;
        tmpWt.info = tmpOut.info;

        int tmp_elements = tmpOut.info.strides[3] * tmpOut.info.dims[3];
        tmpOut.data = bufferAlloc(tmp_elements * sizeof(T));
        tmpWt.data = bufferAlloc(tmp_elements * sizeof(Tw));

        mean_first_launcher<T, Tw, T>(tmpOut, tmpWt, in, iwt, threads_x, groups_x, groups_y);

        unique_ptr<T> h_ptr(new T[tmp_elements]);
        unique_ptr<Tw> h_wptr(new Tw[tmp_elements]);

        getQueue().enqueueReadBuffer(*tmpOut.data, CL_TRUE, 0, sizeof(T) * tmp_elements, h_ptr.get());
        getQueue().enqueueReadBuffer( *tmpWt.data, CL_TRUE, 0, sizeof(Tw) * tmp_elements, h_wptr.get());

        T* h_ptr_raw = h_ptr.get();
        Tw* h_wptr_raw = h_wptr.get();

        MeanOp<T, Tw> Op(h_ptr_raw[0], h_wptr_raw[0]);
        for (int i = 1; i < (int)tmp_elements; i++) {
            Op(h_ptr_raw[i], h_wptr_raw[i]);
        }

        bufferFree(tmpOut.data);
        bufferFree(tmpWt.data);

        return Op.runningMean;

    } else {

        unique_ptr<T> h_ptr(new T[in_elements]);
        unique_ptr<Tw> h_wptr(new Tw[in_elements]);
        T* h_ptr_raw = h_ptr.get();
        Tw* h_wptr_raw = h_wptr.get();

        getQueue().enqueueReadBuffer(*in.data, CL_TRUE, sizeof(T) * in.info.offset,
                sizeof(T) * in_elements, h_ptr_raw);
        getQueue().enqueueReadBuffer(*iwt.data, CL_TRUE, sizeof(Tw) * iwt.info.offset,
                sizeof(Tw) * in_elements, h_wptr_raw);

        MeanOp<T, Tw> Op(h_ptr_raw[0], h_wptr_raw[0]);
        for (int i = 1; i < (int)in_elements; i++) {
            Op(h_ptr_raw[i], h_wptr_raw[i]);
        }

        return Op.runningMean;
    }
}

template<typename Ti, typename Tw, typename To>
To mean_all(Param in)
{
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

        Param tmpOut;
        uint groups_x = divup(in.info.dims[0], threads_x * REPEAT);
        uint groups_y = divup(in.info.dims[1], threads_y);

        tmpOut.info.offset = 0;
        tmpOut.info.dims[0] = groups_x;
        tmpOut.info.strides[0] = 1;

        for (int k = 1; k < 4; k++) {
            tmpOut.info.dims[k] = in.info.dims[k];
            tmpOut.info.strides[k] = tmpOut.info.dims[k - 1] * tmpOut.info.strides[k - 1];
        }

        Param iWt; //dummy input weights
        iWt.info.offset = 0;
        for (int k = 0; k < 4; ++k) {
            iWt.info.dims[k] = 0;
            iWt.info.strides[k] = 0;
        }
        Param tmpCt;
        tmpCt.info = tmpOut.info;

        int tmp_elements = tmpOut.info.strides[3] * tmpOut.info.dims[3];
        tmpOut.data = bufferAlloc(tmp_elements * sizeof(To));
        tmpCt.data = bufferAlloc(tmp_elements * sizeof(Tw));

        mean_first_launcher<Ti, Tw, To>(tmpOut, tmpCt, in, iWt, threads_x, groups_x, groups_y);

        unique_ptr<To> h_ptr(new To[tmp_elements]);
        unique_ptr<Tw> h_cptr(new Tw[tmp_elements]);

        getQueue().enqueueReadBuffer(*tmpOut.data, CL_TRUE, 0, sizeof(To) * tmp_elements, h_ptr.get());
        getQueue().enqueueReadBuffer( *tmpCt.data, CL_TRUE, 0, sizeof(Tw) * tmp_elements, h_cptr.get());

        To* h_ptr_raw = h_ptr.get();
        Tw* h_cptr_raw = h_cptr.get();

        MeanOp<To, Tw> Op(h_ptr_raw[0], h_cptr_raw[0]);
        for (int i = 1; i < (int)tmp_elements; i++) {
            Op(h_ptr_raw[i], h_cptr_raw[i]);
        }

        bufferFree(tmpOut.data);
        bufferFree(tmpCt.data);

        return Op.runningMean;

    } else {

        unique_ptr<Ti> h_ptr(new Ti[in_elements]);
        Ti* h_ptr_raw = h_ptr.get();

        getQueue().enqueueReadBuffer(*in.data, CL_TRUE, sizeof(Ti) * in.info.offset,
                sizeof(Ti) * in_elements, h_ptr_raw);


        //TODO : MeanOp with (Tw)1
        Transform<Ti, To, af_add_t> transform;
        Transform<uint, Tw, af_add_t> transform_weight;
        MeanOp<To, Tw> Op(transform(h_ptr_raw[0]), transform_weight(1));
        for (int i = 1; i < (int)in_elements; i++) {
            Op(transform(h_ptr_raw[i]), transform_weight(1));
        }

        return Op.runningMean;
    }
}
}

}

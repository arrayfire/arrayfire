/*******************************************************
 * Copyright (c) 2018, ArrayFire
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
#include <kernel_headers/ops.hpp>
#include <kernel_headers/reduce_blocks_by_key_dim.hpp>
#include <kernel_headers/reduce_blocks_by_key_first.hpp>
#include <kernel_headers/reduce_by_key_boundary.hpp>
#include <kernel_headers/reduce_by_key_boundary_dim.hpp>
#include <kernel_headers/reduce_by_key_compact.hpp>
#include <kernel_headers/reduce_by_key_compact_dim.hpp>
#include <kernel_headers/reduce_by_key_needs_reduction.hpp>
#include <memory.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <type_util.hpp>
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include "config.hpp"
#include "names.hpp"

#include <boost/compute/algorithm/inclusive_scan.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/functional/operator.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

namespace compute = boost::compute;

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;
using std::unique_ptr;
using std::vector;

namespace opencl {

namespace kernel {

template<typename Ti, typename Tk, typename To, af_op_t op>
void launch_reduce_blocks_dim_by_key(cl::Buffer *reduced_block_sizes,
                                     Param keys_out, Param vals_out,
                                     const Param keys, const Param vals,
                                     int change_nan, double nanval, const int n,
                                     const uint threads_x, const int dim,
                                     vector<int> dim_ordering) {
    std::string ref_name =
        std::string("reduce_blocks_dim_by_key_") +
        std::string(dtype_traits<Ti>::getName()) + std::string("_") +
        std::string(dtype_traits<Tk>::getName()) + std::string("_") +
        std::string(dtype_traits<To>::getName()) + std::string("_") +
        std::to_string(op) + std::string("_") + std::to_string(threads_x);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        Binary<To, op> reduce;
        ToNumStr<To> toNumStr;

        std::ostringstream options;
        options << " -D To=" << dtype_traits<To>::getName()
                << " -D Tk=" << dtype_traits<Tk>::getName()
                << " -D Ti=" << dtype_traits<Ti>::getName() << " -D T=To"
                << " -D DIMX=" << threads_x << " -D DIM=" << dim
                << " -D init=" << toNumStr(reduce.init()) << " -D "
                << binOpName<op>() << " -D CPLX=" << af::iscplx<Ti>();

        options << getTypeBuildDefinition<Ti>();

        const char *ker_strs[] = {ops_cl, reduce_blocks_by_key_dim_cl};
        const int ker_lens[]   = {ops_cl_len, reduce_blocks_by_key_dim_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());

        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "reduce_blocks_by_key_dim");

        addKernelToCache(device, ref_name, entry);
    }

    int numBlocks = divup(n, threads_x);

    NDRange local(threads_x);
    NDRange global(threads_x * numBlocks, vals_out.info.dims[dim_ordering[1]],
                   vals_out.info.dims[dim_ordering[2]] *
                       vals_out.info.dims[dim_ordering[3]]);

    auto reduceOp =
        KernelFunctor<Buffer, Buffer, KParam, Buffer, KParam, Buffer, KParam,
                      Buffer, KParam, int, To, int, int>(*entry.ker);

    reduceOp(EnqueueArgs(getQueue(), global, local), *reduced_block_sizes,
             *keys_out.data, keys_out.info, *vals_out.data, vals_out.info,
             *keys.data, keys.info, *vals.data, vals.info, change_nan,
             scalar<To>(nanval), n, vals_out.info.dims[dim_ordering[2]]);

    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op>
void launch_reduce_blocks_by_key(cl::Buffer *reduced_block_sizes,
                                 Param keys_out, Param vals_out,
                                 const Param keys, const Param vals,
                                 int change_nan, double nanval, const int n,
                                 const uint threads_x) {
    std::string ref_name =
        std::string("reduce_blocks_by_key_0_") +
        std::string(dtype_traits<Ti>::getName()) + std::string("_") +
        std::string(dtype_traits<Tk>::getName()) + std::string("_") +
        std::string(dtype_traits<To>::getName()) + std::string("_") +
        std::to_string(op) + std::string("_") + std::to_string(threads_x);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        Binary<To, op> reduce;
        ToNumStr<To> toNumStr;

        std::ostringstream options;
        options << " -D To=" << dtype_traits<To>::getName()
                << " -D Tk=" << dtype_traits<Tk>::getName()
                << " -D Ti=" << dtype_traits<Ti>::getName() << " -D T=To"
                << " -D DIMX=" << threads_x
                << " -D init=" << toNumStr(reduce.init()) << " -D "
                << binOpName<op>() << " -D CPLX=" << af::iscplx<Ti>();

        options << getTypeBuildDefinition<Ti>();

        const char *ker_strs[] = {ops_cl, reduce_blocks_by_key_first_cl};
        const int ker_lens[] = {ops_cl_len, reduce_blocks_by_key_first_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());

        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "reduce_blocks_by_key_first");

        addKernelToCache(device, ref_name, entry);
    }

    int numBlocks = divup(n, threads_x);

    NDRange local(threads_x);
    NDRange global(threads_x * numBlocks, vals_out.info.dims[1],
                   vals_out.info.dims[2] * vals_out.info.dims[3]);

    auto reduceOp =
        KernelFunctor<Buffer, Buffer, KParam, Buffer, KParam, Buffer, KParam,
                      Buffer, KParam, int, To, int, int>(*entry.ker);

    reduceOp(EnqueueArgs(getQueue(), global, local), *reduced_block_sizes,
             *keys_out.data, keys_out.info, *vals_out.data, vals_out.info,
             *keys.data, keys.info, *vals.data, vals.info, change_nan,
             scalar<To>(nanval), n, vals_out.info.dims[2]);

    CL_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename To, af_op_t op>
void launch_final_boundary_reduce(cl::Buffer *reduced_block_sizes,
                                  Param keys_out, Param vals_out, const int n,
                                  const int numBlocks, const int threads_x) {
    std::string ref_name =
        std::string("final_boundary_reduce") +
        std::string(dtype_traits<Tk>::getName()) + std::string("_") +
        std::string(dtype_traits<To>::getName()) + std::string("_") +
        std::to_string(op) + std::string("_") + std::to_string(threads_x);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        Binary<To, op> reduce;
        ToNumStr<To> toNumStr;

        std::ostringstream options;
        options << " -D To=" << dtype_traits<To>::getName()
                << " -D Ti=" << dtype_traits<To>::getName()
                << " -D Tk=" << dtype_traits<Tk>::getName() << " -D T=To"
                << " -D DIMX=" << threads_x
                << " -D init=" << toNumStr(reduce.init()) << " -D "
                << binOpName<op>() << " -D CPLX=" << af::iscplx<To>();

        options << getTypeBuildDefinition<To>();

        const char *ker_strs[] = {ops_cl, reduce_by_key_boundary_cl};
        const int ker_lens[]   = {ops_cl_len, reduce_by_key_boundary_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());

        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "final_boundary_reduce");

        addKernelToCache(device, ref_name, entry);
    }

    NDRange local(threads_x);
    NDRange global(threads_x * numBlocks);

    auto reduceOp =
        KernelFunctor<Buffer, Buffer, KParam, Buffer, KParam, int>(*entry.ker);

    reduceOp(EnqueueArgs(getQueue(), global, local), *reduced_block_sizes,
             *keys_out.data, keys_out.info, *vals_out.data, vals_out.info, n);

    CL_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename To, af_op_t op>
void launch_final_boundary_reduce_dim(cl::Buffer *reduced_block_sizes,
                                      Param keys_out, Param vals_out,
                                      const int n, const int numBlocks,
                                      const int threads_x, const int dim,
                                      vector<int> dim_ordering) {
    std::string ref_name =
        std::string("final_boundary_reduce") +
        std::string(dtype_traits<Tk>::getName()) + std::string("_") +
        std::string(dtype_traits<To>::getName()) + std::string("_") +
        std::to_string(op) + std::string("_") + std::to_string(threads_x);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        Binary<To, op> reduce;
        ToNumStr<To> toNumStr;

        std::ostringstream options;
        options << " -D To=" << dtype_traits<To>::getName()
                << " -D Ti=" << dtype_traits<To>::getName()
                << " -D Tk=" << dtype_traits<Tk>::getName() << " -D T=To"
                << " -D DIMX=" << threads_x << " -D DIM=" << dim
                << " -D init=" << toNumStr(reduce.init()) << " -D "
                << binOpName<op>() << " -D CPLX=" << af::iscplx<To>();

        options << getTypeBuildDefinition<To>();

        const char *ker_strs[] = {ops_cl, reduce_by_key_boundary_dim_cl};
        const int ker_lens[] = {ops_cl_len, reduce_by_key_boundary_dim_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());

        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "final_boundary_reduce_dim");

        addKernelToCache(device, ref_name, entry);
    }

    NDRange local(threads_x);
    NDRange global(threads_x * numBlocks, vals_out.info.dims[dim_ordering[1]],
                   vals_out.info.dims[dim_ordering[2]] *
                       vals_out.info.dims[dim_ordering[3]]);

    auto reduceOp =
        KernelFunctor<Buffer, Buffer, KParam, Buffer, KParam, int, int>(
            *entry.ker);

    reduceOp(EnqueueArgs(getQueue(), global, local), *reduced_block_sizes,
             *keys_out.data, keys_out.info, *vals_out.data, vals_out.info, n,
             vals_out.info.dims[dim_ordering[2]]);

    CL_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename To>
void launch_compact(cl::Buffer *reduced_block_sizes, Param keys_out,
                    Param vals_out, const Param keys, const Param vals,
                    const int numBlocks, const int threads_x) {
    std::string ref_name =
        std::string("compact_") + std::string(dtype_traits<Tk>::getName()) +
        std::string("_") + std::string(dtype_traits<To>::getName()) +
        std::string("_") + std::to_string(threads_x);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D To=" << dtype_traits<To>::getName()
                << " -D Tk=" << dtype_traits<Tk>::getName() << " -D T=To"
                << " -D DIMX=" << threads_x << " -D CPLX=" << af::iscplx<To>();

        options << getTypeBuildDefinition<To>();

        const char *ker_strs[] = {ops_cl, reduce_by_key_compact_cl};
        const int ker_lens[]   = {ops_cl_len, reduce_by_key_compact_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());

        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "compact");

        addKernelToCache(device, ref_name, entry);
    }

    NDRange local(threads_x);
    NDRange global(threads_x * numBlocks, vals_out.info.dims[1],
                   vals_out.info.dims[2] * vals_out.info.dims[3]);

    auto reduceOp =
        KernelFunctor<Buffer, Buffer, KParam, Buffer, KParam, Buffer, KParam,
                      Buffer, KParam, int>(*entry.ker);

    reduceOp(EnqueueArgs(getQueue(), global, local), *reduced_block_sizes,
             *keys_out.data, keys_out.info, *vals_out.data, vals_out.info,
             *keys.data, keys.info, *vals.data, vals.info,
             vals_out.info.dims[2]);

    CL_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename To>
void launch_compact_dim(cl::Buffer *reduced_block_sizes, Param keys_out,
                        Param vals_out, const Param keys, const Param vals,
                        const int numBlocks, const int threads_x, const int dim,
                        vector<int> dim_ordering) {
    std::string ref_name =
        std::string("compact_dim_") + std::string(dtype_traits<Tk>::getName()) +
        std::string("_") + std::string(dtype_traits<To>::getName()) +
        std::string("_") + std::to_string(threads_x);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D To=" << dtype_traits<To>::getName()
                << " -D Tk=" << dtype_traits<Tk>::getName() << " -D T=To"
                << " -D DIMX=" << threads_x << " -D DIM=" << dim
                << " -D CPLX=" << af::iscplx<To>();

        options << getTypeBuildDefinition<To>();

        const char *ker_strs[] = {ops_cl, reduce_by_key_compact_dim_cl};
        const int ker_lens[]   = {ops_cl_len, reduce_by_key_compact_dim_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());

        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "compact_dim");

        addKernelToCache(device, ref_name, entry);
    }

    NDRange local(threads_x);
    NDRange global(threads_x * numBlocks, vals_out.info.dims[dim_ordering[1]],
                   vals_out.info.dims[dim_ordering[2]] *
                       vals_out.info.dims[dim_ordering[3]]);

    auto reduceOp =
        KernelFunctor<Buffer, Buffer, KParam, Buffer, KParam, Buffer, KParam,
                      Buffer, KParam, int>(*entry.ker);

    reduceOp(EnqueueArgs(getQueue(), global, local), *reduced_block_sizes,
             *keys_out.data, keys_out.info, *vals_out.data, vals_out.info,
             *keys.data, keys.info, *vals.data, vals.info,
             vals_out.info.dims[dim_ordering[2]]);

    CL_DEBUG_FINISH(getQueue());
}

template<typename Tk>
void launch_test_needs_reduction(cl::Buffer needs_reduction,
                                 cl::Buffer needs_boundary, const Param keys,
                                 const int n, const int numBlocks,
                                 const int threads_x) {
    std::string ref_name = std::string("test_needs_reduction_") +
                           std::string(dtype_traits<Tk>::getName()) +
                           std::string("_") + std::to_string(threads_x);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D Tk=" << dtype_traits<Tk>::getName()
                << " -D DIMX=" << threads_x;

        const char *ker_strs[] = {ops_cl, reduce_by_key_needs_reduction_cl};
        const int ker_lens[]   = {ops_cl_len,
                                reduce_by_key_needs_reduction_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());

        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "test_needs_reduction");

        addKernelToCache(device, ref_name, entry);
    }

    NDRange local(threads_x);
    NDRange global(threads_x * numBlocks);

    auto reduceOp =
        KernelFunctor<Buffer, Buffer, Buffer, KParam, int>(*entry.ker);

    reduceOp(EnqueueArgs(getQueue(), global, local), needs_reduction,
             needs_boundary, *keys.data, keys.info, n);

    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op>
int reduce_by_key_first(Array<Tk> &keys_out, Array<To> &vals_out,
                        const Param keys, const Param vals, bool change_nan,
                        double nanval) {
    dim4 kdims(4, keys.info.dims);
    dim4 odims(4, vals.info.dims);

    auto reduced_keys   = createEmptyArray<Tk>(kdims);
    auto reduced_vals   = createEmptyArray<To>(odims);
    auto t_reduced_keys = createEmptyArray<Tk>(kdims);
    auto t_reduced_vals = createEmptyArray<To>(odims);

    // flags determining more reduction is necessary
    auto needs_another_reduction        = memAlloc<int>(1);
    auto needs_block_boundary_reduction = memAlloc<int>(1);

    int nelems = kdims[0];

    const unsigned int numThreads = 128;
    int numBlocksD0               = divup(nelems, numThreads);

    auto reduced_block_sizes = memAlloc<int>(numBlocksD0);

    compute::command_queue c_queue(getQueue()());
    compute::buffer val_buf((*reduced_block_sizes.get())());

    int n_reduced_host = nelems;
    int needs_another_reduction_host;

    int needs_block_boundary_reduction_host;
    bool first_pass = true;
    do {
        numBlocksD0 = divup(n_reduced_host, numThreads);

        if (first_pass) {
            launch_reduce_blocks_by_key<Ti, Tk, To, op>(
                reduced_block_sizes.get(), reduced_keys, reduced_vals, keys,
                vals, change_nan, nanval, n_reduced_host, numThreads);
            first_pass = false;
        } else {
            launch_reduce_blocks_by_key<To, Tk, To, op>(
                reduced_block_sizes.get(), reduced_keys, reduced_vals,
                t_reduced_keys, t_reduced_vals, change_nan, nanval,
                n_reduced_host, numThreads);
        }

        compute::inclusive_scan(
            compute::make_buffer_iterator<int>(val_buf),
            compute::make_buffer_iterator<int>(val_buf, numBlocksD0),
            compute::make_buffer_iterator<int>(val_buf), c_queue);

        launch_compact<Tk, To>(reduced_block_sizes.get(), t_reduced_keys,
                               t_reduced_vals, reduced_keys, reduced_vals,
                               numBlocksD0, numThreads);

        getQueue().enqueueReadBuffer(*reduced_block_sizes.get(), true,
                                     (numBlocksD0 - 1) * sizeof(int),
                                     sizeof(int), &n_reduced_host);

        // reset flags
        needs_block_boundary_reduction_host = 0;
        needs_another_reduction_host        = 0;

        getQueue().enqueueWriteBuffer(*needs_another_reduction.get(), CL_FALSE,
                                      0, sizeof(int),
                                      &needs_another_reduction_host);
        getQueue().enqueueWriteBuffer(*needs_block_boundary_reduction.get(),
                                      CL_FALSE, 0, sizeof(int),
                                      &needs_block_boundary_reduction_host);
        numBlocksD0 = divup(n_reduced_host, numThreads);

        launch_test_needs_reduction<Tk>(*needs_another_reduction.get(),
                                        *needs_block_boundary_reduction.get(),
                                        t_reduced_keys, n_reduced_host,
                                        numBlocksD0, numThreads);

        getQueue().enqueueReadBuffer(*needs_another_reduction.get(), CL_FALSE,
                                     0, sizeof(int),
                                     &needs_another_reduction_host);
        getQueue().enqueueReadBuffer(*needs_block_boundary_reduction.get(),
                                     CL_TRUE, 0, sizeof(int),
                                     &needs_block_boundary_reduction_host);

        if (needs_block_boundary_reduction_host &&
            !needs_another_reduction_host) {
            launch_final_boundary_reduce<Tk, To, op>(
                reduced_block_sizes.get(), t_reduced_keys, t_reduced_vals,
                n_reduced_host, numBlocksD0, numThreads);

            compute::inclusive_scan(
                compute::make_buffer_iterator<int>(val_buf),
                compute::make_buffer_iterator<int>(val_buf, numBlocksD0),
                compute::make_buffer_iterator<int>(val_buf), c_queue);

            getQueue().enqueueReadBuffer(*reduced_block_sizes.get(), true,
                                         (numBlocksD0 - 1) * sizeof(int),
                                         sizeof(int), &n_reduced_host);

            launch_compact<Tk, To>(reduced_block_sizes.get(), reduced_keys,
                                   reduced_vals, t_reduced_keys, t_reduced_vals,
                                   numBlocksD0, numThreads);

            std::swap(t_reduced_keys, reduced_keys);
            std::swap(t_reduced_vals, reduced_vals);
        }
    } while (needs_another_reduction_host ||
             needs_block_boundary_reduction_host);

    keys_out = t_reduced_keys;
    vals_out = t_reduced_vals;

    return n_reduced_host;
}

template<typename Ti, typename Tk, typename To, af_op_t op>
int reduce_by_key_dim(Array<Tk> &keys_out, Array<To> &vals_out,
                      const Param keys, const Param vals, bool change_nan,
                      double nanval, const int dim) {
    vector<int> dim_ordering = {dim};
    for (int i = 0; i < 4; ++i) {
        if (i != dim) { dim_ordering.push_back(i); }
    }

    dim4 kdims(4, keys.info.dims);
    dim4 odims(4, vals.info.dims);

    auto reduced_keys   = createEmptyArray<Tk>(kdims);
    auto reduced_vals   = createEmptyArray<To>(odims);
    auto t_reduced_keys = createEmptyArray<Tk>(kdims);
    auto t_reduced_vals = createEmptyArray<To>(odims);

    // flags determining more reduction is necessary
    auto needs_another_reduction        = memAlloc<int>(1);
    auto needs_block_boundary_reduction = memAlloc<int>(1);

    int nelems = kdims[0];

    const unsigned int numThreads = 128;
    int numBlocksD0               = divup(nelems, numThreads);

    auto reduced_block_sizes = memAlloc<int>(numBlocksD0);

    compute::command_queue c_queue(getQueue()());
    compute::buffer val_buf((*reduced_block_sizes.get())());

    int n_reduced_host = nelems;
    int needs_another_reduction_host;
    int needs_block_boundary_reduction_host;

    bool first_pass = true;
    do {
        numBlocksD0 = divup(n_reduced_host, numThreads);

        if (first_pass) {
            launch_reduce_blocks_dim_by_key<Ti, Tk, To, op>(
                reduced_block_sizes.get(), reduced_keys, reduced_vals, keys,
                vals, change_nan, nanval, n_reduced_host, numThreads, dim,
                dim_ordering);
            first_pass = false;
        } else {
            launch_reduce_blocks_dim_by_key<To, Tk, To, op>(
                reduced_block_sizes.get(), reduced_keys, reduced_vals,
                t_reduced_keys, t_reduced_vals, change_nan, nanval,
                n_reduced_host, numThreads, dim, dim_ordering);
        }

        compute::inclusive_scan(
            compute::make_buffer_iterator<int>(val_buf),
            compute::make_buffer_iterator<int>(val_buf, numBlocksD0),
            compute::make_buffer_iterator<int>(val_buf), c_queue);

        launch_compact_dim<Tk, To>(reduced_block_sizes.get(), t_reduced_keys,
                                   t_reduced_vals, reduced_keys, reduced_vals,
                                   numBlocksD0, numThreads, dim, dim_ordering);

        getQueue().enqueueReadBuffer(*reduced_block_sizes.get(), true,
                                     (numBlocksD0 - 1) * sizeof(int),
                                     sizeof(int), &n_reduced_host);

        // reset flags
        needs_block_boundary_reduction_host = 0;
        needs_another_reduction_host        = 0;

        getQueue().enqueueWriteBuffer(*needs_another_reduction.get(), CL_FALSE,
                                      0, sizeof(int),
                                      &needs_another_reduction_host);
        getQueue().enqueueWriteBuffer(*needs_block_boundary_reduction.get(),
                                      CL_FALSE, 0, sizeof(int),
                                      &needs_block_boundary_reduction_host);

        numBlocksD0 = divup(n_reduced_host, numThreads);

        launch_test_needs_reduction<Tk>(*needs_another_reduction.get(),
                                        *needs_block_boundary_reduction.get(),
                                        t_reduced_keys, n_reduced_host,
                                        numBlocksD0, numThreads);

        getQueue().enqueueReadBuffer(*needs_another_reduction.get(), CL_FALSE,
                                     0, sizeof(int),
                                     &needs_another_reduction_host);
        getQueue().enqueueReadBuffer(*needs_block_boundary_reduction.get(),
                                     CL_TRUE, 0, sizeof(int),
                                     &needs_block_boundary_reduction_host);

        if (needs_block_boundary_reduction_host &&
            !needs_another_reduction_host) {
            launch_final_boundary_reduce_dim<Tk, To, op>(
                reduced_block_sizes.get(), t_reduced_keys, t_reduced_vals,
                n_reduced_host, numBlocksD0, numThreads, dim, dim_ordering);

            compute::inclusive_scan(
                compute::make_buffer_iterator<int>(val_buf),
                compute::make_buffer_iterator<int>(val_buf, numBlocksD0),
                compute::make_buffer_iterator<int>(val_buf), c_queue);

            getQueue().enqueueReadBuffer(*reduced_block_sizes.get(), true,
                                         (numBlocksD0 - 1) * sizeof(int),
                                         sizeof(int), &n_reduced_host);

            launch_compact_dim<Tk, To>(reduced_block_sizes.get(), reduced_keys,
                                       reduced_vals, t_reduced_keys,
                                       t_reduced_vals, numBlocksD0, numThreads,
                                       dim, dim_ordering);

            std::swap(t_reduced_keys, reduced_keys);
            std::swap(t_reduced_vals, reduced_vals);
        }
    } while (needs_another_reduction_host ||
             needs_block_boundary_reduction_host);

    keys_out = t_reduced_keys;
    vals_out = t_reduced_vals;

    return n_reduced_host;
}

template<af_op_t op, typename Ti, typename Tk, typename To>
void reduce_by_key(Array<Tk> &keys_out, Array<To> &vals_out,
                   const Array<Tk> &keys, const Array<Ti> &vals, int dim,
                   bool change_nan, double nanval) {
    dim4 kdims = keys.dims();
    dim4 odims = vals.dims();

    // allocate space for output arrays
    Array<Tk> reduced_keys = createEmptyArray<Tk>(dim4());
    Array<To> reduced_vals = createEmptyArray<To>(dim4());

    int n_reduced = 0;
    if (dim == 0) {
        n_reduced = reduce_by_key_first<Ti, Tk, To, op>(
            reduced_keys, reduced_vals, keys, vals, change_nan, nanval);
    } else {
        n_reduced = reduce_by_key_dim<Ti, Tk, To, op>(
            reduced_keys, reduced_vals, keys, vals, change_nan, nanval, dim);
    }

    kdims[0]   = n_reduced;
    odims[dim] = n_reduced;
    std::vector<af_seq> kindex, vindex;
    for (int i = 0; i < odims.ndims(); ++i) {
        af_seq sk = {0.0, (double)kdims[i] - 1, 1.0};
        af_seq sv = {0.0, (double)odims[i] - 1, 1.0};
        kindex.push_back(sk);
        vindex.push_back(sv);
    }

    keys_out = createSubArray<Tk>(reduced_keys, kindex, true);
    vals_out = createSubArray<To>(reduced_vals, vindex, true);
}
}  // namespace kernel
}  // namespace opencl

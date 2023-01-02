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
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel/config.hpp>
#include <kernel/names.hpp>
#include <kernel_headers/ops.hpp>
#include <kernel_headers/reduce_blocks_by_key_dim.hpp>
#include <kernel_headers/reduce_blocks_by_key_first.hpp>
#include <kernel_headers/reduce_by_key_boundary.hpp>
#include <kernel_headers/reduce_by_key_boundary_dim.hpp>
#include <kernel_headers/reduce_by_key_compact.hpp>
#include <kernel_headers/reduce_by_key_compact_dim.hpp>
#include <kernel_headers/reduce_by_key_needs_reduction.hpp>
#include <memory.hpp>
#include <traits.hpp>

#include <boost/compute/algorithm/inclusive_scan.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/functional/operator.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

#include <string>
#include <vector>

namespace compute = boost::compute;

namespace arrayfire {
namespace opencl {
namespace kernel {

template<typename Ti, typename Tk, typename To, af_op_t op>
void reduceBlocksByKeyDim(cl::Buffer *reduced_block_sizes, Param keys_out,
                          Param vals_out, const Param keys, const Param vals,
                          int change_nan, double nanval, const int n,
                          const uint threads_x, const int dim,
                          std::vector<int> dim_ordering) {
    ToNumStr<To> toNumStr;
    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<Ti>(), TemplateTypename<To>(), TemplateTypename<Tk>(),
        TemplateArg(op),        TemplateArg(threads_x),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(Tk, dtype_traits<Tk>::getName()),
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(DIMX, threads_x),
        DefineKeyValue(DIM, dim),
        DefineKeyValue(init, toNumStr(common::Binary<To, op>::init())),
        DefineKeyFromStr(binOpName<op>()),
        DefineKeyValue(CPLX, iscplx<Ti>()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<Ti>());

    auto reduceBlocksByKeyDim =
        common::getKernel("reduce_blocks_by_key_dim",
                          {{ops_cl_src, reduce_blocks_by_key_dim_cl_src}},
                          tmpltArgs, compileOpts);
    int numBlocks = divup(n, threads_x);

    cl::NDRange local(threads_x);
    cl::NDRange global(threads_x * numBlocks,
                       vals_out.info.dims[dim_ordering[1]],
                       vals_out.info.dims[dim_ordering[2]] *
                           vals_out.info.dims[dim_ordering[3]]);

    reduceBlocksByKeyDim(cl::EnqueueArgs(getQueue(), global, local),
                         *reduced_block_sizes, *keys_out.data, keys_out.info,
                         *vals_out.data, vals_out.info, *keys.data, keys.info,
                         *vals.data, vals.info, change_nan, scalar<To>(nanval),
                         n,
                         static_cast<int>(vals_out.info.dims[dim_ordering[2]]));
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op>
void reduceBlocksByKey(cl::Buffer *reduced_block_sizes, Param keys_out,
                       Param vals_out, const Param keys, const Param vals,
                       int change_nan, double nanval, const int n,
                       const uint threads_x) {
    ToNumStr<To> toNumStr;
    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<Ti>(), TemplateTypename<To>(), TemplateTypename<Tk>(),
        TemplateArg(op),        TemplateArg(threads_x),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(Tk, dtype_traits<Tk>::getName()),
        DefineKeyValue(Ti, dtype_traits<Ti>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(DIMX, threads_x),
        DefineKeyValue(init, toNumStr(common::Binary<To, op>::init())),
        DefineKeyFromStr(binOpName<op>()),
        DefineKeyValue(CPLX, iscplx<Ti>()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<Ti>());

    auto reduceBlocksByKeyFirst =
        common::getKernel("reduce_blocks_by_key_first",
                          {{ops_cl_src, reduce_blocks_by_key_first_cl_src}},
                          tmpltArgs, compileOpts);
    int numBlocks = divup(n, threads_x);

    cl::NDRange local(threads_x);
    cl::NDRange global(threads_x * numBlocks, vals_out.info.dims[1],
                       vals_out.info.dims[2] * vals_out.info.dims[3]);

    reduceBlocksByKeyFirst(
        cl::EnqueueArgs(getQueue(), global, local), *reduced_block_sizes,
        *keys_out.data, keys_out.info, *vals_out.data, vals_out.info,
        *keys.data, keys.info, *vals.data, vals.info, change_nan,
        scalar<To>(nanval), n, static_cast<int>(vals_out.info.dims[2]));
    CL_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename To, af_op_t op>
void finalBoundaryReduce(cl::Buffer *reduced_block_sizes, Param keys_out,
                         Param vals_out, const int n, const int numBlocks,
                         const int threads_x) {
    ToNumStr<To> toNumStr;
    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<To>(),
        TemplateTypename<Tk>(),
        TemplateArg(op),
        TemplateArg(threads_x),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(Tk, dtype_traits<Tk>::getName()),
        DefineKeyValue(Ti, dtype_traits<To>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(DIMX, threads_x),
        DefineKeyValue(init, toNumStr(common::Binary<To, op>::init())),
        DefineKeyFromStr(binOpName<op>()),
        DefineKeyValue(CPLX, iscplx<To>()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<To>());

    auto finalBoundaryReduce = common::getKernel(
        "final_boundary_reduce", {{ops_cl_src, reduce_by_key_boundary_cl_src}},
        tmpltArgs, compileOpts);

    cl::NDRange local(threads_x);
    cl::NDRange global(threads_x * numBlocks);

    finalBoundaryReduce(cl::EnqueueArgs(getQueue(), global, local),
                        *reduced_block_sizes, *keys_out.data, keys_out.info,
                        *vals_out.data, vals_out.info, n);
    CL_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename To, af_op_t op>
void finalBoundaryReduceDim(cl::Buffer *reduced_block_sizes, Param keys_out,
                            Param vals_out, const int n, const int numBlocks,
                            const int threads_x, const int dim,
                            std::vector<int> dim_ordering) {
    ToNumStr<To> toNumStr;
    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<To>(),
        TemplateTypename<Tk>(),
        TemplateArg(op),
        TemplateArg(threads_x),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(Tk, dtype_traits<Tk>::getName()),
        DefineKeyValue(Ti, dtype_traits<To>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(DIMX, threads_x),
        DefineKeyValue(DIM, dim),
        DefineKeyValue(init, toNumStr(common::Binary<To, op>::init())),
        DefineKeyFromStr(binOpName<op>()),
        DefineKeyValue(CPLX, iscplx<To>()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<To>());

    auto finalBoundaryReduceDim =
        common::getKernel("final_boundary_reduce_dim",
                          {{ops_cl_src, reduce_by_key_boundary_dim_cl_src}},
                          tmpltArgs, compileOpts);

    cl::NDRange local(threads_x);
    cl::NDRange global(threads_x * numBlocks,
                       vals_out.info.dims[dim_ordering[1]],
                       vals_out.info.dims[dim_ordering[2]] *
                           vals_out.info.dims[dim_ordering[3]]);

    finalBoundaryReduceDim(
        cl::EnqueueArgs(getQueue(), global, local), *reduced_block_sizes,
        *keys_out.data, keys_out.info, *vals_out.data, vals_out.info, n,
        static_cast<int>(vals_out.info.dims[dim_ordering[2]]));
    CL_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename To>
void compact(cl::Buffer *reduced_block_sizes, Param keys_out, Param vals_out,
             const Param keys, const Param vals, const int numBlocks,
             const int threads_x) {
    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<To>(),
        TemplateTypename<Tk>(),
        TemplateArg(threads_x),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(Tk, dtype_traits<Tk>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(DIMX, threads_x),
        DefineKeyValue(CPLX, iscplx<To>()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<To>());

    auto compact = common::getKernel(
        "compact", {{ops_cl_src, reduce_by_key_compact_cl_src}}, tmpltArgs,
        compileOpts);

    cl::NDRange local(threads_x);
    cl::NDRange global(threads_x * numBlocks, vals_out.info.dims[1],
                       vals_out.info.dims[2] * vals_out.info.dims[3]);

    compact(cl::EnqueueArgs(getQueue(), global, local), *reduced_block_sizes,
            *keys_out.data, keys_out.info, *vals_out.data, vals_out.info,
            *keys.data, keys.info, *vals.data, vals.info,
            static_cast<int>(vals_out.info.dims[2]));
    CL_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename To>
void compactDim(cl::Buffer *reduced_block_sizes, Param keys_out, Param vals_out,
                const Param keys, const Param vals, const int numBlocks,
                const int threads_x, const int dim,
                std::vector<int> dim_ordering) {
    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<To>(),
        TemplateTypename<Tk>(),
        TemplateArg(threads_x),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(Tk, dtype_traits<Tk>::getName()),
        DefineKeyValue(To, dtype_traits<To>::getName()),
        DefineKeyValue(T, "To"),
        DefineKeyValue(DIMX, threads_x),
        DefineKeyValue(DIM, dim),
        DefineKeyValue(CPLX, iscplx<To>()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<To>());

    auto compactDim = common::getKernel(
        "compact_dim", {{ops_cl_src, reduce_by_key_compact_dim_cl_src}},
        tmpltArgs, compileOpts);

    cl::NDRange local(threads_x);
    cl::NDRange global(threads_x * numBlocks,
                       vals_out.info.dims[dim_ordering[1]],
                       vals_out.info.dims[dim_ordering[2]] *
                           vals_out.info.dims[dim_ordering[3]]);

    compactDim(cl::EnqueueArgs(getQueue(), global, local), *reduced_block_sizes,
               *keys_out.data, keys_out.info, *vals_out.data, vals_out.info,
               *keys.data, keys.info, *vals.data, vals.info,
               static_cast<int>(vals_out.info.dims[dim_ordering[2]]));
    CL_DEBUG_FINISH(getQueue());
}

template<typename Tk>
void testNeedsReduction(cl::Buffer needs_reduction, cl::Buffer needs_boundary,
                        const Param keys, const int n, const int numBlocks,
                        const int threads_x) {
    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<Tk>(),
        TemplateArg(threads_x),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(Tk, dtype_traits<Tk>::getName()),
        DefineKeyValue(DIMX, threads_x),
    };

    auto testIfNeedsReduction =
        common::getKernel("test_needs_reduction",
                          {{ops_cl_src, reduce_by_key_needs_reduction_cl_src}},
                          tmpltArgs, compileOpts);

    cl::NDRange local(threads_x);
    cl::NDRange global(threads_x * numBlocks);

    testIfNeedsReduction(cl::EnqueueArgs(getQueue(), global, local),
                         needs_reduction, needs_boundary, *keys.data, keys.info,
                         n);
    CL_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op>
int reduceByKeyFirst(Array<Tk> &keys_out, Array<To> &vals_out, const Param keys,
                     const Param vals, bool change_nan, double nanval) {
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
            reduceBlocksByKey<Ti, Tk, To, op>(
                reduced_block_sizes.get(), reduced_keys, reduced_vals, keys,
                vals, change_nan, nanval, n_reduced_host, numThreads);
            first_pass = false;
        } else {
            constexpr af_op_t op2 = op == af_notzero_t ? af_add_t : op;
            reduceBlocksByKey<To, Tk, To, op2>(
                reduced_block_sizes.get(), reduced_keys, reduced_vals,
                t_reduced_keys, t_reduced_vals, change_nan, nanval,
                n_reduced_host, numThreads);
        }

        compute::inclusive_scan(
            compute::make_buffer_iterator<int>(val_buf),
            compute::make_buffer_iterator<int>(val_buf, numBlocksD0),
            compute::make_buffer_iterator<int>(val_buf), c_queue);

        compact<Tk, To>(reduced_block_sizes.get(), t_reduced_keys,
                        t_reduced_vals, reduced_keys, reduced_vals, numBlocksD0,
                        numThreads);

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

        testNeedsReduction<Tk>(*needs_another_reduction.get(),
                               *needs_block_boundary_reduction.get(),
                               t_reduced_keys, n_reduced_host, numBlocksD0,
                               numThreads);

        getQueue().enqueueReadBuffer(*needs_another_reduction.get(), CL_FALSE,
                                     0, sizeof(int),
                                     &needs_another_reduction_host);
        getQueue().enqueueReadBuffer(*needs_block_boundary_reduction.get(),
                                     CL_TRUE, 0, sizeof(int),
                                     &needs_block_boundary_reduction_host);

        if (needs_block_boundary_reduction_host &&
            !needs_another_reduction_host) {
            finalBoundaryReduce<Tk, To, op>(
                reduced_block_sizes.get(), t_reduced_keys, t_reduced_vals,
                n_reduced_host, numBlocksD0, numThreads);

            compute::inclusive_scan(
                compute::make_buffer_iterator<int>(val_buf),
                compute::make_buffer_iterator<int>(val_buf, numBlocksD0),
                compute::make_buffer_iterator<int>(val_buf), c_queue);

            getQueue().enqueueReadBuffer(*reduced_block_sizes.get(), true,
                                         (numBlocksD0 - 1) * sizeof(int),
                                         sizeof(int), &n_reduced_host);

            compact<Tk, To>(reduced_block_sizes.get(), reduced_keys,
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
int reduceByKeyDim(Array<Tk> &keys_out, Array<To> &vals_out, const Param keys,
                   const Param vals, bool change_nan, double nanval,
                   const int dim) {
    std::vector<int> dim_ordering = {dim};
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
            reduceBlocksByKeyDim<Ti, Tk, To, op>(
                reduced_block_sizes.get(), reduced_keys, reduced_vals, keys,
                vals, change_nan, nanval, n_reduced_host, numThreads, dim,
                dim_ordering);
            first_pass = false;
        } else {
            constexpr af_op_t op2 = op == af_notzero_t ? af_add_t : op;
            reduceBlocksByKeyDim<To, Tk, To, op2>(
                reduced_block_sizes.get(), reduced_keys, reduced_vals,
                t_reduced_keys, t_reduced_vals, change_nan, nanval,
                n_reduced_host, numThreads, dim, dim_ordering);
        }

        compute::inclusive_scan(
            compute::make_buffer_iterator<int>(val_buf),
            compute::make_buffer_iterator<int>(val_buf, numBlocksD0),
            compute::make_buffer_iterator<int>(val_buf), c_queue);

        compactDim<Tk, To>(reduced_block_sizes.get(), t_reduced_keys,
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

        testNeedsReduction<Tk>(*needs_another_reduction.get(),
                               *needs_block_boundary_reduction.get(),
                               t_reduced_keys, n_reduced_host, numBlocksD0,
                               numThreads);

        getQueue().enqueueReadBuffer(*needs_another_reduction.get(), CL_FALSE,
                                     0, sizeof(int),
                                     &needs_another_reduction_host);
        getQueue().enqueueReadBuffer(*needs_block_boundary_reduction.get(),
                                     CL_TRUE, 0, sizeof(int),
                                     &needs_block_boundary_reduction_host);

        if (needs_block_boundary_reduction_host &&
            !needs_another_reduction_host) {
            finalBoundaryReduceDim<Tk, To, op>(
                reduced_block_sizes.get(), t_reduced_keys, t_reduced_vals,
                n_reduced_host, numBlocksD0, numThreads, dim, dim_ordering);

            compute::inclusive_scan(
                compute::make_buffer_iterator<int>(val_buf),
                compute::make_buffer_iterator<int>(val_buf, numBlocksD0),
                compute::make_buffer_iterator<int>(val_buf), c_queue);

            getQueue().enqueueReadBuffer(*reduced_block_sizes.get(), true,
                                         (numBlocksD0 - 1) * sizeof(int),
                                         sizeof(int), &n_reduced_host);

            compactDim<Tk, To>(reduced_block_sizes.get(), reduced_keys,
                               reduced_vals, t_reduced_keys, t_reduced_vals,
                               numBlocksD0, numThreads, dim, dim_ordering);

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
void reduceByKey(Array<Tk> &keys_out, Array<To> &vals_out,
                 const Array<Tk> &keys, const Array<Ti> &vals, int dim,
                 bool change_nan, double nanval) {
    dim4 kdims = keys.dims();
    dim4 odims = vals.dims();

    // allocate space for output arrays
    Array<Tk> reduced_keys = createEmptyArray<Tk>(dim4());
    Array<To> reduced_vals = createEmptyArray<To>(dim4());

    int n_reduced = 0;
    if (dim == 0) {
        n_reduced = reduceByKeyFirst<Ti, Tk, To, op>(
            reduced_keys, reduced_vals, keys, vals, change_nan, nanval);
    } else {
        n_reduced = reduceByKeyDim<Ti, Tk, To, op>(
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
}  // namespace arrayfire

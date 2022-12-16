/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/dim4.hpp>

#undef _GLIBCXX_USE_INT128
#include <Array.hpp>
#include <Event.hpp>
#include <err_cuda.hpp>
#include <kernel/reduce.hpp>
#include <kernel/reduce_by_key.hpp>
#include <reduce.hpp>
#include <set.hpp>

#include <cub/device/device_scan.cuh>

#include <complex>

using af::dim4;
using std::swap;

namespace arrayfire {
namespace cuda {
template<af_op_t op, typename Ti, typename To>
Array<To> reduce(const Array<Ti> &in, const int dim, bool change_nan,
                 double nanval) {
    dim4 odims    = in.dims();
    odims[dim]    = 1;
    Array<To> out = createEmptyArray<To>(odims);
    kernel::reduce<Ti, To, op>(out, in, dim, change_nan, nanval);
    return out;
}

template<af_op_t op, typename Ti, typename Tk, typename To>
void reduce_by_key_dim(Array<Tk> &keys_out, Array<To> &vals_out,
                       const Array<Tk> &keys, const Array<Ti> &vals,
                       bool change_nan, double nanval, const int dim) {
    std::vector<int> dim_ordering = {dim};
    for (int i = 0; i < 4; ++i) {
        if (i != dim) { dim_ordering.push_back(i); }
    }

    dim4 kdims = keys.dims();
    dim4 odims = vals.dims();

    // allocate space for output and temporary working arrays
    Array<Tk> reduced_keys   = createEmptyArray<Tk>(kdims);
    Array<To> reduced_vals   = createEmptyArray<To>(odims);
    Array<Tk> t_reduced_keys = createEmptyArray<Tk>(kdims);
    Array<To> t_reduced_vals = createEmptyArray<To>(odims);

    // flags determining more reduction is necessary
    auto needs_another_reduction        = memAlloc<int>(1);
    auto needs_block_boundary_reduction = memAlloc<int>(1);

    // reset flags
    CUDA_CHECK(cudaMemsetAsync(needs_another_reduction.get(), 0, sizeof(int),
                               getActiveStream()));
    CUDA_CHECK(cudaMemsetAsync(needs_block_boundary_reduction.get(), 0,
                               sizeof(int), getActiveStream()));

    int nelems = kdims[0];

    const unsigned int numThreads = 128;
    int numBlocksD0               = divup(nelems, numThreads);

    auto reduced_block_sizes = memAlloc<int>(numBlocksD0);

    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        NULL, temp_storage_bytes, reduced_block_sizes.get(),
        reduced_block_sizes.get(), numBlocksD0, getActiveStream());
    auto d_temp_storage = memAlloc<char>(temp_storage_bytes);

    int n_reduced_host = nelems;
    int needs_another_reduction_host;
    int needs_block_boundary_reduction_host;

    bool first_pass = true;
    do {
        numBlocksD0 = divup(n_reduced_host, numThreads);
        dim3 blocks(numBlocksD0, odims[dim_ordering[1]],
                    odims[dim_ordering[2]] * odims[dim_ordering[3]]);

        int folded_dim_sz = odims[dim_ordering[2]];
        if (first_pass) {
            CUDA_LAUNCH(
                (kernel::reduce_blocks_dim_by_key<Ti, Tk, To, op, numThreads>),
                blocks, numThreads, reduced_block_sizes.get(), reduced_keys,
                reduced_vals, keys, vals, nelems, change_nan,
                scalar<To>(nanval), dim, folded_dim_sz);
            POST_LAUNCH_CHECK();
            first_pass = false;
        } else {
            constexpr af_op_t op2 = op == af_notzero_t ? af_add_t : op;
            CUDA_LAUNCH(
                (kernel::reduce_blocks_dim_by_key<To, Tk, To, op2, numThreads>),
                blocks, numThreads, reduced_block_sizes.get(), reduced_keys,
                reduced_vals, t_reduced_keys, t_reduced_vals, n_reduced_host,
                change_nan, scalar<To>(nanval), dim, folded_dim_sz);
            POST_LAUNCH_CHECK();
        }

        cub::DeviceScan::InclusiveSum(
            (void *)d_temp_storage.get(), temp_storage_bytes,
            reduced_block_sizes.get(), reduced_block_sizes.get(), numBlocksD0,
            getActiveStream());

        CUDA_LAUNCH((kernel::compact_dim<Tk, To>), blocks, numThreads,
                    reduced_block_sizes.get(), t_reduced_keys, t_reduced_vals,
                    reduced_keys, reduced_vals, dim, folded_dim_sz);
        POST_LAUNCH_CHECK();

        CUDA_CHECK(cudaMemcpyAsync(
            &n_reduced_host, reduced_block_sizes.get() + (numBlocksD0 - 1),
            sizeof(int), cudaMemcpyDeviceToHost, getActiveStream()));
        Event reduce_host_event = makeEvent(getActiveStream());

        // reset flags
        CUDA_CHECK(cudaMemsetAsync(needs_another_reduction.get(), 0,
                                   sizeof(int), getActiveStream()));
        CUDA_CHECK(cudaMemsetAsync(needs_block_boundary_reduction.get(), 0,
                                   sizeof(int), getActiveStream()));

        reduce_host_event.block();
        numBlocksD0 = divup(n_reduced_host, numThreads);

        CUDA_LAUNCH((kernel::test_needs_reduction<Tk>), numBlocksD0, numThreads,
                    needs_another_reduction.get(),
                    needs_block_boundary_reduction.get(), t_reduced_keys,
                    n_reduced_host);
        POST_LAUNCH_CHECK();

        CUDA_CHECK(cudaMemcpyAsync(&needs_another_reduction_host,
                                   needs_another_reduction.get(), sizeof(int),
                                   cudaMemcpyDeviceToHost, getActiveStream()));
        CUDA_CHECK(cudaMemcpyAsync(&needs_block_boundary_reduction_host,
                                   needs_block_boundary_reduction.get(),
                                   sizeof(int), cudaMemcpyDeviceToHost,
                                   getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(getActiveStream()));

        if (needs_block_boundary_reduction_host &&
            !needs_another_reduction_host) {
            dim3 blocks(numBlocksD0, odims[dim_ordering[1]],
                        odims[dim_ordering[2]] * odims[dim_ordering[3]]);
            CUDA_LAUNCH((kernel::final_boundary_reduce<Tk, To, op>), blocks,
                        numThreads, reduced_block_sizes.get(), t_reduced_keys,
                        t_reduced_vals, n_reduced_host);
            POST_LAUNCH_CHECK();

            cub::DeviceScan::InclusiveSum(
                (void *)d_temp_storage.get(), temp_storage_bytes,
                reduced_block_sizes.get(), reduced_block_sizes.get(),
                numBlocksD0, getActiveStream());

            CUDA_CHECK(cudaMemcpyAsync(
                &n_reduced_host, reduced_block_sizes.get() + (numBlocksD0 - 1),
                sizeof(int), cudaMemcpyDeviceToHost, getActiveStream()));
            reduce_host_event.mark(getActiveStream());

            CUDA_LAUNCH((kernel::compact_dim<Tk, To>), blocks, numThreads,
                        reduced_block_sizes.get(), reduced_keys, reduced_vals,
                        t_reduced_keys, t_reduced_vals, dim, folded_dim_sz);
            POST_LAUNCH_CHECK();

            swap(t_reduced_keys, reduced_keys);
            swap(t_reduced_vals, reduced_vals);
            reduce_host_event.block();
        }
    } while (needs_another_reduction_host ||
             needs_block_boundary_reduction_host);

    kdims[0]   = n_reduced_host;
    odims[dim] = n_reduced_host;
    std::vector<af_seq> kindex, vindex;
    for (int i = 0; i < odims.ndims(); ++i) {
        af_seq sk = {0.0, (double)kdims[i] - 1, 1.0};
        af_seq sv = {0.0, (double)odims[i] - 1, 1.0};
        kindex.push_back(sk);
        vindex.push_back(sv);
    }

    keys_out = createSubArray<Tk>(t_reduced_keys, kindex, true);
    vals_out = createSubArray<To>(t_reduced_vals, vindex, true);
}

template<af_op_t op, typename Ti, typename Tk, typename To>
void reduce_by_key_first(Array<Tk> &keys_out, Array<To> &vals_out,
                         const Array<Tk> &keys, const Array<Ti> &vals,
                         bool change_nan, double nanval) {
    dim4 kdims = keys.dims();
    dim4 odims = vals.dims();

    // allocate space for output and temporary working arrays
    Array<Tk> reduced_keys   = createEmptyArray<Tk>(kdims);
    Array<To> reduced_vals   = createEmptyArray<To>(odims);
    Array<Tk> t_reduced_keys = createEmptyArray<Tk>(kdims);
    Array<To> t_reduced_vals = createEmptyArray<To>(odims);

    // flags determining more reduction is necessary
    auto needs_another_reduction        = memAlloc<int>(1);
    auto needs_block_boundary_reduction = memAlloc<int>(1);

    // reset flags
    CUDA_CHECK(cudaMemsetAsync(needs_another_reduction.get(), 0, sizeof(int),
                               getActiveStream()));
    CUDA_CHECK(cudaMemsetAsync(needs_block_boundary_reduction.get(), 0,
                               sizeof(int), getActiveStream()));

    int nelems = kdims[0];

    const unsigned int numThreads = 128;
    int numBlocksD0               = divup(nelems, numThreads);

    auto reduced_block_sizes = memAlloc<int>(numBlocksD0);

    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        NULL, temp_storage_bytes, reduced_block_sizes.get(),
        reduced_block_sizes.get(), numBlocksD0, getActiveStream());
    auto d_temp_storage = memAlloc<char>(temp_storage_bytes);

    int n_reduced_host = nelems;
    int needs_another_reduction_host;
    int needs_block_boundary_reduction_host;

    bool first_pass = true;
    do {
        numBlocksD0 = divup(n_reduced_host, numThreads);
        dim3 blocks(numBlocksD0, odims[1], odims[2] * odims[3]);

        if (first_pass) {
            CUDA_LAUNCH(
                (kernel::reduce_blocks_by_key<Ti, Tk, To, op, numThreads>),
                blocks, numThreads, reduced_block_sizes.get(), reduced_keys,
                reduced_vals, keys, vals, nelems, change_nan,
                scalar<To>(nanval), odims[2]);
            POST_LAUNCH_CHECK();
            first_pass = false;
        } else {
            constexpr af_op_t op2 = op == af_notzero_t ? af_add_t : op;
            CUDA_LAUNCH(
                (kernel::reduce_blocks_by_key<To, Tk, To, op2, numThreads>),
                blocks, numThreads, reduced_block_sizes.get(), reduced_keys,
                reduced_vals, t_reduced_keys, t_reduced_vals, n_reduced_host,
                change_nan, scalar<To>(nanval), odims[2]);
            POST_LAUNCH_CHECK();
        }

        cub::DeviceScan::InclusiveSum(
            (void *)d_temp_storage.get(), temp_storage_bytes,
            reduced_block_sizes.get(), reduced_block_sizes.get(), numBlocksD0,
            getActiveStream());

        CUDA_LAUNCH((kernel::compact<Tk, To>), blocks, numThreads,
                    reduced_block_sizes.get(), t_reduced_keys, t_reduced_vals,
                    reduced_keys, reduced_vals, odims[2]);
        POST_LAUNCH_CHECK();

        CUDA_CHECK(cudaMemcpyAsync(
            &n_reduced_host, reduced_block_sizes.get() + (numBlocksD0 - 1),
            sizeof(int), cudaMemcpyDeviceToHost, getActiveStream()));
        Event reduce_host_event = makeEvent(getActiveStream());

        // reset flags
        CUDA_CHECK(cudaMemsetAsync(needs_another_reduction.get(), 0,
                                   sizeof(int), getActiveStream()));
        CUDA_CHECK(cudaMemsetAsync(needs_block_boundary_reduction.get(), 0,
                                   sizeof(int), getActiveStream()));

        reduce_host_event.block();
        numBlocksD0 = divup(n_reduced_host, numThreads);

        CUDA_LAUNCH((kernel::test_needs_reduction<Tk>), numBlocksD0, numThreads,
                    needs_another_reduction.get(),
                    needs_block_boundary_reduction.get(), t_reduced_keys,
                    n_reduced_host);
        POST_LAUNCH_CHECK();

        CUDA_CHECK(cudaMemcpyAsync(&needs_another_reduction_host,
                                   needs_another_reduction.get(), sizeof(int),
                                   cudaMemcpyDeviceToHost, getActiveStream()));
        CUDA_CHECK(cudaMemcpyAsync(&needs_block_boundary_reduction_host,
                                   needs_block_boundary_reduction.get(),
                                   sizeof(int), cudaMemcpyDeviceToHost,
                                   getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(getActiveStream()));

        if (needs_block_boundary_reduction_host &&
            !needs_another_reduction_host) {
            // TODO: fold 3,4 dimensions
            blocks = dim3(numBlocksD0, odims[1], odims[2]);
            CUDA_LAUNCH((kernel::final_boundary_reduce<Tk, To, op>), blocks,
                        numThreads, reduced_block_sizes.get(), t_reduced_keys,
                        t_reduced_vals, n_reduced_host);
            POST_LAUNCH_CHECK();

            cub::DeviceScan::InclusiveSum(
                (void *)d_temp_storage.get(), temp_storage_bytes,
                reduced_block_sizes.get(), reduced_block_sizes.get(),
                numBlocksD0, getActiveStream());

            CUDA_CHECK(cudaMemcpyAsync(
                &n_reduced_host, reduced_block_sizes.get() + (numBlocksD0 - 1),
                sizeof(int), cudaMemcpyDeviceToHost, getActiveStream()));
            reduce_host_event.mark(getActiveStream());

            CUDA_LAUNCH((kernel::compact<Tk, To>), blocks, numThreads,
                        reduced_block_sizes.get(), reduced_keys, reduced_vals,
                        t_reduced_keys, t_reduced_vals, odims[2]);
            POST_LAUNCH_CHECK();

            swap(t_reduced_keys, reduced_keys);
            swap(t_reduced_vals, reduced_vals);
            reduce_host_event.block();
        }
    } while (needs_another_reduction_host ||
             needs_block_boundary_reduction_host);

    kdims[0] = n_reduced_host;
    odims[0] = n_reduced_host;
    std::vector<af_seq> kindex, vindex;
    for (int i = 0; i < odims.ndims(); ++i) {
        af_seq sk = {0.0, (double)kdims[i] - 1, 1.0};
        af_seq sv = {0.0, (double)odims[i] - 1, 1.0};
        kindex.push_back(sk);
        vindex.push_back(sv);
    }

    keys_out = createSubArray<Tk>(t_reduced_keys, kindex, true);
    vals_out = createSubArray<To>(t_reduced_vals, vindex, true);
}

template<af_op_t op, typename Ti, typename Tk, typename To>
void reduce_by_key(Array<Tk> &keys_out, Array<To> &vals_out,
                   const Array<Tk> &keys, const Array<Ti> &vals, const int dim,
                   bool change_nan, double nanval) {
    if (dim == 0) {
        reduce_by_key_first<op, Ti, Tk, To>(keys_out, vals_out, keys, vals,
                                            change_nan, nanval);
    } else {
        reduce_by_key_dim<op, Ti, Tk, To>(keys_out, vals_out, keys, vals,
                                          change_nan, nanval, dim);
    }
}

template<af_op_t op, typename Ti, typename To>
To reduce_all(const Array<Ti> &in, bool change_nan, double nanval) {
    return kernel::reduce_all<Ti, To, op>(in, change_nan, nanval);
}
}  // namespace cuda
}  // namespace arrayfire

#define INSTANTIATE(Op, Ti, To)                                                \
    template Array<To> reduce<Op, Ti, To>(const Array<Ti> &in, const int dim,  \
                                          bool change_nan, double nanval);     \
    template void reduce_by_key<Op, Ti, int, To>(                              \
        Array<int> & keys_out, Array<To> & vals_out, const Array<int> &keys,   \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval); \
    template void reduce_by_key<Op, Ti, uint, To>(                             \
        Array<uint> & keys_out, Array<To> & vals_out, const Array<uint> &keys, \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval); \
    template To reduce_all<Op, Ti, To>(const Array<Ti> &in, bool change_nan,   \
                                       double nanval);

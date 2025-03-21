/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#if defined(__clang__)
#pragma clang diagnostic push
// temporary ignores for DPL internals
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

// oneDPL headers should be included before standard headers
#define ONEDPL_USE_PREDEFINED_POLICIES 0
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>

#include <Array.hpp>
#include <err_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <kernel/reduce.hpp>
#include <kernel/reduce_by_key.hpp>
#include <reduce.hpp>
#include <af/dim4.hpp>
#include <complex>

using af::dim4;
using std::swap;

namespace arrayfire {
namespace oneapi {

template<af_op_t op, typename Ti, typename To>
Array<To> reduce(const Array<Ti> &in, const int dim, bool change_nan,
                 double nanval) {
    dim4 odims    = in.dims();
    odims[dim]    = 1;
    Array<To> out = createEmptyArray<To>(odims);
    kernel::reduce<Ti, To, op>(out, in, dim, change_nan, nanval);
    return out;
}

template<typename Ti, typename Tk, typename To, af_op_t op>
void reduceBlocksByKey(sycl::buffer<int> &reduced_block_sizes,
                       Array<Tk> keys_out, Array<To> vals_out,
                       const Array<Tk> keys, const Array<Ti> vals,
                       int change_nan, double nanval, const int n,
                       const int threads_x) {
    int numBlocks = divup(n, threads_x);

    sycl::range<3> local(threads_x, 1, 1);
    sycl::range<3> global(local[0] * numBlocks, vals_out.dims()[1],
                          vals_out.dims()[2] * vals_out.dims()[3]);

    auto keys_out_get = keys_out.get();
    auto vals_out_get = vals_out.get();
    auto keys_get = keys.get();
    auto vals_get = vals.get();
    getQueue().submit([&](sycl::handler &h) {
        sycl::accessor<int> reduced_block_sizes_acc{reduced_block_sizes, h};
        write_accessor<Tk> keys_out_acc{*keys_out_get, h};
        write_accessor<To> vals_out_acc{*vals_out_get, h};
        read_accessor<Tk> keys_acc{*keys_get, h};
        read_accessor<Ti> vals_acc{*vals_get, h};

        auto l_keys         = sycl::local_accessor<Tk>(threads_x, h);
        auto l_vals         = sycl::local_accessor<compute_t<To>>(threads_x, h);
        auto l_reduced_keys = sycl::local_accessor<Tk>(threads_x, h);
        auto l_reduced_vals = sycl::local_accessor<compute_t<To>>(threads_x, h);
        auto l_unique_ids   = sycl::local_accessor<int>(threads_x, h);
        auto l_wq_temp      = sycl::local_accessor<int>(threads_x, h);
        auto l_unique_flags = sycl::local_accessor<int>(threads_x, h);
        auto l_reduced_block_size = sycl::local_accessor<int>(1, h);

        h.parallel_for(
            sycl::nd_range<3>(global, local),
            kernel::reduceBlocksByKeyKernel<Ti, Tk, To, op>(
                reduced_block_sizes_acc, keys_out_acc, keys_out, vals_out_acc,
                vals_out, keys_acc, keys, vals_acc, vals, change_nan,
                scalar<To>(nanval), n, static_cast<int>(vals_out.dims()[2]),
                threads_x, l_keys, l_vals, l_reduced_keys, l_reduced_vals,
                l_unique_ids, l_wq_temp, l_unique_flags, l_reduced_block_size));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename Ti, typename Tk, typename To, af_op_t op>
void reduceBlocksByKeyDim(sycl::buffer<int> &reduced_block_sizes,
                          Array<Tk> keys_out, Array<To> vals_out,
                          const Array<Tk> keys, const Array<Ti> vals,
                          int change_nan, double nanval, const int n,
                          const int threads_x, const int dim,
                          std::vector<int> dim_ordering) {
    int numBlocks = divup(n, threads_x);

    sycl::range<3> local(threads_x, 1, 1);
    sycl::range<3> global(
        local[0] * numBlocks, vals_out.dims()[dim_ordering[1]],
        vals_out.dims()[dim_ordering[2]] * vals_out.dims()[dim_ordering[3]]);

    auto keys_out_get = keys_out.get();
    auto vals_out_get = vals_out.get();
    auto keys_get = keys.get();
    auto vals_get = vals.get();
    getQueue().submit([&](sycl::handler &h) {
        sycl::accessor<int> reduced_block_sizes_acc{reduced_block_sizes, h};
        write_accessor<Tk> keys_out_acc{*keys_out_get, h};
        write_accessor<To> vals_out_acc{*vals_out_get, h};
        read_accessor<Tk> keys_acc{*keys_get, h};
        read_accessor<Ti> vals_acc{*vals_get, h};

        auto l_keys         = sycl::local_accessor<Tk>(threads_x, h);
        auto l_vals         = sycl::local_accessor<compute_t<To>>(threads_x, h);
        auto l_reduced_keys = sycl::local_accessor<Tk>(threads_x, h);
        auto l_reduced_vals = sycl::local_accessor<compute_t<To>>(threads_x, h);
        auto l_unique_ids   = sycl::local_accessor<int>(threads_x, h);
        auto l_wq_temp      = sycl::local_accessor<int>(threads_x, h);
        auto l_unique_flags = sycl::local_accessor<int>(threads_x, h);
        auto l_reduced_block_size = sycl::local_accessor<int>(1, h);

        h.parallel_for(
            sycl::nd_range<3>(global, local),
            kernel::reduceBlocksByKeyDimKernel<Ti, Tk, To, op>(
                reduced_block_sizes_acc, keys_out_acc, keys_out, vals_out_acc,
                vals_out, keys_acc, keys, vals_acc, vals, change_nan,
                scalar<To>(nanval), n, static_cast<int>(vals_out.dims()[2]),
                threads_x, dim, l_keys, l_vals, l_reduced_keys, l_reduced_vals,
                l_unique_ids, l_wq_temp, l_unique_flags, l_reduced_block_size));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename To, af_op_t op>
void finalBoundaryReduce(sycl::buffer<int> &reduced_block_sizes, Array<Tk> keys,
                         Array<To> vals_out, const int n, const int numBlocks,
                         const int threads_x) {
    sycl::range<1> local(threads_x);
    sycl::range<1> global(local[0] * numBlocks);

    auto vals_out_get = vals_out.get();
    auto keys_get = keys.get();
    getQueue().submit([&](sycl::handler &h) {
        write_accessor<int> reduced_block_sizes_acc{reduced_block_sizes, h};
        read_accessor<Tk> keys_acc{*keys_get, h};
        sycl::accessor<To> vals_out_acc{*vals_out_get, h};

        h.parallel_for(sycl::nd_range<1>(global, local),
                       kernel::finalBoundaryReduceKernel<Tk, To, op>(
                           reduced_block_sizes_acc, keys_acc, keys,
                           vals_out_acc, vals_out, n));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename To, af_op_t op>
void finalBoundaryReduceDim(sycl::buffer<int> &reduced_block_sizes,
                            Array<Tk> keys, Array<To> vals_out, const int n,
                            const int numBlocks, const int threads_x,
                            const int dim, std::vector<int> dim_ordering) {
    sycl::range<3> local(threads_x, 1, 1);
    sycl::range<3> global(
        local[0] * numBlocks, vals_out.dims()[dim_ordering[1]],
        vals_out.dims()[dim_ordering[2]] * vals_out.dims()[dim_ordering[3]]);

    auto vals_out_get = vals_out.get();
    auto keys_get = keys.get();
    getQueue().submit([&](sycl::handler &h) {
        write_accessor<int> reduced_block_sizes_acc{reduced_block_sizes, h};
        read_accessor<Tk> keys_acc{*keys_get, h};
        sycl::accessor<To> vals_out_acc{*vals_out_get, h};

        // TODO: fold 3,4 dimensions
        h.parallel_for(
            sycl::nd_range<3>(global, local),
            kernel::finalBoundaryReduceDimKernel<Tk, To, op>(
                reduced_block_sizes_acc, keys_acc, keys, vals_out_acc, vals_out,
                n, vals_out.dims()[dim_ordering[2]]));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename To>
void compact(sycl::buffer<int> reduced_block_sizes, Array<Tk> &keys_out,
             Array<To> &vals_out, const Array<Tk> &keys, const Array<To> &vals,
             const int numBlocks, const int threads_x) {
    sycl::range<3> local(threads_x, 1, 1);
    sycl::range<3> global(local[0] * numBlocks, vals_out.dims()[1],
                          vals_out.dims()[2] * vals_out.dims()[3]);

    auto keys_out_get = keys_out.get();
    auto vals_out_get = vals_out.get();
    auto keys_get = keys.get();
    auto vals_get = vals.get();
    getQueue().submit([&](sycl::handler &h) {
        read_accessor<int> reduced_block_sizes_acc{reduced_block_sizes, h};
        write_accessor<Tk> keys_out_acc{*keys_out_get, h};
        write_accessor<To> vals_out_acc{*vals_out_get, h};
        read_accessor<Tk> keys_acc{*keys_get, h};
        read_accessor<To> vals_acc{*vals_get, h};

        h.parallel_for(sycl::nd_range<3>(global, local),
                       kernel::compactKernel<Tk, To>(
                           reduced_block_sizes_acc, keys_out_acc, keys_out,
                           vals_out_acc, vals_out, keys_acc, keys, vals_acc,
                           vals, static_cast<int>(vals_out.dims()[2])));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename Tk, typename To>
void compactDim(sycl::buffer<int> &reduced_block_sizes, Array<Tk> &keys_out,
                Array<To> &vals_out, const Array<Tk> &keys,
                const Array<To> &vals, const int numBlocks, const int threads_x,
                const int dim, std::vector<int> dim_ordering) {
    sycl::range<3> local(threads_x, 1, 1);
    sycl::range<3> global(
        local[0] * numBlocks, vals_out.dims()[dim_ordering[1]],
        vals_out.dims()[dim_ordering[2]] * vals_out.dims()[dim_ordering[3]]);

    auto keys_out_get = keys_out.get();
    auto vals_out_get = vals_out.get();
    auto keys_get = keys.get();
    auto vals_get = vals.get();
    getQueue().submit([&](sycl::handler &h) {
        read_accessor<int> reduced_block_sizes_acc{reduced_block_sizes, h};
        write_accessor<Tk> keys_out_acc{*keys_out_get, h};
        write_accessor<To> vals_out_acc{*vals_out_get, h};
        read_accessor<Tk> keys_acc{*keys_get, h};
        read_accessor<To> vals_acc{*vals_get, h};

        h.parallel_for(
            sycl::nd_range<3>(global, local),
            kernel::compactDimKernel<Tk, To>(
                reduced_block_sizes_acc, keys_out_acc, keys_out, vals_out_acc,
                vals_out, keys_acc, keys, vals_acc, vals,
                static_cast<int>(vals_out.dims()[dim_ordering[2]]), dim));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename Tk>
void testNeedsReduction(sycl::buffer<int> needs_reduction,
                        sycl::buffer<int> needs_boundary, const Array<Tk> &keys,
                        const int n, const int numBlocks, const int threads_x) {
    sycl::range<1> local(threads_x);
    sycl::range<1> global(local[0] * numBlocks);

    auto keys_get = keys.get();
    getQueue().submit([&](sycl::handler &h) {
        sycl::accessor<int> needs_reduction_acc{needs_reduction, h};
        sycl::accessor<int> needs_boundary_acc{needs_boundary, h};
        read_accessor<Tk> keys_acc{*keys_get, h};
        auto l_keys = sycl::local_accessor<Tk>(threads_x, h);

        h.parallel_for(sycl::nd_range<1>(global, local),
                       kernel::testNeedsReductionKernel<Tk>(
                           needs_reduction_acc, needs_boundary_acc, keys_acc,
                           keys, n, threads_x, l_keys));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<af_op_t op, typename Ti, typename Tk, typename To>
int reduce_by_key_first(Array<Tk> &keys_out, Array<To> &vals_out,
                        const Array<Tk> &keys, const Array<Ti> &vals,
                        bool change_nan, double nanval) {
    auto dpl_policy = ::oneapi::dpl::execution::make_device_policy(getQueue());

    dim4 kdims = keys.dims();
    dim4 odims = vals.dims();

    Array<Tk> reduced_keys   = createEmptyArray<Tk>(kdims);
    Array<To> reduced_vals   = createEmptyArray<To>(odims);
    Array<Tk> t_reduced_keys = createEmptyArray<Tk>(kdims);
    Array<To> t_reduced_vals = createEmptyArray<To>(odims);

    // flags determining more reduction is necessary
    auto needs_another_reduction        = memAlloc<int>(1);
    auto needs_block_boundary_reduction = memAlloc<int>(1);

    // reset flags
    getQueue().submit([&](sycl::handler &h) {
        auto wacc =
            needs_another_reduction->get_access<sycl::access_mode::write>(h);
        h.fill(wacc, 0);
    });
    getQueue().submit([&](sycl::handler &h) {
        auto wacc = needs_block_boundary_reduction
                        ->get_access<sycl::access_mode::write>(h);
        h.fill(wacc, 0);
    });

    size_t nelems = kdims[0];

    const unsigned int numThreads = 128;
    int numBlocksD0               = divup(nelems, numThreads);
    auto reduced_block_sizes      = memAlloc<int>(numBlocksD0);

    int n_reduced_host = nelems;

    int needs_another_reduction_host        = 0;
    int needs_block_boundary_reduction_host = 0;

    bool first_pass = true;
    do {
        numBlocksD0 = divup(n_reduced_host, numThreads);

        if (first_pass) {
            reduceBlocksByKey<Ti, Tk, To, op>(
                *reduced_block_sizes.get(), reduced_keys, reduced_vals, keys,
                vals, change_nan, nanval, n_reduced_host, numThreads);
            first_pass = false;
        } else {
            constexpr af_op_t op2 = (op == af_notzero_t) ? af_add_t : op;
            reduceBlocksByKey<To, Tk, To, op2>(
                *reduced_block_sizes.get(), reduced_keys, reduced_vals,
                t_reduced_keys, t_reduced_vals, change_nan, nanval,
                n_reduced_host, numThreads);
        }

        auto val_buf_begin = ::oneapi::dpl::begin(*reduced_block_sizes.get());
        auto val_buf_end   = val_buf_begin + numBlocksD0;
        std::inclusive_scan(dpl_policy, val_buf_begin, val_buf_end,
                            val_buf_begin);

        compact<Tk, To>(*reduced_block_sizes.get(), t_reduced_keys,
                        t_reduced_vals, reduced_keys, reduced_vals, numBlocksD0,
                        numThreads);

        sycl::event reduce_host_event =
            getQueue().submit([&](sycl::handler &h) {
                sycl::range rr(1);
                sycl::id offset_id(numBlocksD0 - 1);
                auto offset_acc =
                    reduced_block_sizes
                        ->template get_access<sycl::access_mode::read>(
                            h, rr, offset_id);
                h.copy(offset_acc, &n_reduced_host);
            });

        // reset flags
        getQueue().submit([&](sycl::handler &h) {
            auto wacc =
                needs_another_reduction->get_access<sycl::access_mode::write>(
                    h);
            h.fill(wacc, 0);
        });
        getQueue().submit([&](sycl::handler &h) {
            auto wacc = needs_block_boundary_reduction
                            ->get_access<sycl::access_mode::write>(h);
            h.fill(wacc, 0);
        });

        reduce_host_event.wait();

        numBlocksD0 = divup(n_reduced_host, numThreads);

        testNeedsReduction<Tk>(*needs_another_reduction.get(),
                               *needs_block_boundary_reduction.get(),
                               t_reduced_keys, n_reduced_host, numBlocksD0,
                               numThreads);

        sycl::event host_flag0_event = getQueue().submit([&](sycl::handler &h) {
            sycl::range rr(1);
            auto acc =
                needs_another_reduction
                    ->template get_access<sycl::access_mode::read>(h, rr);
            h.copy(acc, &needs_another_reduction_host);
        });
        sycl::event host_flag1_event = getQueue().submit([&](sycl::handler &h) {
            sycl::range rr(1);
            auto acc =
                needs_block_boundary_reduction
                    ->template get_access<sycl::access_mode::read>(h, rr);
            h.copy(acc, &needs_block_boundary_reduction_host);
        });

        host_flag1_event.wait();
        host_flag0_event.wait();

        if (needs_block_boundary_reduction_host &&
            !needs_another_reduction_host) {
            finalBoundaryReduce<Tk, To, op>(
                *reduced_block_sizes.get(), t_reduced_keys, t_reduced_vals,
                n_reduced_host, numBlocksD0, numThreads);

            auto val_buf_begin =
                ::oneapi::dpl::begin(*reduced_block_sizes.get());
            auto val_buf_end = val_buf_begin + numBlocksD0;
            std::inclusive_scan(dpl_policy, val_buf_begin, val_buf_end,
                                val_buf_begin);

            sycl::event reduce_host_event =
                getQueue().submit([&](sycl::handler &h) {
                    sycl::range rr(1);
                    sycl::id offset_id(numBlocksD0 - 1);
                    auto offset_acc =
                        reduced_block_sizes
                            ->template get_access<sycl::access_mode::read>(
                                h, rr, offset_id);
                    h.copy(offset_acc, &n_reduced_host);
                });

            compact<Tk, To>(*reduced_block_sizes.get(), reduced_keys,
                            reduced_vals, t_reduced_keys, t_reduced_vals,
                            numBlocksD0, numThreads);

            std::swap(t_reduced_keys, reduced_keys);
            std::swap(t_reduced_vals, reduced_vals);
            reduce_host_event.wait();
        }
    } while (needs_another_reduction_host ||
             needs_block_boundary_reduction_host);

    keys_out = t_reduced_keys;
    vals_out = t_reduced_vals;
    return n_reduced_host;
}

template<af_op_t op, typename Ti, typename Tk, typename To>
int reduce_by_key_dim(Array<Tk> &keys_out, Array<To> &vals_out,
                      const Array<Tk> &keys, const Array<Ti> &vals,
                      bool change_nan, double nanval, const int dim) {
    auto dpl_policy = ::oneapi::dpl::execution::make_device_policy(getQueue());

    std::vector<int> dim_ordering = {dim};
    for (int i = 0; i < 4; ++i) {
        if (i != dim) { dim_ordering.push_back(i); }
    }

    dim4 kdims = keys.dims();
    dim4 odims = vals.dims();

    Array<Tk> reduced_keys   = createEmptyArray<Tk>(kdims);
    Array<To> reduced_vals   = createEmptyArray<To>(odims);
    Array<Tk> t_reduced_keys = createEmptyArray<Tk>(kdims);
    Array<To> t_reduced_vals = createEmptyArray<To>(odims);

    // flags determining more reduction is necessary
    auto needs_another_reduction        = memAlloc<int>(1);
    auto needs_block_boundary_reduction = memAlloc<int>(1);

    // reset flags
    getQueue().submit([&](sycl::handler &h) {
        auto wacc =
            needs_another_reduction->get_access<sycl::access_mode::write>(h);
        h.fill(wacc, 0);
    });
    getQueue().submit([&](sycl::handler &h) {
        auto wacc = needs_block_boundary_reduction
                        ->get_access<sycl::access_mode::write>(h);
        h.fill(wacc, 0);
    });

    int nelems = kdims[0];

    const unsigned int numThreads = 128;
    int numBlocksD0               = divup(nelems, numThreads);
    auto reduced_block_sizes      = memAlloc<int>(numBlocksD0);

    int n_reduced_host = nelems;

    int needs_another_reduction_host        = 0;
    int needs_block_boundary_reduction_host = 0;

    bool first_pass = true;
    do {
        numBlocksD0 = divup(n_reduced_host, numThreads);

        if (first_pass) {
            reduceBlocksByKeyDim<Ti, Tk, To, op>(
                *reduced_block_sizes.get(), reduced_keys, reduced_vals, keys,
                vals, change_nan, nanval, n_reduced_host, numThreads, dim,
                dim_ordering);
            first_pass = false;
        } else {
            constexpr af_op_t op2 = op == af_notzero_t ? af_add_t : op;
            reduceBlocksByKeyDim<To, Tk, To, op2>(
                *reduced_block_sizes.get(), reduced_keys, reduced_vals,
                t_reduced_keys, t_reduced_vals, change_nan, nanval,
                n_reduced_host, numThreads, dim, dim_ordering);
        }

        auto val_buf_begin = ::oneapi::dpl::begin(*reduced_block_sizes.get());
        auto val_buf_end   = val_buf_begin + numBlocksD0;
        std::inclusive_scan(dpl_policy, val_buf_begin, val_buf_end,
                            val_buf_begin);

        compactDim<Tk, To>(*reduced_block_sizes.get(), t_reduced_keys,
                           t_reduced_vals, reduced_keys, reduced_vals,
                           numBlocksD0, numThreads, dim, dim_ordering);

        sycl::event reduce_host_event =
            getQueue().submit([&](sycl::handler &h) {
                sycl::range rr(1);
                sycl::id offset_id(numBlocksD0 - 1);
                auto offset_acc =
                    reduced_block_sizes
                        ->template get_access<sycl::access_mode::read>(
                            h, rr, offset_id);
                h.copy(offset_acc, &n_reduced_host);
            });

        // reset flags
        getQueue().submit([&](sycl::handler &h) {
            auto wacc =
                needs_another_reduction->get_access<sycl::access_mode::write>(
                    h);
            h.fill(wacc, 0);
        });
        getQueue().submit([&](sycl::handler &h) {
            auto wacc = needs_block_boundary_reduction
                            ->get_access<sycl::access_mode::write>(h);
            h.fill(wacc, 0);
        });

        reduce_host_event.wait();

        numBlocksD0 = divup(n_reduced_host, numThreads);

        testNeedsReduction<Tk>(*needs_another_reduction.get(),
                               *needs_block_boundary_reduction.get(),
                               t_reduced_keys, n_reduced_host, numBlocksD0,
                               numThreads);

        sycl::event host_flag0_event = getQueue().submit([&](sycl::handler &h) {
            sycl::range rr(1);
            auto acc =
                needs_another_reduction
                    ->template get_access<sycl::access_mode::read>(h, rr);
            h.copy(acc, &needs_another_reduction_host);
        });
        sycl::event host_flag1_event = getQueue().submit([&](sycl::handler &h) {
            sycl::range rr(1);
            auto acc =
                needs_block_boundary_reduction
                    ->template get_access<sycl::access_mode::read>(h, rr);
            h.copy(acc, &needs_block_boundary_reduction_host);
        });

        host_flag1_event.wait();
        host_flag0_event.wait();

        if (needs_block_boundary_reduction_host &&
            !needs_another_reduction_host) {
            finalBoundaryReduceDim<Tk, To, op>(
                *reduced_block_sizes.get(), t_reduced_keys, t_reduced_vals,
                n_reduced_host, numBlocksD0, numThreads, dim, dim_ordering);

            auto val_buf_begin =
                ::oneapi::dpl::begin(*reduced_block_sizes.get());
            auto val_buf_end = val_buf_begin + numBlocksD0;
            std::inclusive_scan(dpl_policy, val_buf_begin, val_buf_end,
                                val_buf_begin);

            sycl::event reduce_host_event =
                getQueue().submit([&](sycl::handler &h) {
                    sycl::range rr(1);
                    sycl::id offset_id(numBlocksD0 - 1);
                    auto offset_acc =
                        reduced_block_sizes
                            ->template get_access<sycl::access_mode::read>(
                                h, rr, offset_id);
                    h.copy(offset_acc, &n_reduced_host);
                });

            compactDim<Tk, To>(*reduced_block_sizes.get(), reduced_keys,
                               reduced_vals, t_reduced_keys, t_reduced_vals,
                               numBlocksD0, numThreads, dim, dim_ordering);

            std::swap(t_reduced_keys, reduced_keys);
            std::swap(t_reduced_vals, reduced_vals);
            reduce_host_event.wait();
        }
    } while (needs_another_reduction_host ||
             needs_block_boundary_reduction_host);

    keys_out = t_reduced_keys;
    vals_out = t_reduced_vals;

    return n_reduced_host;
}

template<af_op_t op, typename Ti, typename Tk, typename To>
void reduce_by_key(Array<Tk> &keys_out, Array<To> &vals_out,
                   const Array<Tk> &keys, const Array<Ti> &vals, const int dim,
                   bool change_nan, double nanval) {
    dim4 kdims = keys.dims();
    dim4 odims = vals.dims();

    // prepare output arrays
    Array<Tk> reduced_keys = createEmptyArray<Tk>(dim4());
    Array<To> reduced_vals = createEmptyArray<To>(dim4());

    size_t n_reduced = 0;
    if (dim == 0) {
        n_reduced = reduce_by_key_first<op, Ti, Tk, To>(
            reduced_keys, reduced_vals, keys, vals, change_nan, nanval);
    } else {
        n_reduced = reduce_by_key_dim<op, Ti, Tk, To>(
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

template<af_op_t op, typename Ti, typename To>
Array<To> reduce_all(const Array<Ti> &in, bool change_nan, double nanval) {
    Array<To> out = createEmptyArray<To>(1);
    kernel::reduce_all<Ti, To, op>(out, in, change_nan, nanval);
    return out;
}

}  // namespace oneapi
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
    template Array<To> reduce_all<Op, Ti, To>(const Array<Ti> &in,             \
                                              bool change_nan, double nanval);

#if defined(__clang__)
/* Clang/LLVM */
#pragma clang diagnostic pop
#endif

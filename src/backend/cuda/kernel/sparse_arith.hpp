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
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_cuda.hpp>
#include <nvrtc_kernel_headers/sparse_arith_cuh.hpp>
#include <optypes.hpp>

namespace arrayfire {
namespace cuda {
namespace kernel {

constexpr unsigned TX      = 32;
constexpr unsigned TY      = 8;
constexpr unsigned THREADS = TX * TY;

template<typename T, af_op_t op>
void sparseArithOpCSR(Param<T> out, CParam<T> values, CParam<int> rowIdx,
                      CParam<int> colIdx, CParam<T> rhs, const bool reverse) {
    auto csrArithDSD = common::getKernel(
        "arrayfire::cuda::csrArithDSD", {{sparse_arith_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(op)),
        {{DefineValue(TX), DefineValue(TY)}});

    // Each Y for threads does one row
    dim3 threads(TX, TY, 1);

    // No. of blocks = divup(no. of rows / threads.y). No blocks on Y
    dim3 blocks(divup(out.dims[0], TY), 1, 1);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    csrArithDSD(qArgs, out, values, rowIdx, colIdx, rhs, reverse);
    POST_LAUNCH_CHECK();
}

template<typename T, af_op_t op>
void sparseArithOpCOO(Param<T> out, CParam<T> values, CParam<int> rowIdx,
                      CParam<int> colIdx, CParam<T> rhs, const bool reverse) {
    auto cooArithDSD = common::getKernel(
        "arrayfire::cuda::cooArithDSD", {{sparse_arith_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(op)),
        {{DefineValue(THREADS)}});

    // Linear indexing with one elements per thread
    dim3 threads(THREADS, 1, 1);

    // No. of blocks = divup(no. of rows / threads.y). No blocks on Y
    dim3 blocks(divup(values.dims[0], THREADS), 1, 1);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    cooArithDSD(qArgs, out, values, rowIdx, colIdx, rhs, reverse);
    POST_LAUNCH_CHECK();
}

template<typename T, af_op_t op>
void sparseArithOpCSR(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
                      CParam<T> rhs, const bool reverse) {
    auto csrArithSSD = common::getKernel(
        "arrayfire::cuda::csrArithSSD", {{sparse_arith_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(op)),
        {{DefineValue(TX), DefineValue(TY)}});

    // Each Y for threads does one row
    dim3 threads(TX, TY, 1);

    // No. of blocks = divup(no. of rows / threads.y). No blocks on Y
    dim3 blocks(divup(rhs.dims[0], TY), 1, 1);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    csrArithSSD(qArgs, values, rowIdx, colIdx, rhs, reverse);
    POST_LAUNCH_CHECK();
}

template<typename T, af_op_t op>
void sparseArithOpCOO(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
                      CParam<T> rhs, const bool reverse) {
    auto cooArithSSD = common::getKernel(
        "arrayfire::cuda::cooArithSSD", {{sparse_arith_cuh_src}},
        TemplateArgs(TemplateTypename<T>(), TemplateArg(op)),
        {{DefineValue(THREADS)}});

    // Linear indexing with one elements per thread
    dim3 threads(THREADS, 1, 1);

    // No. of blocks = divup(no. of rows / threads.y). No blocks on Y
    dim3 blocks(divup(values.dims[0], THREADS), 1, 1);

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    cooArithSSD(qArgs, values, rowIdx, colIdx, rhs, reverse);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire

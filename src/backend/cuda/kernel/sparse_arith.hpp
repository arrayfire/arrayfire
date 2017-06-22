/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/dispatch.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <backend.hpp>

namespace cuda
{

namespace kernel
{

static const unsigned TX = 32;
static const unsigned TY = 8;
static const unsigned THREADS = TX * TY;

template<typename T, af_op_t op>
struct arith_op
{
    __DH__ T operator()(T v1, T v2)
    {
        return T(0);
    }
};

template<typename T>
struct arith_op<T, af_add_t>
{
    __device__ T operator()(T v1, T v2)
    {
        return v1 + v2;
    }
};

template<typename T>
struct arith_op<T, af_sub_t>
{
    __device__ T operator()(T v1, T v2)
    {
        return v1 - v2;
    }
};

template<typename T>
struct arith_op<T, af_mul_t>
{
    __device__ T operator()(T v1, T v2)
    {
        return v1 * v2;
    }
};

template<typename T>
struct arith_op<T, af_div_t>
{
    __device__ T operator()(T v1, T v2)
    {
        return v1 / v2;
    }
};

template<typename T, af_op_t op>
__global__
void sparseArithCSRKernel(Param<T> out,
                          CParam<T> values, CParam<int> rowIdx, CParam<int> colIdx,
                          CParam<T> rhs,
                          const bool reverse)
{
    const int row         = blockIdx.x * TY + threadIdx.y;

    if(row >= out.dims[0]) return;

    const int rowStartIdx = rowIdx.ptr[row  ];
    const int rowEndIdx   = rowIdx.ptr[row+1];

    // Repeat loop until all values in the row are computed
    for(int idx = rowStartIdx + threadIdx.x; idx < rowEndIdx; idx += TX) {
        const int col = colIdx.ptr[idx];

        if(row >= out.dims[0] || col >= out.dims[1]) continue;    // Bad indices

        // Get Values
        const T val  = values.ptr[idx];
        const T rval = rhs.ptr[col * rhs.strides[1] + row];

        const int offset = col * out.strides[1] + row;
        if(reverse) out.ptr[offset] = arith_op<T, op>()(rval, val);
        else        out.ptr[offset] = arith_op<T, op>()(val, rval);
    }
}

template<typename T, af_op_t op>
__global__
void sparseArithCOOKernel(Param<T> out,
                          CParam<T> values, CParam<int> rowIdx, CParam<int> colIdx,
                          CParam<T> rhs,
                          const bool reverse)
{
    const int idx = blockIdx.x * THREADS + threadIdx.x;

    if(idx >= values.dims[0]) return;

    const int row = rowIdx.ptr[idx];
    const int col = colIdx.ptr[idx];

    if(row >= out.dims[0] || col >= out.dims[1]) return;    // Bad indices

    // Get Values
    const T val  = values.ptr[idx];
    const T rval = rhs.ptr[col * rhs.strides[1] + row];

    const int offset = col * out.strides[1] + row;
    if(reverse) out.ptr[offset] = arith_op<T, op>()(rval, val);
    else        out.ptr[offset] = arith_op<T, op>()(val, rval);
}

template<typename T, af_op_t op>
void sparseArithOpCSR(Param<T> out,
                      CParam<T> values, CParam<int> rowIdx, CParam<int> colIdx,
                      CParam<T> rhs,
                      const bool reverse)
{
    // Each Y for threads does one row
    dim3 threads(TX, TY, 1);

    // No. of blocks = divup(no. of rows / threads.y). No blocks on Y
    dim3 blocks(divup(out.dims[0], TY), 1, 1);

    CUDA_LAUNCH((sparseArithCSRKernel<T, op>), blocks, threads,
                 out, values, rowIdx, colIdx, rhs, reverse);

    POST_LAUNCH_CHECK();
}

template<typename T, af_op_t op>
void sparseArithOpCOO(Param<T> out,
                      CParam<T> values, CParam<int> rowIdx, CParam<int> colIdx,
                      CParam<T> rhs,
                      const bool reverse)
{
    // Linear indexing with one elements per thread
    dim3 threads(THREADS, 1, 1);

    // No. of blocks = divup(no. of rows / threads.y). No blocks on Y
    dim3 blocks(divup(values.dims[0], THREADS), 1, 1);

    CUDA_LAUNCH((sparseArithCOOKernel<T, op>), blocks, threads,
                 out, values, rowIdx, colIdx, rhs, reverse);

    POST_LAUNCH_CHECK();
}

template<typename T, af_op_t op>
__global__
void sparseArithCSRKernel(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
                          CParam<T> rhs, const bool reverse)
{
    const int row         = blockIdx.x * TY + threadIdx.y;

    if(row >= rhs.dims[0]) return;

    const int rowStartIdx = rowIdx.ptr[row  ];
    const int rowEndIdx   = rowIdx.ptr[row+1];

    // Repeat loop until all values in the row are computed
    for(int idx = rowStartIdx + threadIdx.x; idx < rowEndIdx; idx += TX) {
        const int col = colIdx.ptr[idx];

        if(row >= rhs.dims[0] || col >= rhs.dims[1]) continue;    // Bad indices

        // Get Values
        const T val  = values.ptr[idx];
        const T rval = rhs.ptr[col * rhs.strides[1] + row];

        if(reverse) values.ptr[idx] = arith_op<T, op>()(rval, val);
        else        values.ptr[idx] = arith_op<T, op>()(val, rval);
    }
}

template<typename T, af_op_t op>
__global__
void sparseArithCOOKernel(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
                          CParam<T> rhs, const bool reverse)
{
    const int idx = blockIdx.x * THREADS + threadIdx.x;

    if(idx >= values.dims[0]) return;

    const int row = rowIdx.ptr[idx];
    const int col = colIdx.ptr[idx];

    if(row >= rhs.dims[0] || col >= rhs.dims[1]) return;    // Bad indices

    // Get Values
    const T val  = values.ptr[idx];
    const T rval = rhs.ptr[col * rhs.strides[1] + row];

    if(reverse) values.ptr[idx] = arith_op<T, op>()(rval, val);
    else        values.ptr[idx] = arith_op<T, op>()(val, rval);
}

template<typename T, af_op_t op>
void sparseArithOpCSR(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
                      CParam<T> rhs, const bool reverse)
{
    // Each Y for threads does one row
    dim3 threads(TX, TY, 1);

    // No. of blocks = divup(no. of rows / threads.y). No blocks on Y
    dim3 blocks(divup(rhs.dims[0], TY), 1, 1);

    CUDA_LAUNCH((sparseArithCSRKernel<T, op>), blocks, threads,
                 values, rowIdx, colIdx, rhs, reverse);

    POST_LAUNCH_CHECK();
}

template<typename T, af_op_t op>
void sparseArithOpCOO(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
                      CParam<T> rhs,
                      const bool reverse)
{
    // Linear indexing with one elements per thread
    dim3 threads(THREADS, 1, 1);

    // No. of blocks = divup(no. of rows / threads.y). No blocks on Y
    dim3 blocks(divup(values.dims[0], THREADS), 1, 1);

    CUDA_LAUNCH((sparseArithCOOKernel<T, op>), blocks, threads,
                 values, rowIdx, colIdx, rhs, reverse);

    POST_LAUNCH_CHECK();
}

} // namespace kernel

} // namespace cuda

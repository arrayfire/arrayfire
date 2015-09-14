/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <ops.hpp>
#include <backend.hpp>
#include <Param.hpp>
#include <dispatch.hpp>
#include <math.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include "config.hpp"
#include <memory.hpp>
#include <boost/scoped_ptr.hpp>

using boost::scoped_ptr;

namespace cuda
{
namespace kernel
{
    template<typename Ti, typename To, af_op_t op, uint dim, uint DIMY>
    __global__
    static void reduce_dim_kernel(Param<To> out,
                                  CParam <Ti> in,
                                  uint blocks_x, uint blocks_y, uint offset_dim,
                                  bool change_nan, To nanval)
    {
        const uint tidx = threadIdx.x;
        const uint tidy = threadIdx.y;
        const uint tid  = tidy * THREADS_X + tidx;

        const uint zid = blockIdx.x / blocks_x;
        const uint wid = blockIdx.y / blocks_y;
        const uint blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const uint blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const uint xid = blockIdx_x * blockDim.x + tidx;
        const uint yid = blockIdx_y; // yid  of output. updated for input later.

        uint ids[4] = {xid, yid, zid, wid};

        const Ti *iptr = in.ptr;
        To *optr = out.ptr;

        // There is only one element per block for out
        // There are blockDim.y elements per block for in
        // Hence increment ids[dim] just after offseting out and before offsetting in
        optr += ids[3] * out.strides[3] + ids[2] * out.strides[2] + ids[1] * out.strides[1] + ids[0];
        const uint blockIdx_dim = ids[dim];

        ids[dim] = ids[dim] * blockDim.y + tidy;
        iptr  += ids[3] * in.strides[3] + ids[2] * in.strides[2] + ids[1] * in.strides[1] + ids[0];
        const uint id_dim_in = ids[dim];

        const uint istride_dim = in.strides[dim];

        bool is_valid =
            (ids[0] < in.dims[0]) &&
            (ids[1] < in.dims[1]) &&
            (ids[2] < in.dims[2]) &&
            (ids[3] < in.dims[3]);

        Transform<Ti, To, op> transform;
        Binary<To, op> reduce;

        __shared__ To s_val[THREADS_X * DIMY];

        To out_val = reduce.init();
        for (int id = id_dim_in; is_valid && (id < in.dims[dim]); id += offset_dim * blockDim.y) {
            To in_val = transform(*iptr);
            if (change_nan) in_val = !IS_NAN(in_val) ? in_val : nanval;
            out_val = reduce(in_val, out_val);
            iptr = iptr + offset_dim * blockDim.y * istride_dim;
        }

        s_val[tid] = out_val;

        To *s_ptr = s_val + tid;
        __syncthreads();

        if (DIMY == 8) {
            if (tidy < 4) *s_ptr = reduce(*s_ptr, s_ptr[THREADS_X * 4]);
            __syncthreads();
        }

        if (DIMY >= 4) {
            if (tidy < 2) *s_ptr = reduce(*s_ptr, s_ptr[THREADS_X * 2]);
            __syncthreads();
        }

        if (DIMY >= 2) {
            if (tidy < 1) *s_ptr = reduce(*s_ptr, s_ptr[THREADS_X * 1]);
            __syncthreads();
        }

        if (tidy == 0 && is_valid &&
            (blockIdx_dim < out.dims[dim])) {
            *optr = *s_ptr;
        }

    }

    template<typename Ti, typename To, af_op_t op, int dim>
    void reduce_dim_launcher(Param<To> out, CParam<Ti> in,
                             const uint threads_y, const uint blocks_dim[4],
                             bool change_nan, double nanval)
    {
        dim3 threads(THREADS_X, threads_y);

        dim3 blocks(blocks_dim[0] * blocks_dim[2],
                    blocks_dim[1] * blocks_dim[3]);

        switch (threads_y) {
        case 8:
            CUDA_LAUNCH((reduce_dim_kernel<Ti, To, op, dim, 8>), blocks, threads,
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim],
                change_nan, scalar<To>(nanval)); break;
        case 4:
            CUDA_LAUNCH((reduce_dim_kernel<Ti, To, op, dim, 4>), blocks, threads,
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim],
                change_nan, scalar<To>(nanval)); break;
        case 2:
            CUDA_LAUNCH((reduce_dim_kernel<Ti, To, op, dim, 2>), blocks, threads,
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim],
                change_nan, scalar<To>(nanval)); break;
        case 1:
            CUDA_LAUNCH((reduce_dim_kernel<Ti, To, op, dim, 1>), blocks, threads,
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim],
                change_nan, scalar<To>(nanval)); break;
        }

        POST_LAUNCH_CHECK();
    }

    template<typename Ti, typename To, af_op_t op, int dim>
    void reduce_dim(Param<To> out,  CParam<Ti> in, bool change_nan, double nanval)
    {
        uint threads_y = std::min(THREADS_Y, nextpow2(in.dims[dim]));
        uint threads_x = THREADS_X;

        uint blocks_dim[] = {divup(in.dims[0], threads_x),
                             in.dims[1], in.dims[2], in.dims[3]};

        blocks_dim[dim] = divup(in.dims[dim], threads_y * REPEAT);

        Param<To> tmp = out;

        if (blocks_dim[dim] > 1) {
            int tmp_elements = 1;
            tmp.dims[dim] = blocks_dim[dim];

            for (int k = 0; k < 4; k++) tmp_elements *= tmp.dims[k];
            tmp.ptr = memAlloc<To>(tmp_elements);

            for (int k = dim + 1; k < 4; k++) tmp.strides[k] *= blocks_dim[dim];
        }

        reduce_dim_launcher<Ti, To, op, dim>(tmp, in, threads_y, blocks_dim, change_nan, nanval);

        if (blocks_dim[dim] > 1) {
            blocks_dim[dim] = 1;

            if (op == af_notzero_t) {
                reduce_dim_launcher<To, To, af_add_t, dim>(out, tmp, threads_y, blocks_dim,
                                                           change_nan, nanval);
            } else {
                reduce_dim_launcher<To, To,       op, dim>(out, tmp, threads_y, blocks_dim,
                                                           change_nan, nanval);
            }

            memFree(tmp.ptr);
        }

    }

    template<typename To, af_op_t op>
    struct WarpReduce
    {
        __device__ To operator()(To *s_ptr, uint tidx)
        {
            Binary<To, op> reduce;
#pragma unroll
            for (int n = 16; n >= 1; n >>= 1) {
                if (tidx < n) {
                    s_ptr[tidx] = reduce(s_ptr[tidx], s_ptr[tidx + n]);
                }
                __syncthreads();
            }
            return s_ptr[tidx];
        }
    };


#if (__CUDA_ARCH__ >= 300)
#define WARP_REDUCE(T)                                  \
    template<af_op_t op>                                \
    struct WarpReduce<T, op>                            \
    {                                                   \
        __device__ T operator()(T *s_ptr, uint tidx)    \
        {                                               \
            Binary<T, op> reduce;                       \
                                                        \
            T val = s_ptr[tidx];                        \
                                                        \
            for (int n = 16; n >= 1; n >>= 1) {         \
                val = reduce(val, __shfl_down(val, n)); \
            }                                           \
            return val;                                 \
        }                                               \
    };                                                  \

    WARP_REDUCE(float)
    WARP_REDUCE(int)
    WARP_REDUCE(uchar) // upcasted to int
    WARP_REDUCE(char)  // upcasted to int
#endif

    template<typename Ti, typename To, af_op_t op, uint DIMX>
    __global__
    static void reduce_first_kernel(Param<To> out,
                                    CParam<Ti>  in,
                                    uint blocks_x, uint blocks_y, uint repeat,
                                    bool change_nan, To nanval)
    {
        const uint tidx = threadIdx.x;
        const uint tidy = threadIdx.y;
        const uint tid  = tidy * blockDim.x + tidx;

        const uint zid = blockIdx.x / blocks_x;
        const uint wid = blockIdx.y / blocks_y;
        const uint blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const uint blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const uint xid = blockIdx_x * blockDim.x * repeat + tidx;
        const uint yid = blockIdx_y * blockDim.y + tidy;

        const Ti *iptr = in.ptr;
        To *optr = out.ptr;

        iptr += wid *  in.strides[3] + zid *  in.strides[2] + yid *  in.strides[1];
        optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];

        if (yid >= in.dims[1] ||
            zid >= in.dims[2] ||
            wid >= in.dims[3]) return;

        Transform<Ti, To, op> transform;
        Binary<To, op> reduce;

        __shared__ To s_val[THREADS_PER_BLOCK];

        To out_val = reduce.init();
        int lim = min((int)(xid + repeat * DIMX), in.dims[0]);

        for (int id = xid; id < lim; id += DIMX) {
            To in_val = transform(iptr[id]);
            if (change_nan) in_val = !IS_NAN(in_val) ? in_val : nanval;
            out_val = reduce(in_val, out_val);
        }

        s_val[tid] = out_val;
        __syncthreads();
        To *s_ptr = s_val + tidy * DIMX;

        if (DIMX == 256) {
            if (tidx < 128) s_ptr[tidx] = reduce(s_ptr[tidx], s_ptr[tidx + 128]);
            __syncthreads();
        }

        if (DIMX >= 128) {
            if (tidx <  64) s_ptr[tidx] = reduce(s_ptr[tidx], s_ptr[tidx +  64]);
            __syncthreads();
        }

        if (DIMX >=  64) {
            if (tidx <  32) s_ptr[tidx] = reduce(s_ptr[tidx], s_ptr[tidx +  32]);
            __syncthreads();
        }

        out_val = WarpReduce<To, op>()(s_ptr, tidx);

        if (tidx == 0) {
            optr[blockIdx_x] = out_val;
        }
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce_first_launcher(Param<To> out, CParam<Ti> in,
                               const uint blocks_x, const uint blocks_y, const uint threads_x,
                               bool change_nan, double nanval)
    {

        dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
        dim3 blocks(blocks_x * in.dims[2],
                    blocks_y * in.dims[3]);

        uint repeat = divup(in.dims[0], (blocks_x * threads_x));

        switch (threads_x) {
        case 32:
            CUDA_LAUNCH((reduce_first_kernel<Ti, To, op,  32>), blocks, threads,
                out, in, blocks_x, blocks_y, repeat, change_nan, scalar<To>(nanval)); break;
        case 64:
            CUDA_LAUNCH((reduce_first_kernel<Ti, To, op,  64>), blocks, threads,
                out, in, blocks_x, blocks_y, repeat, change_nan, scalar<To>(nanval)); break;
        case 128:
            CUDA_LAUNCH((reduce_first_kernel<Ti, To, op,  128>), blocks, threads,
                out, in, blocks_x, blocks_y, repeat, change_nan, scalar<To>(nanval)); break;
        case 256:
            CUDA_LAUNCH((reduce_first_kernel<Ti, To, op,  256>), blocks, threads,
                out, in, blocks_x, blocks_y, repeat, change_nan, scalar<To>(nanval)); break;
        }

        POST_LAUNCH_CHECK();
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce_first(Param<To> out, CParam<Ti> in, bool change_nan, double nanval)
    {
        uint threads_x = nextpow2(std::max(32u, (uint)in.dims[0]));
        threads_x = std::min(threads_x, THREADS_PER_BLOCK);
        uint threads_y = THREADS_PER_BLOCK / threads_x;

        uint blocks_x = divup(in.dims[0], threads_x * REPEAT);
        uint blocks_y = divup(in.dims[1], threads_y);

        Param<To> tmp = out;
        if (blocks_x > 1) {
            tmp.ptr = memAlloc<To>(blocks_x *
                                   in.dims[1] *
                                   in.dims[2] *
                                   in.dims[3]);

            tmp.dims[0] = blocks_x;
            for (int k = 1; k < 4; k++) tmp.strides[k] *= blocks_x;
        }

        reduce_first_launcher<Ti, To, op>(tmp, in, blocks_x, blocks_y, threads_x, change_nan, nanval);

        if (blocks_x > 1) {

            //FIXME: Is there an alternative to the if condition ?
            if (op == af_notzero_t) {
                reduce_first_launcher<To, To, af_add_t>(out, tmp, 1, blocks_y, threads_x,
                                                        change_nan, nanval);
            } else {
                reduce_first_launcher<To, To,       op>(out, tmp, 1, blocks_y, threads_x,
                                                        change_nan, nanval);
            }

            memFree(tmp.ptr);
        }
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce(Param<To> out, CParam<Ti> in, int dim, bool change_nan, double nanval)
    {
        switch (dim) {
        case 0: return reduce_first<Ti, To, op   >(out, in, change_nan, nanval);
        case 1: return reduce_dim  <Ti, To, op, 1>(out, in, change_nan, nanval);
        case 2: return reduce_dim  <Ti, To, op, 2>(out, in, change_nan, nanval);
        case 3: return reduce_dim  <Ti, To, op, 3>(out, in, change_nan, nanval);
        }
    }

    template<typename Ti, typename To, af_op_t op>
    To reduce_all(CParam<Ti> in, bool change_nan, double nanval)
    {
        int in_elements = in.dims[0] * in.dims[1] * in.dims[2] * in.dims[3];

        // FIXME: Use better heuristics to get to the optimum number
        if (in_elements > 4096) {

            bool is_linear = (in.strides[0] == 1);
            for (int k = 1; k < 4; k++) {
                is_linear &= (in.strides[k] == (in.strides[k - 1] * in.dims[k - 1]));
            }

            if (is_linear) {
                in.dims[0] = in_elements;
                for (int k = 1; k < 4; k++) {
                    in.dims[k] = 1;
                    in.strides[k] = in_elements;
                }
            }

            uint threads_x = nextpow2(std::max(32u, (uint)in.dims[0]));
            threads_x = std::min(threads_x, THREADS_PER_BLOCK);
            uint threads_y = THREADS_PER_BLOCK / threads_x;

            Param<To> tmp;

            uint blocks_x = divup(in.dims[0], threads_x * REPEAT);
            uint blocks_y = divup(in.dims[1], threads_y);

            tmp.dims[0] = blocks_x;
            tmp.strides[0] = 1;

            for (int k = 1; k < 4; k++) {
                tmp.dims[k] = in.dims[k];
                tmp.strides[k] = tmp.dims[k - 1] * tmp.strides[k - 1];
            }

            int tmp_elements = tmp.strides[3] * tmp.dims[3];

            tmp.ptr = memAlloc<To>(tmp_elements);
            reduce_first_launcher<Ti, To, op>(tmp, in, blocks_x, blocks_y, threads_x,
                                              change_nan, nanval);

            scoped_ptr<To> h_ptr(new To[tmp_elements]);
            To* h_ptr_raw = h_ptr.get();

            CUDA_CHECK(cudaMemcpy(h_ptr_raw, tmp.ptr, tmp_elements * sizeof(To), cudaMemcpyDeviceToHost));
            memFree(tmp.ptr);

            Binary<To, op> reduce;
            To out = reduce.init();
            for (int i = 0; i < tmp_elements; i++) {
                out = reduce(out, h_ptr_raw[i]);
            }

            return out;

        } else {

            scoped_ptr<Ti> h_ptr(new Ti[in_elements]);
            Ti* h_ptr_raw = h_ptr.get();
            CUDA_CHECK(cudaMemcpy(h_ptr_raw, in.ptr, in_elements * sizeof(Ti), cudaMemcpyDeviceToHost));

            Transform<Ti, To, op> transform;
            Binary<To, op> reduce;
            To out = reduce.init();
            To nanval_to = scalar<To>(nanval);

            for (int i = 0; i < in_elements; i++) {
                To in_val = transform(h_ptr_raw[i]);
                if (change_nan) in_val = !IS_NAN(in_val) ? in_val : nanval_to;
                out = reduce(out, in_val);
            }

            return out;
        }
    }

}
}

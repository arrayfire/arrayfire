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
                                  uint blocks_x, uint blocks_y, uint offset_dim)
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
                             const uint threads_y, const uint blocks_dim[4])
    {
        dim3 threads(THREADS_X, threads_y);

        dim3 blocks(blocks_dim[0] * blocks_dim[2],
                    blocks_dim[1] * blocks_dim[3]);

        switch (threads_y) {
        case 8:
            (reduce_dim_kernel<Ti, To, op, dim, 8>)<<<blocks, threads>>>(
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim]); break;
        case 4:
            (reduce_dim_kernel<Ti, To, op, dim, 4>)<<<blocks, threads>>>(
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim]); break;
        case 2:
            (reduce_dim_kernel<Ti, To, op, dim, 2>)<<<blocks, threads>>>(
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim]); break;
        case 1:
            (reduce_dim_kernel<Ti, To, op, dim, 1>)<<<blocks, threads>>>(
                out, in, blocks_dim[0], blocks_dim[1], blocks_dim[dim]); break;
        }

        POST_LAUNCH_CHECK();
    }

    template<typename Ti, typename To, af_op_t op, int dim>
    void reduce_dim(Param<To> out,  CParam<Ti> in)
    {
        uint threads_y = std::min(THREADS_Y, nextpow2(in.dims[dim]));
        uint threads_x = THREADS_X;

        uint blocks_dim[] = {divup(in.dims[0], threads_x),
                             in.dims[1], in.dims[2], in.dims[3]};

        blocks_dim[dim] = divup(in.dims[dim], threads_y * REPEAT);

        Param<To> tmp = out;

        if (blocks_dim[dim] > 1) {
            dim_type tmp_elements = 1;
            tmp.dims[dim] = blocks_dim[dim];

            for (int k = 0; k < 4; k++) tmp_elements *= tmp.dims[k];
            tmp.ptr = memAlloc<To>(tmp_elements);

            for (int k = dim + 1; k < 4; k++) tmp.strides[k] *= blocks_dim[dim];
        }

        reduce_dim_launcher<Ti, To, op, dim>(tmp, in, threads_y, blocks_dim);

        if (blocks_dim[dim] > 1) {
            blocks_dim[dim] = 1;

            if (op == af_notzero_t) {
                reduce_dim_launcher<To, To, af_add_t, dim>(out, tmp, threads_y, blocks_dim);
            } else {
                reduce_dim_launcher<To, To,       op, dim>(out, tmp, threads_y, blocks_dim);
            }

            memFree(tmp.ptr);
        }

    }


    template<typename To>
    __inline__ __device__ void assign_vol(volatile To *s_ptr_vol, To &tmp)
    {
        *s_ptr_vol = tmp;
    }

    template<> __inline__ __device__
    void assign_vol<cfloat>(volatile cfloat *s_ptr_vol, cfloat &tmp)
    {
        s_ptr_vol->x = tmp.x;
        s_ptr_vol->y = tmp.y;
    }

    template<> __inline__ __device__
    void assign_vol<cdouble>(volatile cdouble *s_ptr_vol, cdouble &tmp)
    {
        s_ptr_vol->x = tmp.x;
        s_ptr_vol->y = tmp.y;
    }

    template<typename To>
    __inline__ __device__ void assign_vol(To &dst, volatile To *s_ptr_vol)
    {
        dst = *s_ptr_vol;
    }

    template<> __inline__ __device__
    void assign_vol<cfloat>(cfloat &dst, volatile cfloat *s_ptr_vol)
    {
        dst.x = s_ptr_vol->x;
        dst.y = s_ptr_vol->y;
    }

    template<> __inline__ __device__
    void assign_vol<cdouble>(cdouble &dst, volatile cdouble *s_ptr_vol)
    {
        dst.x = s_ptr_vol->x;
        dst.y = s_ptr_vol->y;
    }

    template<typename To, af_op_t op>
    __device__ void warp_reduce(To *s_ptr, uint tidx)
    {
        Binary<To, op> reduce;
        volatile To *s_ptr_vol = s_ptr + tidx;
        To tmp = *s_ptr;

#pragma unroll
        for (int n = 16; n >= 1; n >>= 1) {
            if (tidx < n) {
                To val1, val2;
                assign_vol(val1, s_ptr_vol);
                assign_vol(val2, s_ptr_vol + n);
                tmp = reduce(val1, val2);
                assign_vol(s_ptr_vol, tmp);
                __syncthreads();
            }
        }
    }


    template<typename Ti, typename To, af_op_t op, uint DIMX>
    __global__
    static void reduce_first_kernel(Param<To> out,
                                    CParam<Ti>  in,
                                    uint blocks_x, uint blocks_y, uint repeat)
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

        warp_reduce<To, op>(s_ptr, tidx);
        if (tidx == 0) {
            optr[blockIdx_x] = s_ptr[0];
        }
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce_first_launcher(Param<To> out, CParam<Ti> in,
                               const uint blocks_x, const uint blocks_y, const uint threads_x)
    {

        dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
        dim3 blocks(blocks_x * in.dims[2],
                    blocks_y * in.dims[3]);

        uint repeat = divup(in.dims[0], (blocks_x * threads_x));

        switch (threads_x) {
        case 32:
            (reduce_first_kernel<Ti, To, op,  32>)<<<blocks, threads>>>(
                out, in, blocks_x, blocks_y, repeat); break;
        case 64:
            (reduce_first_kernel<Ti, To, op,  64>)<<<blocks, threads>>>(
                out, in, blocks_x, blocks_y, repeat); break;
        case 128:
            (reduce_first_kernel<Ti, To, op,  128>)<<<blocks, threads>>>(
                out, in, blocks_x, blocks_y, repeat); break;
        case 256:
            (reduce_first_kernel<Ti, To, op,  256>)<<<blocks, threads>>>(
                out, in, blocks_x, blocks_y, repeat); break;
        }

        POST_LAUNCH_CHECK();
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce_first(Param<To> out, CParam<Ti> in)
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

        reduce_first_launcher<Ti, To, op>(tmp, in, blocks_x, blocks_y, threads_x);

        if (blocks_x > 1) {

            //FIXME: Is there an alternative to the if condition ?
            if (op == af_notzero_t) {
                reduce_first_launcher<To, To, af_add_t>(out, tmp, 1, blocks_y, threads_x);
            } else {
                reduce_first_launcher<To, To,       op>(out, tmp, 1, blocks_y, threads_x);
            }

            memFree(tmp.ptr);
        }
    }

    template<typename Ti, typename To, af_op_t op>
    void reduce(Param<To> out, CParam<Ti> in, dim_type dim)
    {
        switch (dim) {
        case 0: return reduce_first<Ti, To, op   >(out, in);
        case 1: return reduce_dim  <Ti, To, op, 1>(out, in);
        case 2: return reduce_dim  <Ti, To, op, 2>(out, in);
        case 3: return reduce_dim  <Ti, To, op, 3>(out, in);
        }
    }

    template<typename Ti, typename To, af_op_t op>
    To reduce_all(CParam<Ti> in)
    {
        dim_type in_elements = in.strides[3] * in.dims[3];

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

            dim_type tmp_elements = tmp.strides[3] * tmp.dims[3];

            tmp.ptr = memAlloc<To>(tmp_elements);
            reduce_first_launcher<Ti, To, op>(tmp, in, blocks_x, blocks_y, threads_x);

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

            for (int i = 0; i < in_elements; i++) {
                out = reduce(out, transform(h_ptr_raw[i]));
            }

            return out;
        }
    }

}
}

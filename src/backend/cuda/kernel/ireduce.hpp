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

    template<typename T> __host__ __device__ double cabs(const T in) { return (double)in; }
    static double __host__ __device__ cabs(const char in) { return (double)(in > 0); }
    static double __host__ __device__ cabs(const cfloat &in) { return (double)abs(in); }
    static double __host__ __device__ cabs(const cdouble &in) { return (double)abs(in); }

    template<af_op_t op, typename T>
    struct MinMaxOp
    {
        T m_val;
        uint m_idx;
        __host__ __device__ MinMaxOp(T val, uint idx) :
            m_val(val), m_idx(idx)
        {
        }

        __host__ __device__ void operator()(T val, uint idx)
        {
            if (cabs(val) < cabs(m_val) ||
                (cabs(val) == cabs(m_val) &&
                 idx > m_idx)) {
                m_val = val;
                m_idx = idx;
            }
        }
    };

    template<typename T>
    struct MinMaxOp<af_max_t, T>
    {
        T m_val;
        uint m_idx;
        __host__ __device__ MinMaxOp(T val, uint idx) :
            m_val(val), m_idx(idx)
        {
        }

        __host__ __device__ void operator()(T val, uint idx)
        {
            if (cabs(val) > cabs(m_val) ||
                (cabs(val) == cabs(m_val) &&
                 idx <= m_idx)) {
                m_val = val;
                m_idx = idx;
            }
        }
    };

    template<typename T, af_op_t op, uint dim, bool is_first, uint DIMY>
    __global__
    static void ireduce_dim_kernel(Param<T> out, uint *olptr,
                                  CParam <T> in, const uint *ilptr,
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

        const T *iptr = in.ptr;
        T *optr = out.ptr;

        // There is only one element per block for out
        // There are blockDim.y elements per block for in
        // Hence increment ids[dim] just after offseting out and before offsetting in
        optr += ids[3] * out.strides[3] + ids[2] * out.strides[2] + ids[1] * out.strides[1] + ids[0];
        olptr += ids[3] * out.strides[3] + ids[2] * out.strides[2] + ids[1] * out.strides[1] + ids[0];
        const uint blockIdx_dim = ids[dim];

        ids[dim] = ids[dim] * blockDim.y + tidy;
        iptr  += ids[3] * in.strides[3] + ids[2] * in.strides[2] + ids[1] * in.strides[1] + ids[0];
        if (!is_first) ilptr  += ids[3] * in.strides[3] + ids[2] * in.strides[2] + ids[1] * in.strides[1] + ids[0];
        const uint id_dim_in = ids[dim];

        const uint istride_dim = in.strides[dim];

        bool is_valid =
            (ids[0] < in.dims[0]) &&
            (ids[1] < in.dims[1]) &&
            (ids[2] < in.dims[2]) &&
            (ids[3] < in.dims[3]);

        Binary<T, op> ireduce;

        T val = ireduce.init();
        uint idx = id_dim_in;

        if (is_valid && id_dim_in < in.dims[dim]) {
            val = *iptr;
            if (!is_first) idx = *ilptr;
        }

        MinMaxOp<op, T> Op(val, idx);

        const uint id_dim_in_start = id_dim_in + offset_dim * blockDim.y;

        __shared__ T s_val[THREADS_X * DIMY];
        __shared__ uint s_idx[THREADS_X * DIMY];

        for (int id = id_dim_in_start;
             is_valid && (id < in.dims[dim]);
             id += offset_dim * blockDim.y) {

            iptr = iptr + offset_dim * blockDim.y * istride_dim;
            if (!is_first) {
                ilptr = ilptr + offset_dim * blockDim.y * istride_dim;
                Op(*iptr, *ilptr);
            } else {
                Op(*iptr, id);
            }
        }

        s_val[tid] = Op.m_val;
        s_idx[tid] = Op.m_idx;

        T *s_vptr = s_val + tid;
        uint *s_iptr = s_idx + tid;
        __syncthreads();

        if (DIMY == 8) {
            if (tidy < 4) {
                Op(s_vptr[THREADS_X * 4], s_iptr[THREADS_X * 4]);
                *s_vptr = Op.m_val;
                *s_iptr = Op.m_idx;
            }
            __syncthreads();
        }

        if (DIMY >= 4) {
            if (tidy < 2) {
                Op(s_vptr[THREADS_X * 2], s_iptr[THREADS_X * 2]);
                *s_vptr = Op.m_val;
                *s_iptr = Op.m_idx;
            }
            __syncthreads();
        }

        if (DIMY >= 2) {
            if (tidy < 1) {
                Op(s_vptr[THREADS_X * 1], s_iptr[THREADS_X * 1]);
                *s_vptr = Op.m_val;
                *s_iptr = Op.m_idx;
            }
            __syncthreads();
        }

        if (tidy == 0 && is_valid &&
            (blockIdx_dim < out.dims[dim])) {
            *optr = *s_vptr;
            *olptr = *s_iptr;
        }

    }

    template<typename T, af_op_t op, int dim, bool is_first>
    void ireduce_dim_launcher(Param<T> out, uint *olptr,
                             CParam<T> in, const uint *ilptr,
                             const uint threads_y, const uint blocks_dim[4])
    {
        dim3 threads(THREADS_X, threads_y);

        dim3 blocks(blocks_dim[0] * blocks_dim[2],
                    blocks_dim[1] * blocks_dim[3]);

        switch (threads_y) {
        case 8:
            (ireduce_dim_kernel<T, op, dim, is_first, 8>)<<<blocks, threads>>>(
                out, olptr, in, ilptr, blocks_dim[0], blocks_dim[1], blocks_dim[dim]); break;
        case 4:
            (ireduce_dim_kernel<T, op, dim, is_first, 4>)<<<blocks, threads>>>(
                out, olptr, in, ilptr, blocks_dim[0], blocks_dim[1], blocks_dim[dim]); break;
        case 2:
            (ireduce_dim_kernel<T, op, dim, is_first, 2>)<<<blocks, threads>>>(
                out, olptr, in, ilptr, blocks_dim[0], blocks_dim[1], blocks_dim[dim]); break;
        case 1:
            (ireduce_dim_kernel<T, op, dim, is_first, 1>)<<<blocks, threads>>>(
                out, olptr, in, ilptr, blocks_dim[0], blocks_dim[1], blocks_dim[dim]); break;
        }

        POST_LAUNCH_CHECK();
    }

    template<typename T, af_op_t op, int dim>
    void ireduce_dim(Param<T> out,  uint *olptr, CParam<T> in)
    {
        uint threads_y = std::min(THREADS_Y, nextpow2(in.dims[dim]));
        uint threads_x = THREADS_X;

        uint blocks_dim[] = {divup(in.dims[0], threads_x),
                             in.dims[1], in.dims[2], in.dims[3]};

        blocks_dim[dim] = divup(in.dims[dim], threads_y * REPEAT);

        Param<T> tmp = out;
        uint *tlptr = olptr;

        if (blocks_dim[dim] > 1) {
            int tmp_elements = 1;
            tmp.dims[dim] = blocks_dim[dim];

            for (int k = 0; k < 4; k++) tmp_elements *= tmp.dims[k];
            tmp.ptr = memAlloc<T>(tmp_elements);
            tlptr = memAlloc<uint>(tmp_elements);

            for (int k = dim + 1; k < 4; k++) tmp.strides[k] *= blocks_dim[dim];
        }

        ireduce_dim_launcher<T, op, dim, true>(tmp, tlptr, in, NULL, threads_y, blocks_dim);

        if (blocks_dim[dim] > 1) {
            blocks_dim[dim] = 1;

            ireduce_dim_launcher<T, op, dim, false>(out, olptr, tmp, tlptr,
                                                    threads_y, blocks_dim);

            memFree(tmp.ptr);
            memFree(tlptr);
        }

    }

    template<typename T, af_op_t op>
    __device__ void warp_reduce(T *s_ptr, uint *s_idx, uint tidx)
    {
        MinMaxOp<op, T> Op(s_ptr[tidx], s_idx[tidx]);
#pragma unroll
        for (int n = 16; n >= 1; n >>= 1) {
            if (tidx < n) {
                Op(s_ptr[tidx + n], s_idx[tidx + n]);
                s_ptr[tidx] = Op.m_val;
                s_idx[tidx] = Op.m_idx;
            }
            __syncthreads();
        }
    }


    template<typename T, af_op_t op, bool is_first, uint DIMX>
    __global__
    static void ireduce_first_kernel(Param<T> out, uint *olptr,
                                    CParam<T>  in, const uint *ilptr,
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

        const T *iptr = in.ptr;
        T *optr = out.ptr;

        iptr += wid *  in.strides[3] + zid *  in.strides[2] + yid *  in.strides[1];
        optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];

        if (!is_first) ilptr += wid *  in.strides[3] + zid *  in.strides[2] + yid *  in.strides[1];
        olptr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];

        if (yid >= in.dims[1] ||
            zid >= in.dims[2] ||
            wid >= in.dims[3]) return;

        int lim = min((int)(xid + repeat * DIMX), in.dims[0]);

        Binary<T, op> ireduce;

        T val = ireduce.init();
        uint idx = xid;

        if (xid < lim) {
            val = iptr[xid];
            if (!is_first) idx = ilptr[xid];
        }

        MinMaxOp<op, T> Op(val, idx);

        __shared__ T s_val[THREADS_PER_BLOCK];
        __shared__ uint s_idx[THREADS_PER_BLOCK];


        for (int id = xid + DIMX; id < lim; id += DIMX) {
            Op(iptr[id], (!is_first) ? ilptr[id] : id);
        }

        s_val[tid] = Op.m_val;
        s_idx[tid] = Op.m_idx;
        __syncthreads();

        T *s_vptr = s_val + tidy * DIMX;
        uint *s_iptr = s_idx + tidy * DIMX;

        if (DIMX == 256) {
            if (tidx < 128) {
                Op(s_vptr[tidx + 128], s_iptr[tidx + 128]);
                s_vptr[tidx] = Op.m_val;
                s_iptr[tidx] = Op.m_idx;
            }
            __syncthreads();
        }

        if (DIMX >= 128) {
            if (tidx <  64) {
                Op(s_vptr[tidx +  64], s_iptr[tidx +  64]);
                s_vptr[tidx] = Op.m_val;
                s_iptr[tidx] = Op.m_idx;
            }
            __syncthreads();
        }

        if (DIMX >=  64) {
            if (tidx <  32) {
                Op(s_vptr[tidx +  32], s_iptr[tidx +  32]);
                s_vptr[tidx] = Op.m_val;
                s_iptr[tidx] = Op.m_idx;
            }
            __syncthreads();
        }

        warp_reduce<T, op>(s_vptr, s_iptr, tidx);

        if (tidx == 0) {
            optr[blockIdx_x] = s_vptr[0];
            olptr[blockIdx_x] = s_iptr[0];
        }
    }

    template<typename T, af_op_t op, bool is_first>
    void ireduce_first_launcher(Param<T> out, uint *olptr, CParam<T> in, const uint *ilptr,
                               const uint blocks_x, const uint blocks_y, const uint threads_x)
    {

        dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
        dim3 blocks(blocks_x * in.dims[2],
                    blocks_y * in.dims[3]);

        uint repeat = divup(in.dims[0], (blocks_x * threads_x));

        switch (threads_x) {
        case 32:
            (ireduce_first_kernel<T, op, is_first,  32>)<<<blocks, threads>>>(
                out, olptr, in, ilptr, blocks_x, blocks_y, repeat); break;
        case 64:
            (ireduce_first_kernel<T, op, is_first,  64>)<<<blocks, threads>>>(
                out, olptr, in, ilptr, blocks_x, blocks_y, repeat); break;
        case 128:
            (ireduce_first_kernel<T, op, is_first,  128>)<<<blocks, threads>>>(
                out, olptr, in, ilptr, blocks_x, blocks_y, repeat); break;
        case 256:
            (ireduce_first_kernel<T, op, is_first,  256>)<<<blocks, threads>>>(
                out, olptr, in, ilptr, blocks_x, blocks_y, repeat); break;
        }

        POST_LAUNCH_CHECK();
    }

    template<typename T, af_op_t op>
    void ireduce_first(Param<T> out, uint *olptr, CParam<T> in)
    {
        uint threads_x = nextpow2(std::max(32u, (uint)in.dims[0]));
        threads_x = std::min(threads_x, THREADS_PER_BLOCK);
        uint threads_y = THREADS_PER_BLOCK / threads_x;

        uint blocks_x = divup(in.dims[0], threads_x * REPEAT);
        uint blocks_y = divup(in.dims[1], threads_y);

        Param<T> tmp = out;
        uint *tlptr = olptr;
        if (blocks_x > 1) {
            tmp.ptr = memAlloc<T>(blocks_x *
                                  in.dims[1] *
                                  in.dims[2] *
                                  in.dims[3]);

            tlptr = memAlloc<uint>(blocks_x *
                                   in.dims[1] *
                                   in.dims[2] *
                                   in.dims[3]);

            tmp.dims[0] = blocks_x;
            for (int k = 1; k < 4; k++) tmp.strides[k] *= blocks_x;
        }

        ireduce_first_launcher<T, op, true>(tmp, tlptr, in, NULL, blocks_x, blocks_y, threads_x);

        if (blocks_x > 1) {
            ireduce_first_launcher<T, op, false>(out, olptr, tmp, tlptr, 1, blocks_y, threads_x);

            memFree(tmp.ptr);
            memFree(tlptr);
        }
    }

    template<typename T, af_op_t op>
    void ireduce(Param<T> out, uint *olptr, CParam<T> in, int dim)
    {
        switch (dim) {
        case 0: return ireduce_first<T, op   >(out, olptr, in);
        case 1: return ireduce_dim  <T, op, 1>(out, olptr, in);
        case 2: return ireduce_dim  <T, op, 2>(out, olptr, in);
        case 3: return ireduce_dim  <T, op, 3>(out, olptr, in);
        }
    }

    template<typename T, af_op_t op>
    T ireduce_all(uint *idx, CParam<T> in)
    {
        int in_elements = in.strides[3] * in.dims[3];

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

            Param<T> tmp;
            uint *tlptr;

            uint blocks_x = divup(in.dims[0], threads_x * REPEAT);
            uint blocks_y = divup(in.dims[1], threads_y);

            tmp.dims[0] = blocks_x;
            tmp.strides[0] = 1;

            for (int k = 1; k < 4; k++) {
                tmp.dims[k] = in.dims[k];
                tmp.strides[k] = tmp.dims[k - 1] * tmp.strides[k - 1];
            }

            int tmp_elements = tmp.strides[3] * tmp.dims[3];

            //TODO: Use scoped_ptr
            tmp.ptr = memAlloc<T>(tmp_elements);
            tlptr = memAlloc<uint>(tmp_elements);
            ireduce_first_launcher<T, op, true>(tmp, tlptr, in, NULL, blocks_x, blocks_y, threads_x);

            scoped_ptr<T>       h_ptr(new T[tmp_elements]);
            scoped_ptr<uint>    h_lptr(new uint[tmp_elements]);
            T*      h_ptr_raw = h_ptr.get();
            uint*   h_lptr_raw = h_lptr.get();

            CUDA_CHECK(cudaMemcpy(h_ptr_raw, tmp.ptr, tmp_elements * sizeof(T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_lptr_raw, tlptr, tmp_elements * sizeof(uint), cudaMemcpyDeviceToHost));
            memFree(tmp.ptr);
            memFree(tlptr);

            MinMaxOp<op, T> Op(h_ptr_raw[0], h_lptr_raw[0]);

            for (int i = 1; i < tmp_elements; i++) {
                Op(h_ptr_raw[i], h_lptr_raw[i]);
            }

            *idx = Op.m_idx;
            return Op.m_val;
        } else {

            scoped_ptr<T> h_ptr(new T[in_elements]);
            T* h_ptr_raw = h_ptr.get();
            CUDA_CHECK(cudaMemcpy(h_ptr_raw, in.ptr, in_elements * sizeof(T), cudaMemcpyDeviceToHost));

            MinMaxOp<op, T> Op(h_ptr_raw[0], 0);
            for (int i = 1; i < in_elements; i++) {
                Op(h_ptr_raw[i], i);
            }

            *idx = Op.m_idx;
            return Op.m_val;
        }
    }

}
}

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <ops.hpp>
#include <backend.hpp>
#include <Param.hpp>
#include <common/dispatch.hpp>
#include <math.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <memory.hpp>
#include "config.hpp"

namespace cuda
{
namespace kernel
{
    template<typename Tk>
    __device__
    inline static char calculate_head_flags(const Tk *kptr, int id, int previd)
    {
        return (id == 0)? 1 : (kptr[id] != kptr[previd]);
    }

    template<typename Ti, typename Tk, typename To, af_op_t op, uint DIMX>
    __global__
    static void scan_nonfinal_kernel(Param<To> out,
                                     Param<To> tmp,
                                     Param<char> tflg,
                                     Param<int> tlid,
                                     CParam<Ti> in,
                                     CParam<Tk> key,
                                     uint blocks_x,
                                     uint blocks_y,
                                     uint lim,
                                     bool inclusive_scan)
    {
        Transform<Ti, To, op> transform;
        Binary<To, op> binop;
        const To init = binop.init();
        To val = init;

        const int istride = in.strides[0];
        const int DIMY = THREADS_PER_BLOCK / DIMX;
        const int SHARED_MEM_SIZE = (2 * DIMX + 1) * (DIMY);
        __shared__ char s_flg[SHARED_MEM_SIZE];
        __shared__ To s_val[SHARED_MEM_SIZE];
        __shared__ char s_ftmp[DIMY];
        __shared__ To s_tmp[DIMY];
        __shared__ int boundaryid[DIMY];

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
        const int zid = blockIdx.x / blocks_x;
        const int wid = blockIdx.y / blocks_y;
        const int blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const int blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const int xid = blockIdx_x * blockDim.x * lim + tidx;
        const int yid = blockIdx_y * blockDim.y + tidy;
        bool cond_yzw = (yid < out.dims[1]) && (zid < out.dims[2]) && (wid < out.dims[3]);
        if (!cond_yzw) return; // retire warps early

        To *sptr = s_val + tidy * (2 * DIMX + 1);
        char *sfptr = s_flg + tidy * (2 * DIMX + 1);
        int id = xid;

        const bool isLast = (tidx == (DIMX - 1));
        if (isLast) {
            s_tmp[tidy] = init;
            s_ftmp[tidy] = 0;
            boundaryid[tidy] = -1;
        }
        __syncthreads();

        const Ti *iptr = in.ptr;
        const Tk *kptr = key.ptr;
        To *optr = out.ptr;
        To *tptr = tmp.ptr;
        char *tfptr = tflg.ptr;
        int *tiptr = tlid.ptr;
        iptr  += wid *   in.strides[3] + zid *   in.strides[2] + yid *   in.strides[1];
        kptr  += wid *  key.strides[3] + zid *  key.strides[2] + yid *  key.strides[1];
        optr  += wid *  out.strides[3] + zid *  out.strides[2] + yid *  out.strides[1];
        tptr  += wid *  tmp.strides[3] + zid *  tmp.strides[2] + yid *  tmp.strides[1];
        tfptr += wid * tflg.strides[3] + zid * tflg.strides[2] + yid * tflg.strides[1];
        tiptr += wid * tlid.strides[3] + zid * tlid.strides[2] + yid * tlid.strides[1];

        char flag = 0;
        for (int k = 0; k < lim; k++) {
            if (id < out.dims[0]) {
                flag = calculate_head_flags(kptr, id, id - 1);
            } else {
                flag = 0;
            }

            //Load val from global in
            if (inclusive_scan) {
                if (id >= out.dims[0]) {
                    val = init;
                } else {
                    val = transform(iptr[id]);
                }
            } else {
                if ((id == 0) || (id >= out.dims[0]) || flag) {
                    val = init;
                } else {
                    val = transform(iptr[id-istride]);
                }
            }

            //Add partial result from last iteration before scan operation
            if ((tidx == 0) && (flag == 0)) {
                val = binop(val, s_tmp[tidy]);
                flag = s_ftmp[tidy];
            }

            //Write to shared memory
            sptr[tidx] = val;
            sfptr[tidx] = flag;
            __syncthreads();

            //Segmented Scan
            int start = 0;
#pragma unroll
            for (int off = 1; off < DIMX; off *= 2) {
                if (tidx >= off) {
                    val = sfptr[start + tidx]? val : binop(val, sptr[(start - off) + tidx]);
                    flag = sfptr[start + tidx] | sfptr[(start - off) + tidx];
                }
                start = DIMX - start;
                sptr[start + tidx] = val;
                sfptr[start + tidx] = flag;

                __syncthreads();
            }

            //Identify segment boundary
            if (tidx == 0) {
                if ((s_ftmp[tidy] == 0) && (sfptr[tidx] == 1)) {
                    boundaryid[tidy] = id;
                }
            } else {
                if ((sfptr[tidx-1] == 0) && (sfptr[tidx] == 1)) {
                    boundaryid[tidy] = id;
                }
            }
            __syncthreads();

            if (id < out.dims[0]) optr[id] = val;
            if (isLast) {
                s_tmp[tidy] = val;
                s_ftmp[tidy] = flag;
            }
            id += blockDim.x;
            __syncthreads();
        }
        if (isLast) {
            tptr[blockIdx_x] = val;
            tfptr[blockIdx_x] = flag;
            int boundary = boundaryid[tidy];
            tiptr[blockIdx_x] = (boundary == -1)? id : boundary;
        }
    }

    template<typename Ti, typename Tk, typename To, af_op_t op, uint DIMX>
    __global__
    static void scan_final_kernel(Param<To> out,
                                  CParam<Ti> in,
                                  CParam<Tk> key,
                                  uint blocks_x,
                                  uint blocks_y,
                                  uint lim,
                                  bool calculateFlags,
                                  bool inclusive_scan)
    {
        Transform<Ti, To, op> transform;
        Binary<To, op> binop;
        const To init = binop.init();
        To val = init;

        const int istride = in.strides[0];
        const int DIMY = THREADS_PER_BLOCK / DIMX;
        const int SHARED_MEM_SIZE = (2 * DIMX + 1) * (DIMY);
        __shared__ char s_flg[SHARED_MEM_SIZE];
        __shared__ To s_val[SHARED_MEM_SIZE];
        __shared__ char s_ftmp[DIMY];
        __shared__ To s_tmp[DIMY];

        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;
        const int zid = blockIdx.x / blocks_x;
        const int wid = blockIdx.y / blocks_y;
        const int blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const int blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const int xid = blockIdx_x * blockDim.x * lim + tidx;
        const int yid = blockIdx_y * blockDim.y + tidy;
        bool cond_yzw = (yid < out.dims[1]) && (zid < out.dims[2]) && (wid < out.dims[3]);
        if (!cond_yzw) return; // retire warps early

        To *sptr = s_val + tidy * (2 * DIMX + 1);
        char *sfptr = s_flg + tidy * (2 * DIMX + 1);
        int id = xid;

        const bool isLast = (tidx == (DIMX - 1));
        if (isLast) {
            s_tmp[tidy] = init;
            s_ftmp[tidy] = 0;
        }
        __syncthreads();

        const Ti *iptr = in.ptr;
        const Tk *kptr = key.ptr;
        To *optr = out.ptr;
        iptr += wid *  in.strides[3] + zid *  in.strides[2] + yid *  in.strides[1];
        kptr += wid * key.strides[3] + zid * key.strides[2] + yid * key.strides[1];
        optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];

        for (int k = 0; k < lim; k++) {
            char flag = 0;
            if (calculateFlags) {
                if (id < out.dims[0]) {
                    flag = calculate_head_flags(kptr, id, id - key.strides[0]);
                }
            } else {
                flag = kptr[id];
            }

            //Load val from global in
            if (inclusive_scan) {
                if (id >= out.dims[0]) {
                    val = init;
                } else {
                    val = transform(iptr[id]);
                }
            } else {
                if ((id == 0) || (id >= out.dims[0]) || flag) {
                    val = init;
                } else {
                    val = transform(iptr[id-istride]);
                }
            }

            //Add partial result from last iteration before scan operation
            if ((tidx == 0) && (flag == 0)) {
                val = binop(val, s_tmp[tidy]);
                flag = flag | s_ftmp[tidy];
            }

            //Write to shared memory
            sptr[tidx] = val;
            sfptr[tidx] = flag;
            __syncthreads();

            //Segmented Scan
            int start = 0;
#pragma unroll
            for (int off = 1; off < DIMX; off *= 2) {
                if (tidx >= off) {
                    val = sfptr[start + tidx]? val : binop(val, sptr[(start - off) + tidx]);
                    flag = sfptr[start + tidx] | sfptr[(start - off) + tidx];
                }
                start = DIMX - start;
                sptr[start + tidx] = val;
                sfptr[start + tidx] = flag;

                __syncthreads();
            }

            if (id < out.dims[0]) optr[id] = val;
            if (isLast) {
                s_tmp[tidy] = val;
                s_ftmp[tidy] = flag;
            }
            id += blockDim.x;
            __syncthreads();
        }
    }

    template<typename Ti, typename Tk, typename To, af_op_t op>
    static void scan_nonfinal_launcher(Param<To> out,
                                       Param<To> tmp,
                                       Param<char> tflg,
                                       Param<int> tlid,
                                       CParam<Ti> in,
                                       CParam<Tk> key,
                                       const uint blocks_x,
                                       const uint blocks_y,
                                       const uint threads_x,
                                       bool inclusive_scan)
    {

        dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
        dim3 blocks(blocks_x * out.dims[2],
                    blocks_y * out.dims[3]);

        uint lim = divup(out.dims[0], (threads_x * blocks_x));

        switch (threads_x) {
        case 32:
            CUDA_LAUNCH((scan_nonfinal_kernel<Ti, Tk, To, op, 32>),
                        blocks, threads, out, tmp, tflg, tlid, in, key,
                        blocks_x, blocks_y, lim, inclusive_scan); break;
        case 64:
            CUDA_LAUNCH((scan_nonfinal_kernel<Ti, Tk, To, op, 64>),
                        blocks, threads, out, tmp, tflg, tlid, in, key,
                        blocks_x, blocks_y, lim, inclusive_scan); break;
        case 128:
            CUDA_LAUNCH((scan_nonfinal_kernel<Ti, Tk, To, op, 128>),
                        blocks, threads, out, tmp, tflg, tlid, in, key,
                        blocks_x, blocks_y, lim, inclusive_scan); break;
        case 256:
            CUDA_LAUNCH((scan_nonfinal_kernel<Ti, Tk, To, op, 256>),
                        blocks, threads, out, tmp, tflg, tlid, in, key,
                        blocks_x, blocks_y, lim, inclusive_scan); break;
        }

        POST_LAUNCH_CHECK();
    }

    template<typename Ti, typename Tk, typename To, af_op_t op>
    static void scan_final_launcher(Param<To> out,
                                    CParam<Ti> in,
                                    CParam<Tk> key,
                                    const uint blocks_x,
                                    const uint blocks_y,
                                    const uint threads_x,
                                    bool calculateFlags,
                                    bool inclusive_scan)
    {

        dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
        dim3 blocks(blocks_x * out.dims[2],
                    blocks_y * out.dims[3]);

        uint lim = divup(out.dims[0], (threads_x * blocks_x));

        switch (threads_x) {
        case 32:
            CUDA_LAUNCH((scan_final_kernel<Ti, Tk, To, op, 32>),
                        blocks, threads, out, in, key, blocks_x,
                        blocks_y, lim, calculateFlags, inclusive_scan); break;
        case 64:
            CUDA_LAUNCH((scan_final_kernel<Ti, Tk, To, op, 64>),
                        blocks, threads, out, in, key, blocks_x,
                        blocks_y, lim, calculateFlags, inclusive_scan); break;
        case 128:
            CUDA_LAUNCH((scan_final_kernel<Ti, Tk, To, op, 128>),
                        blocks, threads, out, in, key, blocks_x,
                        blocks_y, lim, calculateFlags, inclusive_scan); break;
        case 256:
            CUDA_LAUNCH((scan_final_kernel<Ti, Tk, To, op, 256>),
                        blocks, threads, out, in, key, blocks_x,
                        blocks_y, lim, calculateFlags, inclusive_scan); break;
        }

        POST_LAUNCH_CHECK();
    }

    template<typename To, af_op_t op>
    __global__
    static void bcast_first_kernel(Param<To> out,
                                   Param<To> tmp,
                                   Param<int> tlid,
                                   uint blocks_x,
                                   uint blocks_y,
                                   uint lim)
    {
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int zid = blockIdx.x / blocks_x;
        const int wid = blockIdx.y / blocks_y;
        const int blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const int blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const int xid = blockIdx_x * blockDim.x * lim + tidx;
        const int yid = blockIdx_y * blockDim.y + tidy;

        if (blockIdx_x == 0) return;

        bool cond = (yid < out.dims[1]) && (zid < out.dims[2]) && (wid < out.dims[3]);
        if (!cond) return;

        To *optr = out.ptr;
        const To *tptr = tmp.ptr;
        const int *iptr = tlid.ptr;

        optr += wid *  out.strides[3] + zid *  out.strides[2] + yid *  out.strides[1];
        tptr += wid *  tmp.strides[3] + zid *  tmp.strides[2] + yid *  tmp.strides[1];
        iptr += wid * tlid.strides[3] + zid * tlid.strides[2] + yid * tlid.strides[1];

        Binary<To, op> binop;
        int boundary = iptr[blockIdx_x];
        To accum = tptr[blockIdx_x - 1];

        for (int k = 0, id = xid;
             k < lim && id < boundary;
             k++, id += blockDim.x) {

            optr[id] = binop(accum, optr[id]);
        }
    }

    template<typename To, af_op_t op>
    static void bcast_first_launcher(Param<To> out,
                                     Param<To> tmp,
                                     Param<int> tlid,
                                     const dim_t blocks_x,
                                     const dim_t blocks_y,
                                     const uint threads_x)
    {

        dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
        dim3 blocks(blocks_x * out.dims[2],
                    blocks_y * out.dims[3]);
        uint lim = divup(out.dims[0], (threads_x * blocks_x));
        CUDA_LAUNCH((bcast_first_kernel<To, op>), blocks, threads, out, tmp, tlid, blocks_x, blocks_y, lim);

        POST_LAUNCH_CHECK();
    }

    template<typename Ti, typename Tk, typename To, af_op_t op>
    void scan_first_by_key(Param<To> out, CParam<Ti> in, CParam<Tk> key, bool inclusive_scan)
    {
        uint threads_x = nextpow2(std::max(32u, (uint)out.dims[0]));
        threads_x = std::min(threads_x, THREADS_PER_BLOCK);
        uint threads_y = THREADS_PER_BLOCK / threads_x;

        uint blocks_x = static_cast<uint>(divup(out.dims[0], threads_x * REPEAT));
        uint blocks_y = static_cast<uint>(divup(out.dims[1], threads_y));

        if (blocks_x == 1) {
            scan_final_launcher<Ti, Tk, To, op>(
                out, in, key,
                blocks_x, blocks_y, threads_x,
                true, inclusive_scan);

        } else {

            Param<To> tmp = out;
            Param<char> tmpflg;
            Param<int> tmpid;

            tmp.dims[0] = blocks_x;
            tmpflg.dims[0] = blocks_x;
            tmpid.dims[0] = blocks_x;
            tmp.strides[0] = 1;
            tmpflg.strides[0] = 1;
            tmpid.strides[0] = 1;
            for (int k = 1; k < 4; k++) {
                tmpflg.dims[k] = out.dims[k];
                tmpid.dims[k] = out.dims[k];
                tmp.strides[k] = tmp.strides[k - 1] * tmp.dims[k - 1];
                tmpflg.strides[k] = tmpflg.strides[k - 1] * tmpflg.dims[k - 1];
                tmpid.strides[k] = tmpid.strides[k - 1] * tmpid.dims[k - 1];
            }

            int tmp_elements = tmp.strides[3] * tmp.dims[3];
            tmp.ptr = memAlloc<To>(tmp_elements);
            tmpflg.ptr = memAlloc<char>(tmp_elements);
            tmpid.ptr = memAlloc<int>(tmp_elements);

            scan_nonfinal_launcher<Ti, Tk, To, op>(
                out, tmp, tmpflg, tmpid, in, key,
                blocks_x, blocks_y, threads_x,
                inclusive_scan);

            scan_final_launcher<To, char, To, op>(
                tmp, tmp, tmpflg,
                1, blocks_y, threads_x,
                false, true);

            bcast_first_launcher<To, op>(out, tmp, tmpid, blocks_x, blocks_y, threads_x);

            memFree(tmp.ptr);
            memFree(tmpflg.ptr);
            memFree(tmpid.ptr);
        }
    }
}

#define INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, Ti, Tk, To)\
    template void scan_first_by_key<Ti, Tk, To, ROp>(Param<To> out, CParam<Ti> in, CParam<Tk> key, bool inclusive_scan); \

#define INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, Tk)        \
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, float  , Tk, float  )\
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, double , Tk, double )\
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, cfloat , Tk, cfloat )\
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, cdouble, Tk, cdouble)\
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, int    , Tk, int    )\
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, uint   , Tk, uint   )\
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, intl   , Tk, intl   )\
    INSTANTIATE_SCAN_FIRST_BY_KEY(ROp, uintl  , Tk, uintl  )\

#define INSTANTIATE_SCAN_FIRST_BY_KEY_OP(ROp)       \
    INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, int  ) \
    INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, uint ) \
    INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, intl ) \
    INSTANTIATE_SCAN_FIRST_BY_KEY_TYPES(ROp, uintl)
}

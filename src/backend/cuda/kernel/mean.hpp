/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <ops.hpp>
#include <backend.hpp>
#include <copy.hpp>
#include <Param.hpp>
#include <common/dispatch.hpp>
#include <math.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include "config.hpp"
#include <memory.hpp>

#include <memory>
#include <vector>

using std::vector;

namespace cuda
{
namespace kernel
{

    template<typename To, typename Tw>
    __device__ __host__
    void stable_mean(To *lhs, Tw *l_wt, To rhs, Tw r_wt)
    {
        if (((*l_wt) != 0) || (r_wt != 0)) {
            Tw l_scale = (*l_wt);
            (*l_wt) += r_wt;
            l_scale = l_scale/(*l_wt);

            Tw r_scale = r_wt/(*l_wt);
            (*lhs) = (l_scale * (*lhs)) + (r_scale * rhs);
        }
    }

    template<typename Ti, typename Tw, typename To, uint dim, uint DIMY>
    __global__
    static void mean_dim_kernel(Param<To> out, Param<Tw> owt,
                                CParam <Ti> in, CParam<Tw> iwt,
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
        const Tw *iwptr = iwt.ptr;
        To *optr = out.ptr;
        Tw *owptr = owt.ptr;

        int ooffset = ids[3] * out.strides[3] +
                      ids[2] * out.strides[2] +
                      ids[1] * out.strides[1] + ids[0];
        // There is only one element per block for out
        // There are blockDim.y elements per block for in
        // Hence increment ids[dim] just after offseting out and before offsetting in
        optr += ooffset;
        if (owptr != NULL) owptr += ooffset;

        const uint blockIdx_dim = ids[dim];

        ids[dim] = ids[dim] * blockDim.y + tidy;

        int ioffset = ids[3] * in.strides[3] +
                      ids[2] * in.strides[2] +
                      ids[1] * in.strides[1] + ids[0];
        iptr  += ioffset;
        if (iwptr != NULL) iwptr  += ioffset;

        const uint id_dim_in = ids[dim];
        const uint istride_dim = in.strides[dim];

        bool is_valid =
            (ids[0] < in.dims[0]) &&
            (ids[1] < in.dims[1]) &&
            (ids[2] < in.dims[2]) &&
            (ids[3] < in.dims[3]);

        Transform<Ti, To, af_add_t> transform;
        Binary<To, af_add_t> mean_obj;
        Binary<Tw, af_add_t> weight_obj;

        To val = mean_obj.init();
        Tw weight = weight_obj.init();

        if (is_valid && id_dim_in < in.dims[dim]) {
            val = transform(*iptr);
            if (iwptr != NULL) {
                weight = *iwptr;
            } else {
                weight = (Tw)1;
            }
        }

        const uint id_dim_in_start = id_dim_in + offset_dim * blockDim.y;

        __shared__ To s_val[THREADS_X * DIMY];
        __shared__ Tw s_idx[THREADS_X * DIMY];

        for (int id = id_dim_in_start;
             is_valid && (id < in.dims[dim]);
             id += offset_dim * blockDim.y) {

            iptr = iptr + offset_dim * blockDim.y * istride_dim;
            if (iwptr != NULL) {
                iwptr = iwptr + offset_dim * blockDim.y * istride_dim;
                stable_mean(&val, &weight, transform(*iptr), *iwptr);
            } else {
                // Faster version of stable_mean when iwptr is NULL
                val = val + (transform(*iptr) - val) / (weight + 1);
                weight = weight + 1;
            }
        }

        s_val[tid] = val;
        s_idx[tid] = weight;

        To *s_vptr = s_val + tid;
        Tw *s_iptr = s_idx + tid;
        __syncthreads();

        if (DIMY == 8) {
            if (tidy < 4) {
                stable_mean(s_vptr, s_iptr, s_vptr[THREADS_X * 4], s_iptr[THREADS_X * 4]);
            }
            __syncthreads();
        }

        if (DIMY >= 4) {
            if (tidy < 2) {
                stable_mean(s_vptr, s_iptr, s_vptr[THREADS_X * 2], s_iptr[THREADS_X * 2]);
            }
            __syncthreads();
        }

        if (DIMY >= 2) {
            if (tidy < 1) {
                stable_mean(s_vptr, s_iptr, s_vptr[THREADS_X * 1], s_iptr[THREADS_X * 1]);
            }
            __syncthreads();
        }

        if (tidy == 0 && is_valid &&
            (blockIdx_dim < out.dims[dim])) {
            *optr = *s_vptr;
            if (owptr != NULL) *owptr = *s_iptr;
        }

    }

    template<typename Ti, typename Tw, typename To, int dim>
    void mean_dim_launcher(Param<To> out, Param<Tw> owt,
                           CParam<Ti> in, CParam<Tw> iwt,
                           const uint threads_y, const dim_t blocks_dim[4])
    {
        dim3 threads(THREADS_X, threads_y);

        dim3 blocks(blocks_dim[0] * blocks_dim[2],
                    blocks_dim[1] * blocks_dim[3]);

        switch (threads_y) {
        case 8:
            CUDA_LAUNCH((mean_dim_kernel<Ti, Tw, To, dim, 8>), blocks, threads,
                out, owt, in, iwt, blocks_dim[0], blocks_dim[1], blocks_dim[dim]); break;
        case 4:
            CUDA_LAUNCH((mean_dim_kernel<Ti, Tw, To, dim, 4>), blocks, threads,
                out, owt, in, iwt, blocks_dim[0], blocks_dim[1], blocks_dim[dim]); break;
        case 2:
            CUDA_LAUNCH((mean_dim_kernel<Ti, Tw, To, dim, 2>), blocks, threads,
                out, owt, in, iwt, blocks_dim[0], blocks_dim[1], blocks_dim[dim]); break;
        case 1:
            CUDA_LAUNCH((mean_dim_kernel<Ti, Tw, To, dim, 1>), blocks, threads,
                out, owt, in, iwt, blocks_dim[0], blocks_dim[1], blocks_dim[dim]); break;
        }

        POST_LAUNCH_CHECK();
    }

    template<typename Ti, typename Tw, typename To, int dim>
    void mean_dim(Param<To> out, CParam<Ti> in, CParam<Tw> iwt)
    {
        uint threads_y = std::min(THREADS_Y, nextpow2(in.dims[dim]));
        uint threads_x = THREADS_X;

        dim_t blocks_dim[] = {divup(in.dims[0], threads_x),
                             in.dims[1], in.dims[2], in.dims[3]};

        blocks_dim[dim] = divup(in.dims[dim], threads_y * REPEAT);

        Array<To> tmpOut = *initArray<To>();
        Array<Tw> tmpWt  = *initArray<Tw>();

        if (blocks_dim[dim] > 1) {
          dim4 dims(4, out.dims);
          dims[dim] = blocks_dim[dim];
          tmpOut = createEmptyArray<To>(dims);
          tmpWt  = createEmptyArray<Tw>(dims);
        }
        else {
          tmpOut = createParamArray(out, false);
        }

        mean_dim_launcher<Ti, Tw, To, dim>(tmpOut, tmpWt, in, iwt, threads_y, blocks_dim);

        if (blocks_dim[dim] > 1) {
            blocks_dim[dim] = 1;

            Array<Tw> owt = *initArray<Tw>();
            mean_dim_launcher<To, Tw, To, dim>(out, owt, tmpOut, tmpWt,
                                                    threads_y, blocks_dim);

        }

    }

    template<typename T, typename Tw>
    __device__ void warp_reduce(T *s_ptr, Tw *s_idx, uint tidx)
    {
#pragma unroll
        for (int n = 16; n >= 1; n >>= 1) {
            if (tidx < n) {
                stable_mean(s_ptr + tidx, s_idx + tidx, s_ptr[tidx + n], s_idx[tidx + n]);
            }
            __syncthreads();
        }
    }

    //Calculate mean along the first dimension. If wt is an empty CParam, use
    //weight as 1 and treat it as count. If owt is empty Param, do not write
    //temporary reduced counts/weights to it.
    template<typename Ti, typename Tw, typename To, uint DIMX>
    __global__
    static void mean_first_kernel(Param<To> out, Param<Tw> owt,
                                  CParam<Ti> in, CParam<Tw> iwt,
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
        const Tw *iwptr = iwt.ptr;
        To *optr = out.ptr;
        Tw *owptr = owt.ptr;

        iptr += wid *  in.strides[3] + zid *  in.strides[2] + yid *  in.strides[1];
        if (iwptr != NULL) iwptr += wid *  iwt.strides[3] + zid *  iwt.strides[2] + yid *  iwt.strides[1];
        optr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];
        if (owptr != NULL) owptr += wid * out.strides[3] + zid * out.strides[2] + yid * out.strides[1];

        if (yid >= in.dims[1] ||
            zid >= in.dims[2] ||
            wid >= in.dims[3]) return;

        int lim = min((int)(xid + repeat * DIMX), in.dims[0]);

        Transform<Ti, To, af_add_t> transform;
        Binary<To, af_add_t> mean_obj;
        Binary<Tw, af_add_t> weight_obj;

        To val = mean_obj.init();
        Tw weight = weight_obj.init();

        if (xid < lim) {
            val = transform(iptr[xid]);
            if (iwptr != NULL) {
                weight = iwptr[xid];
            } else {
                weight = (Tw)1;
            }
        }

        __shared__ To s_val[THREADS_PER_BLOCK];
        __shared__ Tw s_idx[THREADS_PER_BLOCK];

        if (iwptr != NULL) {
            for (int id = xid + DIMX; id < lim; id += DIMX) {
                stable_mean(&val, &weight, transform(iptr[id]), iwptr[id]);
            }
        } else {
            for (int id = xid + DIMX; id < lim; id += DIMX) {
                // Faster version of stable_mean when iwptr is NULL
                val = val + (transform(iptr[id]) - val) / (weight + 1);
                weight = weight + 1;
            }
        }

        s_val[tid] = val;
        s_idx[tid] = weight;
        __syncthreads();

        To *s_vptr = s_val + tidy * DIMX;
        Tw *s_iptr = s_idx + tidy * DIMX;

        if (DIMX == 256) {
            if (tidx < 128) {
                stable_mean(s_vptr + tidx, s_iptr + tidx, s_vptr[tidx + 128], s_iptr[tidx + 128]);
            }
            __syncthreads();
        }

        if (DIMX >= 128) {
            if (tidx <  64) {
                stable_mean(s_vptr + tidx, s_iptr + tidx, s_vptr[tidx +  64], s_iptr[tidx +  64]);
            }
            __syncthreads();
        }

        if (DIMX >=  64) {
            if (tidx <  32) {
                stable_mean(s_vptr + tidx, s_iptr + tidx, s_vptr[tidx +  32], s_iptr[tidx +  32]);
            }
            __syncthreads();
        }

        warp_reduce<To, Tw>(s_vptr, s_iptr, tidx);

        if (tidx == 0) {
            optr[blockIdx_x] = s_vptr[0];
            if (owptr != NULL) owptr[blockIdx_x] = s_iptr[0];
        }
    }


    template<typename Ti, typename Tw, typename To>
    void mean_first_launcher(Param<To> out, Param<Tw> owt, CParam<Ti> in, CParam<Tw> iwt,
                             const uint blocks_x, const uint blocks_y, const uint threads_x)
    {

        dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
        dim3 blocks(blocks_x * in.dims[2],
                    blocks_y * in.dims[3]);

        uint repeat = divup(in.dims[0], (blocks_x * threads_x));

        switch (threads_x) {
        case 32:
            CUDA_LAUNCH((mean_first_kernel<Ti, Tw, To, 32>), blocks, threads,
                out, owt, in, iwt, blocks_x, blocks_y, repeat); break;
        case 64:
            CUDA_LAUNCH((mean_first_kernel<Ti, Tw, To, 64>), blocks, threads,
                out, owt, in, iwt, blocks_x, blocks_y, repeat); break;
        case 128:
            CUDA_LAUNCH((mean_first_kernel<Ti, Tw, To, 128>), blocks, threads,
                out, owt, in, iwt, blocks_x, blocks_y, repeat); break;
        case 256:
            CUDA_LAUNCH((mean_first_kernel<Ti, Tw, To, 256>), blocks, threads,
                out, owt, in, iwt, blocks_x, blocks_y, repeat); break;
        }

        POST_LAUNCH_CHECK();
    }

    template<typename Ti, typename Tw, typename To>
    void mean_first(Param<To> out, CParam<Ti> in, CParam<Tw> iwt)
    {
        uint threads_x = nextpow2(std::max(32u, (uint)in.dims[0]));
        threads_x = std::min(threads_x, THREADS_PER_BLOCK);
        uint threads_y = THREADS_PER_BLOCK / threads_x;

        uint blocks_x = divup(in.dims[0], threads_x * REPEAT);
        uint blocks_y = divup(in.dims[1], threads_y);

        Array<To> tmpOut = *initArray<To>();
        Array<Tw> tmpWt  = *initArray<Tw>();
        if (blocks_x > 1) {
          tmpOut = createEmptyArray<To>({blocks_x, in.dims[1], in.dims[2], in.dims[3]});
          tmpWt = createEmptyArray<Tw>({blocks_x, in.dims[1], in.dims[2], in.dims[3]});
        } else {
          tmpOut = createParamArray(out, false);
        }

        mean_first_launcher<Ti, Tw, To>(tmpOut, tmpWt, in, iwt, blocks_x, blocks_y, threads_x);

        if (blocks_x > 1) {
            Param<Tw> owt;
            owt.ptr = NULL;
            mean_first_launcher<To, Tw, To>(out, owt, tmpOut, tmpWt, 1, blocks_y, threads_x);
        }
    }

    template<typename Ti, typename Tw, typename To>
    void mean_weighted(Param<To> out, CParam<Ti> in, CParam<Tw> iwt, int dim)
    {
        switch (dim) {
        case 0: return mean_first<Ti, Tw, To   >(out, in, iwt);
        case 1: return mean_dim  <Ti, Tw, To, 1>(out, in, iwt);
        case 2: return mean_dim  <Ti, Tw, To, 2>(out, in, iwt);
        case 3: return mean_dim  <Ti, Tw, To, 3>(out, in, iwt);
        }
    }

    template<typename Ti, typename Tw, typename To>
    void mean(Param<To> out, CParam<Ti> in, int dim)
    {
        Param<Tw> dummy_weight;
        mean_weighted<Ti, Tw, To>(out, in, dummy_weight, dim);
    }

    template<typename T, typename Tw>
    T mean_all_weighted(CParam<T> in, CParam<Tw> iwt)
    {
        int in_elements = in.dims[0] * in.dims[1] * in.dims[2] * in.dims[3];

        // FIXME: Use better heuristics to get to the optimum number
        if (in_elements > 4096) {

            bool in_is_linear = (in.strides[0] == 1);
            bool wt_is_linear = (iwt.strides[0] == 1);
            for (int k = 1; k < 4; k++) {
                in_is_linear &= ( in.strides[k] == ( in.strides[k - 1] *  in.dims[k - 1]));
                wt_is_linear &= (iwt.strides[k] == (iwt.strides[k - 1] * iwt.dims[k - 1]));
            }

            if (in_is_linear && wt_is_linear) {
                in.dims[0] = in_elements;
                for (int k = 1; k < 4; k++) {
                    in.dims[k] = 1;
                    in.strides[k] = in_elements;
                }

                for (int k = 0; k < 4; k++) {
                    iwt.dims[k] = in.dims[k];
                    iwt.strides[k] = in.strides[k];
                }
            }

            uint threads_x = nextpow2(std::max(32u, (uint)in.dims[0]));
            threads_x = std::min(threads_x, THREADS_PER_BLOCK);
            uint threads_y = THREADS_PER_BLOCK / threads_x;

            uint blocks_x = divup(in.dims[0], threads_x * REPEAT);
            uint blocks_y = divup(in.dims[1], threads_y);

            Array<T> tmpOut = createEmptyArray<T>({blocks_x, in.dims[1], in.dims[2], in.dims[3]});
            Array<Tw> tmpWt = createEmptyArray<Tw>({blocks_x, in.dims[1], in.dims[2], in.dims[3]});

            int tmp_elements = tmpOut.elements();

            mean_first_launcher<T, Tw, T>(tmpOut, tmpWt, in, iwt, blocks_x, blocks_y, threads_x);

            vector<T>  h_ptr(tmp_elements);
            vector<Tw> h_wptr(tmp_elements);

            copyData<T>(h_ptr.data(),  tmpOut);
            copyData<Tw>(h_wptr.data(), tmpWt);
            CUDA_CHECK(cudaStreamSynchronize(cuda::getStream(cuda::getActiveDeviceId())));

            T val = h_ptr[0];
            Tw weight = h_wptr[0];

            for (int i = 1; i < tmp_elements; i++) {
                stable_mean(&val, &weight, h_ptr[i], h_wptr[i]);
            }

            return val;
        } else {

            vector<T>  h_ptr(in_elements);
            vector<Tw> h_wptr(in_elements);

            CUDA_CHECK(cudaMemcpyAsync(h_ptr.data(), in.ptr, in_elements * sizeof(T),
                       cudaMemcpyDeviceToHost, cuda::getStream(cuda::getActiveDeviceId())));
            CUDA_CHECK(cudaMemcpyAsync(h_wptr.data(), iwt.ptr, in_elements * sizeof(Tw),
                       cudaMemcpyDeviceToHost, cuda::getStream(cuda::getActiveDeviceId())));
            CUDA_CHECK(cudaStreamSynchronize(cuda::getStream(cuda::getActiveDeviceId())));

            T val = h_ptr[0];
            Tw weight = h_wptr[0];
            for (int i = 1; i < in_elements; i++) {
                stable_mean(&val, &weight, h_ptr[i], h_wptr[i]);
            }

            return val;
        }
    }

    template<typename Ti, typename Tw, typename To>
    To mean_all(CParam<Ti> in)
    {
        using std::unique_ptr;
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


            uint blocks_x = divup(in.dims[0], threads_x * REPEAT);
            uint blocks_y = divup(in.dims[1], threads_y);

            Param<Tw> iwt;
            Array<To> tmpOut = createEmptyArray<To>({blocks_x, in.dims[1], in.dims[2], in.dims[3]});
            Array<Tw> tmpCt = createEmptyArray<Tw>({blocks_x, in.dims[1], in.dims[2], in.dims[3]});

            mean_first_launcher<Ti, Tw, To>(tmpOut, tmpCt, in, iwt, blocks_x, blocks_y, threads_x);

            int tmp_elements = tmpOut.elements();
            vector<To> h_ptr(tmp_elements);
            vector<Tw> h_cptr(tmp_elements);

            copyData<To>(h_ptr.data(), tmpOut);
            copyData<Tw>(h_cptr.data(), tmpCt);
            CUDA_CHECK(cudaStreamSynchronize(cuda::getStream(cuda::getActiveDeviceId())));

            To val = h_ptr[0];
            Tw weight = h_cptr[0];

            for (int i = 1; i < tmp_elements; i++) {
                stable_mean(&val, &weight, h_ptr[i], h_cptr[i]);
            }

            return val;
        } else {

            vector<Ti> h_ptr(in_elements);

            CUDA_CHECK(cudaMemcpyAsync(h_ptr.data(), in.ptr, in_elements * sizeof(Ti),
                       cudaMemcpyDeviceToHost, cuda::getStream(cuda::getActiveDeviceId())));
            CUDA_CHECK(cudaStreamSynchronize(cuda::getStream(cuda::getActiveDeviceId())));

            Transform<Ti, To, af_add_t> transform;
            Tw count = (Tw)1;

            To val = transform(h_ptr[0]);
            Tw weight = count;
            for (int i = 1; i < in_elements; i++) {
                stable_mean(&val, &weight, transform(h_ptr[i]), count);
            }

            return val;
        }
    }

}
}

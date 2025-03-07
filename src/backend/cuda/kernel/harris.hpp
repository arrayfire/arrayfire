/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/dispatch.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <memory.hpp>
#include <af/constants.h>

#include "config.hpp"
#include "convolve.hpp"
#include "gradient.hpp"
#include "range.hpp"
#include "sort_by_key.hpp"

#include <vector>

namespace arrayfire {
namespace cuda {

namespace kernel {

static const unsigned BLOCK_SIZE = 16;

template<typename T>
void gaussian1D(T* out, const int dim, double sigma = 0.0) {
    if (!(sigma > 0)) sigma = 0.25 * dim;

    T sum = (T)0;
    for (int i = 0; i < dim; i++) {
        int x = i - (dim - 1) / 2;
        T el  = 1. / sqrt(2 * af::Pi * sigma * sigma) *
               exp(-((x * x) / (2 * (sigma * sigma))));
        out[i] = el;
        sum += el;
    }

    for (int k = 0; k < dim; k++) out[k] /= sum;
}

// max_val()
// Returns max of x and y
inline __device__ int max_val(const int x, const int y) { return max(x, y); }
inline __device__ unsigned max_val(const unsigned x, const unsigned y) {
    return max(x, y);
}
inline __device__ float max_val(const float x, const float y) {
    return fmax(x, y);
}
inline __device__ double max_val(const double x, const double y) {
    return fmax(x, y);
}

template<typename T>
__global__ void second_order_deriv(T* ixx_out, T* ixy_out, T* iyy_out,
                                   const unsigned in_len, const T* ix_in,
                                   const T* iy_in) {
    const unsigned x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x < in_len) {
        ixx_out[x] = ix_in[x] * ix_in[x];
        ixy_out[x] = ix_in[x] * iy_in[x];
        iyy_out[x] = iy_in[x] * iy_in[x];
    }
}

template<typename T>
__global__ void harris_responses(T* resp_out, const unsigned idim0,
                                 const unsigned idim1, const T* ixx_in,
                                 const T* ixy_in, const T* iyy_in,
                                 const float k_thr, const unsigned border_len) {
    const unsigned r = border_len;

    const unsigned x = blockDim.x * blockIdx.x + threadIdx.x + r;
    const unsigned y = blockDim.y * blockIdx.y + threadIdx.y + r;

    if (x < idim1 - r && y < idim0 - r) {
        const unsigned idx = x * idim0 + y;

        // Calculates matrix trace and determinant
        T tr  = ixx_in[idx] + iyy_in[idx];
        T det = ixx_in[idx] * iyy_in[idx] - ixy_in[idx] * ixy_in[idx];

        // Calculates local Harris response
        resp_out[idx] = det - k_thr * (tr * tr);
    }
}

template<typename T>
__global__ void non_maximal(float* x_out, float* y_out, float* resp_out,
                            unsigned* count, const unsigned idim0,
                            const unsigned idim1, const T* resp_in,
                            const float min_resp, const unsigned border_len,
                            const unsigned max_corners) {
    // Responses on the border don't have 8-neighbors to compare, discard them
    const unsigned r = border_len + 1;

    const unsigned x = blockDim.x * blockIdx.x + threadIdx.x + r;
    const unsigned y = blockDim.y * blockIdx.y + threadIdx.y + r;

    if (x < idim1 - r && y < idim0 - r) {
        const T v = resp_in[x * idim0 + y];

        // Find maximum neighborhood response
        T max_v;
        max_v = max_val(resp_in[(x - 1) * idim0 + y - 1],
                        resp_in[x * idim0 + y - 1]);
        max_v = max_val(max_v, resp_in[(x + 1) * idim0 + y - 1]);
        max_v = max_val(max_v, resp_in[(x - 1) * idim0 + y]);
        max_v = max_val(max_v, resp_in[(x + 1) * idim0 + y]);
        max_v = max_val(max_v, resp_in[(x - 1) * idim0 + y + 1]);
        max_v = max_val(max_v, resp_in[(x)*idim0 + y + 1]);
        max_v = max_val(max_v, resp_in[(x + 1) * idim0 + y + 1]);

        // Stores corner to {x,y,resp}_out if it's response is maximum compared
        // to its 8-neighborhood and greater or equal minimum response
        if (v > max_v && v >= (T)min_resp) {
            unsigned idx = atomicAdd(count, 1u);
            if (idx < max_corners) {
                x_out[idx]    = (float)x;
                y_out[idx]    = (float)y;
                resp_out[idx] = (float)v;
            }
        }
    }
}

__global__ void keep_corners(float* x_out, float* y_out, float* resp_out,
                             const float* x_in, const float* y_in,
                             const float* resp_in, const unsigned* resp_idx,
                             const unsigned n_corners) {
    const unsigned f = blockDim.x * blockIdx.x + threadIdx.x;

    // Keep only the first n_feat features
    if (f < n_corners) {
        x_out[f]    = x_in[(unsigned)resp_idx[f]];
        y_out[f]    = y_in[(unsigned)resp_idx[f]];
        resp_out[f] = resp_in[f];
    }
}

int compare(const void* a, const void* b) { return *(float*)a > *(float*)b; }

template<typename T, typename convAccT>
void harris(unsigned* corners_out, float** x_out, float** y_out,
            float** resp_out, CParam<T> in, const unsigned max_corners,
            const float min_response, const float sigma,
            const unsigned filter_len, const float k_thr) {
    // Window filter
    std::vector<convAccT> h_filter(filter_len);
    // Decide between rectangular or circular filter
    if (sigma < 0.5f) {
        for (unsigned i = 0; i < filter_len; i++)
            h_filter[i] = (T)1.f / (filter_len);
    } else {
        gaussian1D<convAccT>(h_filter.data(), (int)filter_len, sigma);
    }

    // Copy filter to device object
    Param<convAccT> filter;
    filter.dims[0]    = filter_len;
    filter.strides[0] = 1;

    for (int k = 1; k < 4; k++) {
        filter.dims[k]    = 1;
        filter.strides[k] = filter.dims[k - 1] * filter.strides[k - 1];
    }

    int filter_elem   = filter.strides[3] * filter.dims[3];
    auto filter_alloc = memAlloc<convAccT>(filter_elem);
    filter.ptr        = filter_alloc.get();
    CUDA_CHECK(cudaMemcpyAsync(filter.ptr, h_filter.data(),
                               filter_elem * sizeof(convAccT),
                               cudaMemcpyHostToDevice, getActiveStream()));

    const unsigned border_len = filter_len / 2 + 1;

    Param<T> ix, iy;
    for (dim_t i = 0; i < 4; i++) {
        ix.dims[i] = iy.dims[i] = in.dims[i];
        ix.strides[i] = iy.strides[i] = in.strides[i];
    }
    auto ix_alloc = memAlloc<T>(ix.dims[3] * ix.strides[3]);
    auto iy_alloc = memAlloc<T>(iy.dims[3] * iy.strides[3]);
    ix.ptr        = ix_alloc.get();
    iy.ptr        = iy_alloc.get();

    // Compute first-order derivatives as gradients
    gradient<T>(iy, ix, in);

    Param<T> ixx, ixy, iyy;
    Param<T> ixx_tmp, ixy_tmp, iyy_tmp;
    for (dim_t i = 0; i < 4; i++) {
        ixx.dims[i] = ixy.dims[i] = iyy.dims[i] = in.dims[i];
        ixx_tmp.dims[i] = ixy_tmp.dims[i] = iyy_tmp.dims[i] = in.dims[i];
        ixx.strides[i] = ixy.strides[i] = iyy.strides[i] = in.strides[i];
        ixx_tmp.strides[i] = ixy_tmp.strides[i] = iyy_tmp.strides[i] =
            in.strides[i];
    }
    auto ixx_alloc = memAlloc<T>(ixx.dims[3] * ixx.strides[3]);
    auto ixy_alloc = memAlloc<T>(ixy.dims[3] * ixy.strides[3]);
    auto iyy_alloc = memAlloc<T>(iyy.dims[3] * iyy.strides[3]);
    ixx.ptr        = ixx_alloc.get();
    ixy.ptr        = ixy_alloc.get();
    iyy.ptr        = iyy_alloc.get();

    // Compute second-order derivatives
    dim3 threads(THREADS_PER_BLOCK, 1);
    dim3 blocks(divup(in.dims[3] * in.strides[3], threads.x), 1);
    CUDA_LAUNCH((second_order_deriv<T>), blocks, threads, ixx.ptr, ixy.ptr,
                iyy.ptr, in.dims[3] * in.strides[3], ix.ptr, iy.ptr);

    auto ixx_tmp_alloc = memAlloc<T>(ixx_tmp.dims[3] * ixx_tmp.strides[3]);
    auto ixy_tmp_alloc = memAlloc<T>(ixy_tmp.dims[3] * ixy_tmp.strides[3]);
    auto iyy_tmp_alloc = memAlloc<T>(iyy_tmp.dims[3] * iyy_tmp.strides[3]);
    ixx_tmp.ptr        = ixx_tmp_alloc.get();
    ixy_tmp.ptr        = ixy_tmp_alloc.get();
    iyy_tmp.ptr        = iyy_tmp_alloc.get();

    // Convolve second-order derivatives with proper window filter
    convolve2<T, convAccT>(ixx_tmp, CParam<T>(ixx), filter, 0, false);
    convolve2<T, convAccT>(ixx, CParam<T>(ixx_tmp), filter, 1, false);
    convolve2<T, convAccT>(ixy_tmp, CParam<T>(ixy), filter, 0, false);
    convolve2<T, convAccT>(ixy, CParam<T>(ixy_tmp), filter, 1, false);
    convolve2<T, convAccT>(iyy_tmp, CParam<T>(iyy), filter, 0, false);
    convolve2<T, convAccT>(iyy, CParam<T>(iyy_tmp), filter, 1, false);

    // Number of corners is not known a priori, limit maximum number of corners
    // according to image dimensions
    unsigned corner_lim = in.dims[3] * in.strides[3] * 0.2f;

    auto d_corners_found = memAlloc<unsigned>(1);
    CUDA_CHECK(cudaMemsetAsync(d_corners_found.get(), 0, sizeof(unsigned),
                               getActiveStream()));

    auto d_x_corners    = memAlloc<float>(corner_lim);
    auto d_y_corners    = memAlloc<float>(corner_lim);
    auto d_resp_corners = memAlloc<float>(corner_lim);

    auto d_responses = memAlloc<T>(in.dims[3] * in.strides[3]);

    // Calculate Harris responses for all pixels
    threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    blocks  = dim3(divup(in.dims[1] - border_len * 2, threads.x),
                   divup(in.dims[0] - border_len * 2, threads.y));
    CUDA_LAUNCH((harris_responses<T>), blocks, threads, d_responses.get(),
                in.dims[0], in.dims[1], ixx.ptr, ixy.ptr, iyy.ptr, k_thr,
                border_len);

    const float min_r = (max_corners > 0) ? 0.f : min_response;

    // Perform non-maximal suppression
    CUDA_LAUNCH((non_maximal<T>), blocks, threads, d_x_corners.get(),
                d_y_corners.get(), d_resp_corners.get(), d_corners_found.get(),
                in.dims[0], in.dims[1], d_responses.get(), min_r, border_len,
                corner_lim);

    unsigned corners_found = 0;
    CUDA_CHECK(cudaMemcpyAsync(&corners_found, d_corners_found.get(),
                               sizeof(unsigned), cudaMemcpyDeviceToHost,
                               getActiveStream()));
    CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));

    *corners_out =
        min(corners_found, (max_corners > 0) ? max_corners : corner_lim);

    if (*corners_out == 0) return;

    if (max_corners > 0 && corners_found > *corners_out) {
        Param<float> harris_responses;
        Param<unsigned> harris_idx;

        harris_responses.dims[0] = harris_idx.dims[0] = corners_found;
        harris_responses.strides[0] = harris_idx.strides[0] = 1;

        for (int k = 1; k < 4; k++) {
            harris_responses.dims[k] = 1;
            harris_responses.strides[k] =
                harris_responses.dims[k - 1] * harris_responses.strides[k - 1];
            harris_idx.dims[k] = 1;
            harris_idx.strides[k] =
                harris_idx.dims[k - 1] * harris_idx.strides[k - 1];
        }

        int sort_elem = harris_responses.strides[3] * harris_responses.dims[3];
        harris_responses.ptr = d_resp_corners.get();
        // Create indices using range
        auto harris_idx_alloc = memAlloc<unsigned>(sort_elem);
        harris_idx.ptr        = harris_idx_alloc.get();
        kernel::range<uint>(harris_idx, 0);

        // Sort Harris responses
        sort0ByKey<float, uint>(harris_responses, harris_idx, false);

        auto x_out_alloc    = memAlloc<float>(*corners_out);
        auto y_out_alloc    = memAlloc<float>(*corners_out);
        auto resp_out_alloc = memAlloc<float>(*corners_out);
        *x_out              = x_out_alloc.get();
        *y_out              = y_out_alloc.get();
        *resp_out           = resp_out_alloc.get();

        // Keep only the first corners_to_keep corners with higher Harris
        // responses
        threads = dim3(THREADS_PER_BLOCK, 1);
        blocks  = dim3(divup(*corners_out, threads.x), 1);
        CUDA_LAUNCH(keep_corners, blocks, threads, *x_out, *y_out, *resp_out,
                    d_x_corners.get(), d_y_corners.get(), harris_responses.ptr,
                    harris_idx.ptr, *corners_out);

        x_out_alloc.release();
        y_out_alloc.release();
        resp_out_alloc.release();
    } else if (max_corners == 0 && corners_found < corner_lim) {
        auto x_out_alloc    = memAlloc<float>(*corners_out);
        auto y_out_alloc    = memAlloc<float>(*corners_out);
        auto resp_out_alloc = memAlloc<float>(*corners_out);
        *x_out              = x_out_alloc.get();
        *y_out              = y_out_alloc.get();
        *resp_out           = resp_out_alloc.get();

        CUDA_CHECK(cudaMemcpyAsync(
            *x_out, d_x_corners.get(), *corners_out * sizeof(float),
            cudaMemcpyDeviceToDevice, getActiveStream()));
        CUDA_CHECK(cudaMemcpyAsync(
            *y_out, d_y_corners.get(), *corners_out * sizeof(float),
            cudaMemcpyDeviceToDevice, getActiveStream()));
        CUDA_CHECK(cudaMemcpyAsync(
            *resp_out, d_resp_corners.get(), *corners_out * sizeof(float),
            cudaMemcpyDeviceToDevice, getActiveStream()));

        x_out_alloc.release();
        y_out_alloc.release();
        resp_out_alloc.release();
    } else {
        *x_out    = d_x_corners.release();
        *y_out    = d_y_corners.release();
        *resp_out = d_resp_corners.release();
    }
    filter_alloc.release();
}

}  // namespace kernel

}  // namespace cuda
}  // namespace arrayfire

/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <memory.hpp>
#include "ireduce.hpp"
#include "reduce.hpp"
#include "sort.hpp"

#include <cfloat>

#include <iostream>

namespace cuda
{

namespace kernel
{

template<typename T>
__device__ T sq(T a)
{
    return a * a;
}

template<typename T>
struct EPS
{
    __device__ T eps() { return FLT_EPSILON; }
};

template<>
struct EPS<float>
{
    __device__ static float eps() { return FLT_EPSILON; }
};

template<>
struct EPS<double>
{
    __device__ static double eps() { return DBL_EPSILON; }
};

#define RANSACConfidence 0.99f
#define LMEDSConfidence 0.99f
#define LMEDSOutlierRatio 0.4f


template<typename T>
__device__ void JacobiSVD(T* S, T* V, int m, int n)
{
    const int iterations = 30;

    int tid_x = threadIdx.x;
    int bsz_x = blockDim.x;
    int tid_y = threadIdx.y;
    int gid_y = blockIdx.y * blockDim.y + tid_y;

    __shared__ T acc[512];
    T* acc1 = acc;
    T* acc2 = acc + 256;

    __shared__ T s_S[16*81];
    __shared__ T s_V[16*81];
    __shared__ T d[16*9];

    for (int i = 0; i <= 4; i++)
        s_S[tid_y * 81 + i*bsz_x + tid_x] = S[gid_y * 81 + i*bsz_x + tid_x];
    if (tid_x == 0)
        s_S[tid_y * 81 + 80] = S[gid_y * 81 + 80];
    __syncthreads();

    // Copy first 80 elements
    for (int i = 0; i <= 4; i++) {
        T t = s_S[tid_y*81 + tid_x+i*bsz_x];
        acc1[tid_y*bsz_x + tid_x] += t*t;
    }
    if (tid_x < 8)
        acc1[tid_y*16 + tid_x] += acc1[tid_y*16 + tid_x+8];
    __syncthreads();
    if (tid_x < 4)
        acc1[tid_y*16 + tid_x] += acc1[tid_y*16 + tid_x+4];
    __syncthreads();
    if (tid_x < 2)
        acc1[tid_y*16 + tid_x] += acc1[tid_y*16 + tid_x+2];
    __syncthreads();
    if (tid_x < 1) {
        // Copy last element
        T t = s_S[tid_y*bsz_x + tid_x+80];
        acc1[tid_y*16 + tid_x] += acc1[tid_y*16 + tid_x+1] + t*t;
    }
    __syncthreads();

    if (tid_x < n)
        d[tid_y*9 + tid_x] = acc1[tid_y*bsz_x + tid_x];

    // V is initialized as an identity matrix
    for (int i = 0; i <= 4; i++) {
        s_V[tid_y*81 + i*bsz_x + tid_x] = 0;
    }
    __syncthreads();
    if (tid_x < m)
        s_V[tid_y*81 + tid_x*m + tid_x] = 1;
    __syncthreads();

    for (int it = 0; it < iterations; it++) {
        bool converged = false;

        for (int i = 0; i < n-1; i++) {
            for (int j = i+1; j < n; j++) {
                T* Si = s_S + tid_y*81 + i*m;
                T* Sj = s_S + tid_y*81 + j*m;

                T p = (T)0;
                for (int k = 0; k < m; k++)
                    p += Si[k]*Sj[k];

                if (abs(p) <= EPS<T>::eps()*sqrt(d[tid_y*9 + i]*d[tid_y*9 + j]))
                    continue;

                T y = d[tid_y*9 + i] - d[tid_y*9 + j];
                T r = hypot(p*2, y);
                T r2 = r*2;
                T c, s;
                if (y >= 0) {
                    c = sqrt((r + y) / r2);
                    s = p / (r2*c);
                }
                else {
                    s = sqrt((r - y) / r2);
                    c = p / (r2*s);
                }

                if (tid_x < m) {
                    T t0 = c*Si[tid_x] + s*Sj[tid_x];
                    T t1 = c*Sj[tid_x] - s*Si[tid_x];
                    Si[tid_x] = t0;
                    Sj[tid_x] = t1;

                    acc1[tid_y*16 + tid_x] = t0*t0;
                    acc2[tid_y*16 + tid_x] = t1*t1;
                }
                __syncthreads();

                if (tid_x < 4) {
                    acc1[tid_y*16 + tid_x] += acc1[tid_y*16 + tid_x+4];
                    acc2[tid_y*16 + tid_x] += acc2[tid_y*16 + tid_x+4];
                }
                __syncthreads();
                if (tid_x < 2) {
                    acc1[tid_y*16 + tid_x] += acc1[tid_y*16 + tid_x+2];
                    acc2[tid_y*16 + tid_x] += acc2[tid_y*16 + tid_x+2];
                }
                __syncthreads();
                if (tid_x < 1) {
                    acc1[tid_y*16 + tid_x] += acc1[tid_y*16 + tid_x+1] + acc1[tid_y*16 + tid_x+8];
                    acc2[tid_y*16 + tid_x] += acc2[tid_y*16 + tid_x+1] + acc2[tid_y*16 + tid_x+8];
                }
                __syncthreads();

                if (tid_x == 0) {
                    d[tid_y*9 + i] = acc1[tid_y*16];
                    d[tid_y*9 + j] = acc2[tid_y*16];
                }
                __syncthreads();

                T* Vi = s_V + tid_y*81 + i*n;
                T* Vj = s_V + tid_y*81 + j*n;

                if (tid_x < n) {
                    T t0 = Vi[tid_x] * c + Vj[tid_x] * s;
                    T t1 = Vj[tid_x] * c - Vi[tid_x] * s;

                    Vi[tid_x] = t0;
                    Vj[tid_x] = t1;
                }
                __syncthreads();

                converged = true;
            }
            if (!converged)
                break;
        }
    }
    __syncthreads();

    for (int i = 0; i <= 4; i++)
        V[gid_y * 81 + tid_x+i*bsz_x] = s_V[tid_y * 81 + tid_x+i*bsz_x];
    if (tid_x == 0)
        V[gid_y * 81 + 80] = s_V[tid_y * 81 + 80];
    __syncthreads();
}

__device__ bool computeMeanScale(
    float* x_src_mean,
    float* y_src_mean,
    float* x_dst_mean,
    float* y_dst_mean,
    float* src_scale,
    float* dst_scale,
    float* src_pt_x,
    float* src_pt_y,
    float* dst_pt_x,
    float* dst_pt_y,
    CParam<float> x_src,
    CParam<float> y_src,
    CParam<float> x_dst,
    CParam<float> y_dst,
    CParam<float> rnd,
    int i)
{
    const unsigned ridx = rnd.dims[0] * i;
    unsigned r[4] = { (unsigned)rnd.ptr[ridx],
                      (unsigned)rnd.ptr[ridx+1],
                      (unsigned)rnd.ptr[ridx+2],
                      (unsigned)rnd.ptr[ridx+3] };

    // If one of the points is repeated, it's a bad samples, will still
    // compute homography to ensure all threads pass __syncthreads()
    bool bad = (r[0] == r[1] || r[0] == r[2] || r[0] == r[3] ||
                r[1] == r[2] || r[1] == r[3] || r[2] == r[3]);

    for (unsigned j = 0; j < 4; j++) {
        src_pt_x[j] = x_src.ptr[r[j]];
        src_pt_y[j] = y_src.ptr[r[j]];
        dst_pt_x[j] = x_dst.ptr[r[j]];
        dst_pt_y[j] = y_dst.ptr[r[j]];
    }

    *x_src_mean = (src_pt_x[0] + src_pt_x[1] + src_pt_x[2] + src_pt_x[3]) / 4.f;
    *y_src_mean = (src_pt_y[0] + src_pt_y[1] + src_pt_y[2] + src_pt_y[3]) / 4.f;
    *x_dst_mean = (dst_pt_x[0] + dst_pt_x[1] + dst_pt_x[2] + dst_pt_x[3]) / 4.f;
    *y_dst_mean = (dst_pt_y[0] + dst_pt_y[1] + dst_pt_y[2] + dst_pt_y[3]) / 4.f;

    float src_var = 0.0f, dst_var = 0.0f;
    for (unsigned j = 0; j < 4; j++) {
        src_var += sq(src_pt_x[j] - *x_src_mean) + sq(src_pt_y[j] - *y_src_mean);
        dst_var += sq(dst_pt_x[j] - *x_dst_mean) + sq(dst_pt_y[j] - *y_dst_mean);
    }

    src_var /= 4.f;
    dst_var /= 4.f;

    *src_scale = sqrt(2.0f) / sqrt(src_var);
    *dst_scale = sqrt(2.0f) / sqrt(dst_var);

    return !bad;
}

#define APTR(Z, Y, X) (A.ptr[(Z) * A.dims[0] * A.dims[1] + (Y) * A.dims[0] + (X)])

template<typename T>
__global__ void buildLinearSystem(
    Param<T> H,
    Param<T> A,
    Param<T> V,
    CParam<float> x_src,
    CParam<float> y_src,
    CParam<float> x_dst,
    CParam<float> y_dst,
    CParam<float> rnd,
    const unsigned iterations)
{
    unsigned i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < iterations) {
        float x_src_mean, y_src_mean;
        float x_dst_mean, y_dst_mean;
        float src_scale, dst_scale;
        float src_pt_x[4], src_pt_y[4], dst_pt_x[4], dst_pt_y[4];

        computeMeanScale(&x_src_mean, &y_src_mean,
                         &x_dst_mean, &y_dst_mean,
                         &src_scale, &dst_scale,
                         src_pt_x, src_pt_y,
                         dst_pt_x, dst_pt_y,
                         x_src, y_src, x_dst, y_dst,
                         rnd, i);

        // Compute input matrix
        for (unsigned j = threadIdx.x; j < 4; j+=blockDim.x) {
            float srcx = (src_pt_x[j] - x_src_mean) * src_scale;
            float srcy = (src_pt_y[j] - y_src_mean) * src_scale;
            float dstx = (dst_pt_x[j] - x_dst_mean) * dst_scale;
            float dsty = (dst_pt_y[j] - y_dst_mean) * dst_scale;

            APTR(i, 3, j*2) = -srcx;
            APTR(i, 4, j*2) = -srcy;
            APTR(i, 5, j*2) = -1.0f;
            APTR(i, 6, j*2) = dsty*srcx;
            APTR(i, 7, j*2) = dsty*srcy;
            APTR(i, 8, j*2) = dsty;

            APTR(i, 0, j*2+1) = srcx;
            APTR(i, 1, j*2+1) = srcy;
            APTR(i, 2, j*2+1) = 1.0f;
            APTR(i, 6, j*2+1) = -dstx*srcx;
            APTR(i, 7, j*2+1) = -dstx*srcy;
            APTR(i, 8, j*2+1) = -dstx;
        }

        JacobiSVD<T>(A.ptr, V.ptr, 9, 9);

        T vH[9], H_tmp[9];
        for (unsigned j = 0; j < 9; j++)
            vH[j] = V.ptr[i * V.dims[0] * V.dims[1] + 8 * V.dims[0] + j];

        H_tmp[0] = src_scale*x_dst_mean*vH[6] + src_scale*vH[0]/dst_scale;
        H_tmp[1] = src_scale*x_dst_mean*vH[7] + src_scale*vH[1]/dst_scale;
        H_tmp[2] = x_dst_mean*(vH[8] - src_scale*y_src_mean*vH[7] - src_scale*x_src_mean*vH[6]) +
                              (vH[2] - src_scale*y_src_mean*vH[1] - src_scale*x_src_mean*vH[0])/dst_scale;

        H_tmp[3] = src_scale*y_dst_mean*vH[6] + src_scale*vH[3]/dst_scale;
        H_tmp[4] = src_scale*y_dst_mean*vH[7] + src_scale*vH[4]/dst_scale;
        H_tmp[5] = y_dst_mean*(vH[8] - src_scale*y_src_mean*vH[7] - src_scale*x_src_mean*vH[6]) +
                              (vH[5] - src_scale*y_src_mean*vH[4] - src_scale*x_src_mean*vH[3])/dst_scale;

        H_tmp[6] = src_scale*vH[6];
        H_tmp[7] = src_scale*vH[7];
        H_tmp[8] = vH[8] - src_scale*y_src_mean*vH[7] - src_scale*x_src_mean*vH[6];

        const unsigned Hidx = H.dims[0] * i;
        T* H_ptr = H.ptr + Hidx;
        for (int h = 0; h < 9; h++)
            H_ptr[h] = H_tmp[h];
    }
}

#undef APTR

// LMedS: http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node25.html
template<typename T>
__global__ void computeEvalHomography(
    Param<unsigned> inliers,
    Param<unsigned> idx,
    Param<T> H,
    Param<float> err,
    CParam<float> x_src,
    CParam<float> y_src,
    CParam<float> x_dst,
    CParam<float> y_dst,
    CParam<float> rnd,
    const unsigned iterations,
    const unsigned nsamples,
    const float inlier_thr,
    const af_homography_type htype)
{
    unsigned bid_x = blockIdx.x;
    unsigned tid_x = threadIdx.x;
    unsigned i = bid_x * blockDim.x + tid_x;

    __shared__ unsigned s_inliers[256];
    __shared__ unsigned s_idx[256];

    s_inliers[tid_x] = 0;
    s_idx[tid_x]     = 0;
    __syncthreads();

    if (i < iterations) {
        const unsigned Hidx = H.dims[0] * i;
        T* H_ptr = H.ptr + Hidx;
        T H_tmp[9];
        for (int h = 0; h < 9; h++)
            H_tmp[h] = H_ptr[h];

        if (htype == AF_RANSAC) {
            // Compute inliers
            unsigned inliers_count = 0;
            for (unsigned j = 0; j < nsamples; j++) {
                float z =  H_tmp[6]*x_src.ptr[j] + H_tmp[7]*y_src.ptr[j] + H_tmp[8];
                float x = (H_tmp[0]*x_src.ptr[j] + H_tmp[1]*y_src.ptr[j] + H_tmp[2]) / z;
                float y = (H_tmp[3]*x_src.ptr[j] + H_tmp[4]*y_src.ptr[j] + H_tmp[5]) / z;

                float dist = sq(x_dst.ptr[j] - x) + sq(y_dst.ptr[j] - y);
                if (dist < inlier_thr*inlier_thr)
                    inliers_count++;
            }

            s_inliers[tid_x] = inliers_count;
            s_idx[tid_x]     = i;
        }
        else if (htype == AF_LMEDS) {
            // Compute error
            for (unsigned j = 0; j < nsamples; j++) {
                float z =  H_tmp[6]*x_src.ptr[j] + H_tmp[7]*y_src.ptr[j] + H_tmp[8];
                float x = (H_tmp[0]*x_src.ptr[j] + H_tmp[1]*y_src.ptr[j] + H_tmp[2]) / z;
                float y = (H_tmp[3]*x_src.ptr[j] + H_tmp[4]*y_src.ptr[j] + H_tmp[5]) / z;

                float dist = sq(x_dst.ptr[j] - x) + sq(y_dst.ptr[j] - y);
                err.ptr[i*err.dims[0] + j] = sqrt(dist);
            }
        }
    }

    if (htype == AF_RANSAC) {
        // Find sample with most inliers
        for (unsigned tx = 128; tx > 0; tx >>= 1) {
            if (tid_x < tx) {
                if (s_inliers[tid_x + tx] > s_inliers[tid_x]) {
                    s_inliers[tid_x] = s_inliers[tid_x + tx];
                    s_idx[tid_x]     = s_idx[tid_x + tx];
                }
            }
            __syncthreads();
        }

        inliers.ptr[bid_x] = s_inliers[0];
        idx.ptr[bid_x]     = s_idx[0];
    }
}

__global__ void computeMedian(
    Param<float> median,
    Param<unsigned> idx,
    CParam<float> err,
    const unsigned iterations)
{
    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned i = bid * blockDim.x + threadIdx.x;

    __shared__ float s_median[256];
    __shared__ unsigned s_idx[256];

    s_median[tid] = FLT_MAX;
    s_idx[tid] = 0;
    __syncthreads();

    if (i < iterations) {
        const int nsamples = err.dims[0];
        float m = err.ptr[i*nsamples + nsamples / 2];
        if (nsamples % 2 == 0)
            m = (m + err.ptr[i*nsamples + nsamples / 2 - 1]) * 0.5f;

        s_idx[tid] = i;
        s_median[tid] = m;
    }
    __syncthreads();

    for (unsigned t = 128; t > 0; t >>= 1) {
        if (tid < t) {
            if (s_median[tid + t] < s_median[tid]) {
                s_median[tid] = s_median[tid + t];
                s_idx[tid]    = s_idx[tid + t];
            }
        }
        __syncthreads();
    }

    median.ptr[bid] = s_median[0];
    idx.ptr[bid] = s_idx[0];
}

#define DIVUP(A, B) (((A) + (B) - 1) / (B))

__global__ void findMinMedian(
    float* minMedian,
    unsigned* minIdx,
    CParam<float> median,
    CParam<unsigned> idx)
{
    const int tid = threadIdx.x;

    __shared__ float s_minMedian[256];
    __shared__ unsigned s_minIdx[256];

    s_minMedian[tid] = FLT_MAX;
    s_minIdx[tid] = 0;
    __syncthreads();

    const int loop = DIVUP(median.dims[0], blockDim.x);

    for (int i = 0; i < loop; i++) {
        int j = i * blockDim.x + tid;
        if (j < median.dims[0] && median.ptr[j] < s_minMedian[tid]) {
            s_minMedian[tid] = median.ptr[j];
            s_minIdx[tid] = idx.ptr[j];
        }
        __syncthreads();
    }

    for (unsigned t = 128; t > 0; t >>= 1) {
        if (tid < t) {
            if (s_minMedian[tid + t] < s_minMedian[tid]) {
                s_minMedian[tid] = s_minMedian[tid + t];
                s_minIdx[tid]    = s_minIdx[tid + t];
            }
        }
        __syncthreads();
    }

    *minMedian = s_minMedian[0];
    *minIdx = s_minIdx[0];
}

#undef DIVUP

template<typename T>
__global__ void computeLMedSInliers(
    Param<unsigned> inliers,
    CParam<T> H,
    CParam<float> x_src,
    CParam<float> y_src,
    CParam<float> x_dst,
    CParam<float> y_dst,
    const float minMedian,
    const unsigned nsamples)
{
    unsigned tid = threadIdx.x;
    unsigned bid = blockIdx.x;
    unsigned i = bid * blockDim.x + tid;

    __shared__ T s_H[9];
    __shared__ unsigned s_inliers[256];

    s_inliers[tid] = 0;
    __syncthreads();

    if (tid < 9)
        s_H[tid] = H.ptr[tid];
    __syncthreads();

    float sigma = max(1.4826f * (1 + 5.f/(nsamples - 4)) * (float)sqrt(minMedian), 1e-6f);
    float dist_thr = sq(2.5f * sigma);

    if (i < nsamples) {
        float z =  s_H[6]*x_src.ptr[i] + s_H[7]*y_src.ptr[i] + s_H[8];
        float x = (s_H[0]*x_src.ptr[i] + s_H[1]*y_src.ptr[i] + s_H[2]) / z;
        float y = (s_H[3]*x_src.ptr[i] + s_H[4]*y_src.ptr[i] + s_H[5]) / z;

        float dist = sq(x_dst.ptr[i] - x) + sq(y_dst.ptr[i] - y);
        if (dist <= dist_thr)
            s_inliers[tid] = 1;
    }
    __syncthreads();

    for (unsigned t = 128; t > 0; t >>= 1) {
        if (tid < t)
            s_inliers[tid] += s_inliers[tid + t];
        __syncthreads();
    }

    inliers.ptr[bid] = s_inliers[0];
}

template<typename T>
int computeH(
    Param<T> bestH,
    Param<T> H,
    Param<T> A,
    Param<T> V,
    Param<float> err,
    CParam<float> x_src,
    CParam<float> y_src,
    CParam<float> x_dst,
    CParam<float> y_dst,
    CParam<float> rnd,
    const unsigned iterations,
    const unsigned nsamples,
    const float inlier_thr,
    const af_homography_type htype)
{
    dim3 threads(16, 16);
    dim3 blocks(1, divup(iterations, threads.y));

    // Build linear system and solve SVD
    CUDA_LAUNCH((buildLinearSystem<T>), blocks, threads,
                H, A, V, x_src, y_src, x_dst, y_dst, rnd, iterations);
    POST_LAUNCH_CHECK();

    threads = dim3(256);
    blocks = dim3(divup(iterations, threads.x));

    // Allocate some temporary buffers
    Param<unsigned> idx, inliers;
    Param<float> median;
    inliers.dims[0] = (htype == AF_RANSAC) ? blocks.x : divup(nsamples, threads.x);
    inliers.strides[0] = 1;
    idx.dims[0] = median.dims[0] = blocks.x;
    idx.strides[0] = median.strides[0] = 1;
    for (int k = 1; k < 4; k++) {
        inliers.dims[k] = 1;
        inliers.strides[k] = inliers.dims[k-1] * inliers.strides[k-1];
        idx.dims[k] = median.dims[k] = 1;
        idx.strides[k] = median.strides[k] = idx.dims[k-1] * idx.strides[k-1];
    }
    idx.ptr = memAlloc<unsigned>(idx.dims[3] * idx.strides[3]);
    inliers.ptr = memAlloc<unsigned>(inliers.dims[3] * inliers.strides[3]);
    if (htype == AF_LMEDS)
        median.ptr = memAlloc<float>(median.dims[3] * median.strides[3]);

    // Compute (and for RANSAC, evaluate) homographies
    CUDA_LAUNCH((computeEvalHomography<T>), blocks, threads,
                 inliers, idx, H, err, x_src, y_src, x_dst, y_dst,
                 rnd, iterations, nsamples, inlier_thr, htype);
    POST_LAUNCH_CHECK();

    unsigned inliersH, idxH;
    if (htype == AF_LMEDS) {
        // TODO: Improve this sorting, if the number of iterations is
        // sufficiently large, this can be *very* slow
        kernel::sort0<float, true>(err);

        unsigned minIdx;
        float minMedian;

        // Compute median of every iteration
        CUDA_LAUNCH((computeMedian), blocks, threads,
                    median, idx, err, iterations);
        POST_LAUNCH_CHECK();

        // Reduce medians, only in case iterations > 256
        if (blocks.x > 1) {
            blocks = dim3(1);

            float* finalMedian = memAlloc<float>(1);
            unsigned* finalIdx = memAlloc<unsigned>(1);

            CUDA_LAUNCH((findMinMedian), blocks, threads,
                        finalMedian, finalIdx, median, idx);
            POST_LAUNCH_CHECK();

            CUDA_CHECK(cudaMemcpy(&minMedian, finalMedian, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&minIdx, finalIdx, sizeof(unsigned), cudaMemcpyDeviceToHost));

            memFree(finalMedian);
            memFree(finalIdx);
        }
        else {
            CUDA_CHECK(cudaMemcpy(&minMedian, median.ptr, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&minIdx, idx.ptr, sizeof(unsigned), cudaMemcpyDeviceToHost));
        }

        // Copy best homography to output
        CUDA_CHECK(cudaMemcpy(bestH.ptr, H.ptr + minIdx * 9, 9*sizeof(T), cudaMemcpyDeviceToDevice));

        blocks = dim3(divup(nsamples, threads.x));

        CUDA_LAUNCH((computeLMedSInliers<T>), blocks, threads,
                    inliers, bestH, x_src, y_src, x_dst, y_dst,
                    minMedian, nsamples);
        POST_LAUNCH_CHECK();

        // Adds up the total number of inliers
        Param<unsigned> totalInliers;
        for (int k = 0; k < 4; k++)
            totalInliers.dims[k] = totalInliers.strides[k] = 1;
        totalInliers.ptr = memAlloc<unsigned>(1);

        kernel::reduce<unsigned, unsigned, af_add_t>(totalInliers, inliers, 0, false, 0.0);

        CUDA_CHECK(cudaMemcpy(&inliersH, totalInliers.ptr, sizeof(unsigned), cudaMemcpyDeviceToHost));

        memFree(totalInliers.ptr);
        memFree(median.ptr);
    }
    else if (htype == AF_RANSAC) {
        Param<unsigned> bestInliers, bestIdx;
        for (int k = 0; k < 4; k++) {
            bestInliers.dims[k] = bestIdx.dims[k] = 1;
            bestInliers.strides[k] = bestIdx.strides[k] = 1;
        }
        bestInliers.ptr = memAlloc<unsigned>(1);
        bestIdx.ptr = memAlloc<unsigned>(1);

        kernel::ireduce<unsigned, af_max_t>(bestInliers, bestIdx.ptr, inliers, 0);

        unsigned blockIdx;
        CUDA_CHECK(cudaMemcpy(&blockIdx, bestIdx.ptr, sizeof(unsigned), cudaMemcpyDeviceToHost));

        // Copies back index and number of inliers of best homography estimation
        CUDA_CHECK(cudaMemcpy(&idxH, idx.ptr+blockIdx, sizeof(unsigned), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&inliersH, bestInliers.ptr, sizeof(unsigned), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(bestH.ptr, H.ptr + idxH * 9, 9*sizeof(T), cudaMemcpyDeviceToDevice));

        memFree(bestInliers.ptr);
        memFree(bestIdx.ptr);
    }

    memFree(inliers.ptr);
    memFree(idx.ptr);

    return (int)inliersH;
}

} // namespace kernel

} // namespace cuda

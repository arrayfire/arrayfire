/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/dispatch.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <memory.hpp>
#include "ireduce.hpp"
#include "reduce.hpp"
#include "sort.hpp"

#include <cfloat>

namespace arrayfire {
namespace cuda {

namespace kernel {

template<typename T>
__device__ T sq(T a) {
    return a * a;
}

template<typename T>
struct EPS {
    __device__ T eps() { return FLT_EPSILON; }
};

template<>
struct EPS<float> {
    __device__ static float eps() { return FLT_EPSILON; }
};

template<>
struct EPS<double> {
    __device__ static double eps() { return DBL_EPSILON; }
};

#define RANSACConfidence 0.99f
#define LMEDSConfidence 0.99f
#define LMEDSOutlierRatio 0.4f

extern __shared__ char sh[];

template<typename T>
__device__ void JacobiSVD(int m, int n) {
    const int iterations = 30;

    int tid_x = threadIdx.x;
    int bsz_x = blockDim.x;
    int tid_y = threadIdx.y;
    // int gid_y = blockIdx.y * blockDim.y + tid_y;

    __shared__ T s_acc1[256];
    __shared__ T s_acc2[256];

    __shared__ T s_d[16 * 9];

    T* s_V = (T*)sh;
    T* s_S = (T*)sh + 16 * 81;

    int doff = tid_y * n;
    int soff = tid_y * 81;

    if (tid_x < n) {
        T acc1 = 0;
        for (int i = 0; i < m; i++) {
            int stid = soff + tid_x * m + i;
            T t      = s_S[stid];
            acc1 += t * t;
            s_V[stid] = (tid_x == i) ? 1 : 0;
        }
        s_d[doff + tid_x] = acc1;
    }
    __syncthreads();

    for (int it = 0; it < iterations; it++) {
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                T* Si = s_S + soff + i * m;
                T* Sj = s_S + soff + j * m;

                T* Vi = s_V + soff + i * n;
                T* Vj = s_V + soff + j * n;

                T p = (T)0;
                for (int k = 0; k < m; k++) p += Si[k] * Sj[k];

                T di = s_d[doff + i];
                T dj = s_d[doff + j];
                __syncthreads();

                T c = 0, s = 0;
                T t0 = 0, t1 = 0;
                int cond = (fabs(p) > m * EPS<T>::eps() * sqrt(di * dj));
                T a = 0, b = 0;

                if (cond) {
                    T y  = di - dj;
                    T r  = hypot(p * 2, y);
                    T r2 = r * 2;
                    if (y >= 0) {
                        c = sqrt((r + y) / r2);
                        s = p / (r2 * c);
                    } else {
                        s = sqrt((r - y) / r2);
                        c = p / (r2 * s);
                    }

                    for (int k = tid_x; k < m; k += bsz_x) {
                        t0                        = c * Si[k] + s * Sj[k];
                        t1                        = c * Sj[k] - s * Si[k];
                        Si[k]                     = t0;
                        Sj[k]                     = t1;
                        s_acc1[tid_y * bsz_x + k] = t0 * t0;
                        s_acc2[tid_y * bsz_x + k] = t1 * t1;
                    }
                }
                __syncthreads();

                if (cond) {
                    a = 0;
                    b = 0;
                    for (int k = 0; k < m; k++) {
                        a += s_acc1[tid_y * bsz_x + k];
                        b += s_acc2[tid_y * bsz_x + k];
                    }
                    s_d[doff + i] = a;
                    s_d[doff + j] = b;
                }
                __syncthreads();

                if (cond) {
                    for (int l = tid_x; l < n; l += bsz_x) {
                        T t0 = Vi[l] * c + Vj[l] * s;
                        T t1 = Vj[l] * c - Vi[l] * s;

                        Vi[l] = t0;
                        Vj[l] = t1;
                    }
                }
                __syncthreads();
            }
        }
    }
}

__device__ bool computeMeanScale(
    float* x_src_mean, float* y_src_mean, float* x_dst_mean, float* y_dst_mean,
    float* src_scale, float* dst_scale, float* src_pt_x, float* src_pt_y,
    float* dst_pt_x, float* dst_pt_y, CParam<float> x_src, CParam<float> y_src,
    CParam<float> x_dst, CParam<float> y_dst, CParam<float> rnd, int i) {
    const unsigned ridx = rnd.dims[0] * i;
    unsigned r[4]       = {(unsigned)rnd.ptr[ridx], (unsigned)rnd.ptr[ridx + 1],
                           (unsigned)rnd.ptr[ridx + 2], (unsigned)rnd.ptr[ridx + 3]};

    // If one of the points is repeated, it's a bad samples, will still
    // compute homography to ensure all threads pass __syncthreads()
    bool bad = (r[0] == r[1] || r[0] == r[2] || r[0] == r[3] || r[1] == r[2] ||
                r[1] == r[3] || r[2] == r[3]);

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
        src_var +=
            sq(src_pt_x[j] - *x_src_mean) + sq(src_pt_y[j] - *y_src_mean);
        dst_var +=
            sq(dst_pt_x[j] - *x_dst_mean) + sq(dst_pt_y[j] - *y_dst_mean);
    }

    src_var /= 4.f;
    dst_var /= 4.f;

    *src_scale = sqrt(2.0f) / sqrt(src_var);
    *dst_scale = sqrt(2.0f) / sqrt(dst_var);

    return !bad;
}

#define SSPTR(Z, Y, X) (s_S[(Z)*81 + (Y)*9 + (X)])

template<typename T>
__global__ void buildLinearSystem(Param<T> H, CParam<float> x_src,
                                  CParam<float> y_src, CParam<float> x_dst,
                                  CParam<float> y_dst, CParam<float> rnd,
                                  const unsigned iterations) {
    unsigned tid_y = threadIdx.y;
    unsigned i     = blockIdx.y * blockDim.y + tid_y;

    if (i < iterations) {
        float x_src_mean, y_src_mean;
        float x_dst_mean, y_dst_mean;
        float src_scale, dst_scale;
        float src_pt_x[4], src_pt_y[4], dst_pt_x[4], dst_pt_y[4];

        computeMeanScale(&x_src_mean, &y_src_mean, &x_dst_mean, &y_dst_mean,
                         &src_scale, &dst_scale, src_pt_x, src_pt_y, dst_pt_x,
                         dst_pt_y, x_src, y_src, x_dst, y_dst, rnd, i);

        T* s_V = (T*)sh;
        T* s_S = (T*)sh + 16 * 81;

        // Compute input matrix
        for (unsigned j = threadIdx.x; j < 4; j += blockDim.x) {
            float srcx = (src_pt_x[j] - x_src_mean) * src_scale;
            float srcy = (src_pt_y[j] - y_src_mean) * src_scale;
            float dstx = (dst_pt_x[j] - x_dst_mean) * dst_scale;
            float dsty = (dst_pt_y[j] - y_dst_mean) * dst_scale;

            SSPTR(tid_y, 0, j * 2) = 0.0f;
            SSPTR(tid_y, 1, j * 2) = 0.0f;
            SSPTR(tid_y, 2, j * 2) = 0.0f;
            SSPTR(tid_y, 3, j * 2) = -srcx;
            SSPTR(tid_y, 4, j * 2) = -srcy;
            SSPTR(tid_y, 5, j * 2) = -1.0f;
            SSPTR(tid_y, 6, j * 2) = dsty * srcx;
            SSPTR(tid_y, 7, j * 2) = dsty * srcy;
            SSPTR(tid_y, 8, j * 2) = dsty;

            SSPTR(tid_y, 0, j * 2 + 1) = srcx;
            SSPTR(tid_y, 1, j * 2 + 1) = srcy;
            SSPTR(tid_y, 2, j * 2 + 1) = 1.0f;
            SSPTR(tid_y, 3, j * 2 + 1) = 0.0f;
            SSPTR(tid_y, 4, j * 2 + 1) = 0.0f;
            SSPTR(tid_y, 5, j * 2 + 1) = 0.0f;
            SSPTR(tid_y, 6, j * 2 + 1) = -dstx * srcx;
            SSPTR(tid_y, 7, j * 2 + 1) = -dstx * srcy;
            SSPTR(tid_y, 8, j * 2 + 1) = -dstx;

            if (j == 4) {
                SSPTR(tid_y, 0, 8) = 0.0f;
                SSPTR(tid_y, 1, 8) = 0.0f;
                SSPTR(tid_y, 2, 8) = 0.0f;
                SSPTR(tid_y, 3, 8) = 0.0f;
                SSPTR(tid_y, 4, 8) = 0.0f;
                SSPTR(tid_y, 5, 8) = 0.0f;
                SSPTR(tid_y, 6, 8) = 0.0f;
                SSPTR(tid_y, 7, 8) = 0.0f;
                SSPTR(tid_y, 8, 8) = 0.0f;
            }
        }
        __syncthreads();

        JacobiSVD<T>(9, 9);

        T vH[9], H_tmp[9];
        for (unsigned j = 0; j < 9; j++) vH[j] = s_V[tid_y * 81 + 8 * 9 + j];

        H_tmp[0] =
            src_scale * x_dst_mean * vH[6] + src_scale * vH[0] / dst_scale;
        H_tmp[1] =
            src_scale * x_dst_mean * vH[7] + src_scale * vH[1] / dst_scale;
        H_tmp[2] = x_dst_mean * (vH[8] - src_scale * y_src_mean * vH[7] -
                                 src_scale * x_src_mean * vH[6]) +
                   (vH[2] - src_scale * y_src_mean * vH[1] -
                    src_scale * x_src_mean * vH[0]) /
                       dst_scale;

        H_tmp[3] =
            src_scale * y_dst_mean * vH[6] + src_scale * vH[3] / dst_scale;
        H_tmp[4] =
            src_scale * y_dst_mean * vH[7] + src_scale * vH[4] / dst_scale;
        H_tmp[5] = y_dst_mean * (vH[8] - src_scale * y_src_mean * vH[7] -
                                 src_scale * x_src_mean * vH[6]) +
                   (vH[5] - src_scale * y_src_mean * vH[4] -
                    src_scale * x_src_mean * vH[3]) /
                       dst_scale;

        H_tmp[6] = src_scale * vH[6];
        H_tmp[7] = src_scale * vH[7];
        H_tmp[8] = vH[8] - src_scale * y_src_mean * vH[7] -
                   src_scale * x_src_mean * vH[6];

        const unsigned Hidx = H.dims[0] * i;
        T* H_ptr            = H.ptr + Hidx;
        for (int h = 0; h < 9; h++) H_ptr[h] = H_tmp[h];
    }
}

#undef SSPTR

// LMedS:
// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node25.html
template<typename T>
__global__ void computeEvalHomography(
    Param<unsigned> inliers, Param<unsigned> idx, Param<T> H, Param<float> err,
    CParam<float> x_src, CParam<float> y_src, CParam<float> x_dst,
    CParam<float> y_dst, CParam<float> rnd, const unsigned iterations,
    const unsigned nsamples, const float inlier_thr,
    const af_homography_type htype) {
    unsigned bid_x = blockIdx.x;
    unsigned tid_x = threadIdx.x;
    unsigned i     = bid_x * blockDim.x + tid_x;

    __shared__ unsigned s_inliers[256];
    __shared__ unsigned s_idx[256];

    s_inliers[tid_x] = 0;
    s_idx[tid_x]     = 0;
    __syncthreads();

    if (i < iterations) {
        const unsigned Hidx = H.dims[0] * i;
        T* H_ptr            = H.ptr + Hidx;
        T H_tmp[9];
        for (int h = 0; h < 9; h++) H_tmp[h] = H_ptr[h];

        if (htype == AF_HOMOGRAPHY_RANSAC) {
            // Compute inliers
            unsigned inliers_count = 0;
            for (unsigned j = 0; j < nsamples; j++) {
                float z = H_tmp[6] * x_src.ptr[j] + H_tmp[7] * y_src.ptr[j] +
                          H_tmp[8];
                float x = (H_tmp[0] * x_src.ptr[j] + H_tmp[1] * y_src.ptr[j] +
                           H_tmp[2]) /
                          z;
                float y = (H_tmp[3] * x_src.ptr[j] + H_tmp[4] * y_src.ptr[j] +
                           H_tmp[5]) /
                          z;

                float dist = sq(x_dst.ptr[j] - x) + sq(y_dst.ptr[j] - y);
                if (dist < inlier_thr * inlier_thr) inliers_count++;
            }

            s_inliers[tid_x] = inliers_count;
            s_idx[tid_x]     = i;
        } else if (htype == AF_HOMOGRAPHY_LMEDS) {
            // Compute error
            for (unsigned j = 0; j < nsamples; j++) {
                float z = H_tmp[6] * x_src.ptr[j] + H_tmp[7] * y_src.ptr[j] +
                          H_tmp[8];
                float x = (H_tmp[0] * x_src.ptr[j] + H_tmp[1] * y_src.ptr[j] +
                           H_tmp[2]) /
                          z;
                float y = (H_tmp[3] * x_src.ptr[j] + H_tmp[4] * y_src.ptr[j] +
                           H_tmp[5]) /
                          z;

                float dist = sq(x_dst.ptr[j] - x) + sq(y_dst.ptr[j] - y);
                err.ptr[i * err.dims[0] + j] = sqrt(dist);
            }
        }
    }
    __syncthreads();

    if (htype == AF_HOMOGRAPHY_RANSAC) {
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

__global__ void computeMedian(Param<float> median, Param<unsigned> idx,
                              CParam<float> err, const unsigned iterations) {
    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned i   = bid * blockDim.x + threadIdx.x;

    __shared__ float s_median[256];
    __shared__ unsigned s_idx[256];

    s_median[tid] = FLT_MAX;
    s_idx[tid]    = 0;

    if (i < iterations) {
        const int nsamples = err.dims[0];
        float m            = err.ptr[i * nsamples + nsamples / 2];
        if (nsamples % 2 == 0)
            m = (m + err.ptr[i * nsamples + nsamples / 2 - 1]) * 0.5f;

        s_idx[tid]    = i;
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
    idx.ptr[bid]    = s_idx[0];
}

#define DIVUP(A, B) (((A) + (B)-1) / (B))

__global__ void findMinMedian(float* minMedian, unsigned* minIdx,
                              CParam<float> median, CParam<unsigned> idx) {
    const int tid = threadIdx.x;

    __shared__ float s_minMedian[256];
    __shared__ unsigned s_minIdx[256];

    s_minMedian[tid] = FLT_MAX;
    s_minIdx[tid]    = 0;
    __syncthreads();

    const int loop = DIVUP(median.dims[0], blockDim.x);

    for (int i = 0; i < loop; i++) {
        int j = i * blockDim.x + tid;
        if (j < median.dims[0] && median.ptr[j] < s_minMedian[tid]) {
            s_minMedian[tid] = median.ptr[j];
            s_minIdx[tid]    = idx.ptr[j];
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
    *minIdx    = s_minIdx[0];
}

#undef DIVUP

template<typename T>
__global__ void computeLMedSInliers(Param<unsigned> inliers, CParam<T> H,
                                    CParam<float> x_src, CParam<float> y_src,
                                    CParam<float> x_dst, CParam<float> y_dst,
                                    const float minMedian,
                                    const unsigned nsamples) {
    unsigned tid = threadIdx.x;
    unsigned bid = blockIdx.x;
    unsigned i   = bid * blockDim.x + tid;

    __shared__ T s_H[9];
    __shared__ unsigned s_inliers[256];

    s_inliers[tid] = 0;
    __syncthreads();

    if (tid < 9) s_H[tid] = H.ptr[tid];
    __syncthreads();

    float sigma = max(
        1.4826f * (1 + 5.f / (nsamples - 4)) * (float)sqrt(minMedian), 1e-6f);
    float dist_thr = sq(2.5f * sigma);

    if (i < nsamples) {
        float z = s_H[6] * x_src.ptr[i] + s_H[7] * y_src.ptr[i] + s_H[8];
        float x = (s_H[0] * x_src.ptr[i] + s_H[1] * y_src.ptr[i] + s_H[2]) / z;
        float y = (s_H[3] * x_src.ptr[i] + s_H[4] * y_src.ptr[i] + s_H[5]) / z;

        float dist = sq(x_dst.ptr[i] - x) + sq(y_dst.ptr[i] - y);
        if (dist <= dist_thr) s_inliers[tid] = 1;
    }
    __syncthreads();

    for (unsigned t = 128; t > 0; t >>= 1) {
        if (tid < t) s_inliers[tid] += s_inliers[tid + t];
        __syncthreads();
    }

    inliers.ptr[bid] = s_inliers[0];
}

template<typename T>
int computeH(Param<T> bestH, Param<T> H, Param<float> err, CParam<float> x_src,
             CParam<float> y_src, CParam<float> x_dst, CParam<float> y_dst,
             CParam<float> rnd, const unsigned iterations,
             const unsigned nsamples, const float inlier_thr,
             const af_homography_type htype) {
    dim3 threads(16, 16);
    dim3 blocks(1, divup(iterations, threads.y));

    // Build linear system and solve SVD
    size_t ls_shared_sz = threads.x * 81 * 2 * sizeof(T);
    CUDA_LAUNCH_SMEM((buildLinearSystem<T>), blocks, threads, ls_shared_sz, H,
                     x_src, y_src, x_dst, y_dst, rnd, iterations);
    POST_LAUNCH_CHECK();

    threads = dim3(256);
    blocks  = dim3(divup(iterations, threads.x));

    // Allocate some temporary buffers
    dim4 idx_dims(blocks.x);
    Array<unsigned> idx     = createEmptyArray<unsigned>(idx_dims);
    Array<unsigned> inliers = createEmptyArray<unsigned>(
        (htype == AF_HOMOGRAPHY_RANSAC) ? blocks.x
                                        : divup(nsamples, threads.x));

    // Compute (and for RANSAC, evaluate) homographies
    CUDA_LAUNCH((computeEvalHomography<T>), blocks, threads, inliers, idx, H,
                err, x_src, y_src, x_dst, y_dst, rnd, iterations, nsamples,
                inlier_thr, htype);
    POST_LAUNCH_CHECK();

    unsigned inliersH, idxH;
    if (htype == AF_HOMOGRAPHY_LMEDS) {
        Array<float> median = createEmptyArray<float>(idx_dims);
        // TODO: Improve this sorting, if the number of iterations is
        // sufficiently large, this can be *very* slow
        kernel::sort0<float>(err, true);

        unsigned minIdx;
        float minMedian;

        // Compute median of every iteration
        CUDA_LAUNCH((computeMedian), blocks, threads, median, idx, err,
                    iterations);
        POST_LAUNCH_CHECK();

        // Reduce medians, only in case iterations > 256
        if (blocks.x > 1) {
            blocks = dim3(1);

            auto finalMedian = memAlloc<float>(1);
            auto finalIdx    = memAlloc<unsigned>(1);

            CUDA_LAUNCH((findMinMedian), blocks, threads, finalMedian.get(),
                        finalIdx.get(), median, idx);
            POST_LAUNCH_CHECK();

            CUDA_CHECK(cudaMemcpyAsync(&minMedian, finalMedian.get(),
                                       sizeof(float), cudaMemcpyDeviceToHost,
                                       getActiveStream()));
            CUDA_CHECK(cudaMemcpyAsync(&minIdx, finalIdx.get(),
                                       sizeof(unsigned), cudaMemcpyDeviceToHost,
                                       getActiveStream()));
            CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(&minMedian, median.get(), sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       getActiveStream()));
            CUDA_CHECK(cudaMemcpyAsync(&minIdx, idx.get(), sizeof(unsigned),
                                       cudaMemcpyDeviceToHost,
                                       getActiveStream()));
            CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
        }

        // Copy best homography to output
        CUDA_CHECK(cudaMemcpyAsync(bestH.ptr, H.ptr + minIdx * 9, 9 * sizeof(T),
                                   cudaMemcpyDeviceToDevice,
                                   getActiveStream()));

        blocks = dim3(divup(nsamples, threads.x));
        // sync stream for the device to host copies to be visible for
        // the subsequent kernel launch

        CUDA_LAUNCH((computeLMedSInliers<T>), blocks, threads, inliers, bestH,
                    x_src, y_src, x_dst, y_dst, minMedian, nsamples);
        POST_LAUNCH_CHECK();

        // Adds up the total number of inliers
        Array<unsigned> totalInliers = createEmptyArray<unsigned>(1);
        kernel::reduce<unsigned, unsigned, af_add_t>(totalInliers, inliers, 0,
                                                     false, 0.0);

        CUDA_CHECK(cudaMemcpyAsync(&inliersH, totalInliers.get(),
                                   sizeof(unsigned), cudaMemcpyDeviceToHost,
                                   getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));

    } else if (htype == AF_HOMOGRAPHY_RANSAC) {
        unsigned blockIdx;
        inliersH = kernel::ireduce_all<unsigned, af_max_t>(&blockIdx, inliers);
        // Copies back index and number of inliers of best homography estimation
        CUDA_CHECK(cudaMemcpyAsync(&idxH, idx.get() + blockIdx,
                                   sizeof(unsigned), cudaMemcpyDeviceToHost,
                                   getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
        CUDA_CHECK(cudaMemcpyAsync(bestH.ptr, H.ptr + idxH * 9, 9 * sizeof(T),
                                   cudaMemcpyDeviceToDevice,
                                   getActiveStream()));
    }

    // sync stream for the device to host copies to be visible for
    // the subsequent kernel launch
    CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));

    return (int)inliersH;
}

}  // namespace kernel

}  // namespace cuda
}  // namespace arrayfire

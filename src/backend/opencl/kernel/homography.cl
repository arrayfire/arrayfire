/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

inline T sq(T a)
{
    return a * a;
}

inline void jacobi_svd(__global T* S, __global T* V, int m, int n,
                       __local T* l_acc1, __local T* l_acc2, __local T* l_S,
                       __local T* l_V, __local T* l_d)
{
    const int iterations = 30;

    int tid_x = get_local_id(0);
    int bsz_x = get_local_size(0);
    int tid_y = get_local_id(1);
    int gid_y = get_global_id(1);

    for (int k = 0; k <= 4; k++)
        l_S[tid_y * 81 + k*bsz_x + tid_x] = S[gid_y * 81 + k*bsz_x + tid_x];
    if (tid_x == 0)
        l_S[tid_y * 81 + 80] = S[gid_y * 81 + 80];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Copy first 80 elements
    T t = l_S[tid_y*81 + tid_x];
    l_acc1[tid_y*bsz_x + tid_x] = t*t;
    for (int i = 1; i <= 4; i++) {
        T t = l_S[tid_y*81 + tid_x+i*bsz_x];
        l_acc1[tid_y*bsz_x + tid_x] += t*t;
    }
    if (tid_x < 8)
        l_acc1[tid_y*16 + tid_x] += l_acc1[tid_y*16 + tid_x+8];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid_x < 4)
        l_acc1[tid_y*16 + tid_x] += l_acc1[tid_y*16 + tid_x+4];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid_x < 2)
        l_acc1[tid_y*16 + tid_x] += l_acc1[tid_y*16 + tid_x+2];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid_x < 1) {
        // Copy last element
        T t = l_S[tid_y*bsz_x + tid_x+80];
        l_acc1[tid_y*16 + tid_x] += l_acc1[tid_y*16 + tid_x+1] + t*t;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid_x < n)
        l_d[tid_y*9 + tid_x] = l_acc1[tid_y*bsz_x + tid_x];

    // V is initialized as an identity matrix
    for (int i = 0; i <= 4; i++) {
        l_V[tid_y*81 + i*bsz_x + tid_x] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid_x < m)
        l_V[tid_y*81 + tid_x*m + tid_x] = 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int it = 0; it < iterations; it++) {
        int converged = 0;

        for (int i = 0; i < n-1; i++) {
            for (int j = i+1; j < n; j++) {
                __local T* Si = l_S + tid_y*81 + i*m;
                __local T* Sj = l_S + tid_y*81 + j*m;

                T p = (T)0;
                for (int k = 0; k < m; k++)
                    p += Si[k]*Sj[k];

                T c = 0, s = 0;

                int cond = (fabs(p) > EPS*sqrt(l_d[tid_y*9 + i]*l_d[tid_y*9 + j]));
                if (cond) {
                    T y = l_d[tid_y*9 + i] - l_d[tid_y*9 + j];
                    T r = hypot(p*2, y);
                    T r2 = r*2;
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

                        l_acc1[tid_y*16 + tid_x] = t0*t0;
                        l_acc2[tid_y*16 + tid_x] = t1*t1;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if (cond && tid_x < 4) {
                    l_acc1[tid_y*16 + tid_x] += l_acc1[tid_y*16 + tid_x+4];
                    l_acc2[tid_y*16 + tid_x] += l_acc2[tid_y*16 + tid_x+4];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                if (cond && tid_x < 2) {
                    l_acc1[tid_y*16 + tid_x] += l_acc1[tid_y*16 + tid_x+2];
                    l_acc2[tid_y*16 + tid_x] += l_acc2[tid_y*16 + tid_x+2];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                if (cond && tid_x < 1) {
                    l_acc1[tid_y*16 + tid_x] += l_acc1[tid_y*16 + tid_x+1] + l_acc1[tid_y*16 + tid_x+8];
                    l_acc2[tid_y*16 + tid_x] += l_acc2[tid_y*16 + tid_x+1] + l_acc2[tid_y*16 + tid_x+8];
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if (cond && tid_x == 0) {
                    l_d[tid_y*9 + i] = l_acc1[tid_y*16];
                    l_d[tid_y*9 + j] = l_acc2[tid_y*16];
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                __local T* Vi = l_V + tid_y*81 + i*n;
                __local T* Vj = l_V + tid_y*81 + j*n;

                if (cond && tid_x < n) {
                    T t0 = Vi[tid_x] * c + Vj[tid_x] * s;
                    T t1 = Vj[tid_x] * c - Vi[tid_x] * s;

                    Vi[tid_x] = t0;
                    Vj[tid_x] = t1;
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                converged = 1;
            }
            if (converged == 0)
                break;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i <= 4; i++)
        V[gid_y * 81 + tid_x+i*bsz_x] = l_V[tid_y * 81 + tid_x+i*bsz_x];
    if (tid_x == 0)
        V[gid_y * 81 + 80] = l_V[tid_y * 81 + 80];
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline int compute_mean_scale(
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
    __global const float* x_src,
    __global const float* y_src,
    __global const float* x_dst,
    __global const float* y_dst,
    __global const float* rnd,
    KParam rInfo,
    int i)
{
    const unsigned ridx = rInfo.dims[0] * i;
    unsigned r[4] = { (unsigned)rnd[ridx],
                      (unsigned)rnd[ridx+1],
                      (unsigned)rnd[ridx+2],
                      (unsigned)rnd[ridx+3] };

    // If one of the points is repeated, it's a bad samples, will still
    // compute homography to ensure all threads pass barrier()
    int bad = (r[0] == r[1] || r[0] == r[2] || r[0] == r[3] ||
               r[1] == r[2] || r[1] == r[3] || r[2] == r[3]);

    for (unsigned j = 0; j < 4; j++) {
        src_pt_x[j] = x_src[r[j]];
        src_pt_y[j] = y_src[r[j]];
        dst_pt_x[j] = x_dst[r[j]];
        dst_pt_y[j] = y_dst[r[j]];
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

#define APTR(Z, Y, X) (A[(Z) * AInfo.dims[0] * AInfo.dims[1] + (Y) * AInfo.dims[0] + (X)])

__kernel void compute_homography(
    __global T* H,
    KParam HInfo,
    __global T* A,
    KParam AInfo,
    __global T* V,
    KParam VInfo,
    __global const float* x_src,
    __global const float* y_src,
    __global const float* x_dst,
    __global const float* y_dst,
    __global const float* rnd,
    KParam rInfo,
    const unsigned iterations)
{
    unsigned i = get_global_id(1);

    float x_src_mean, y_src_mean;
    float x_dst_mean, y_dst_mean;
    float src_scale, dst_scale;
    float src_pt_x[4], src_pt_y[4], dst_pt_x[4], dst_pt_y[4];

    compute_mean_scale(&x_src_mean, &y_src_mean,
                       &x_dst_mean, &y_dst_mean,
                       &src_scale, &dst_scale,
                       src_pt_x, src_pt_y,
                       dst_pt_x, dst_pt_y,
                       x_src, y_src, x_dst, y_dst,
                       rnd, rInfo, i);

    // Compute input matrix
    for (unsigned j = get_local_id(0); j < 4; j+=get_local_size(0)) {
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

    __local T l_acc1[256];
    __local T l_acc2[256];

    __local T l_S[16*81];
    __local T l_V[16*81];
    __local T l_d[16*9];

    jacobi_svd(A, V, 9, 9, l_acc1, l_acc2, l_S, l_V, l_d);

    T vH[9], H_tmp[9];
    for (unsigned j = 0; j < 9; j++)
        vH[j] = V[i * VInfo.dims[0] * VInfo.dims[1] + 8 * VInfo.dims[0] + j];

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

    const unsigned Hidx = HInfo.dims[0] * i;
    __global T* H_ptr = H + Hidx;
    for (int h = 0; h < 9; h++)
        H_ptr[h] = H_tmp[h];
}

#undef APTR

// LMedS: http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node25.html
__kernel void eval_homography(
    __global unsigned* inliers,
    __global unsigned* idx,
    __global T* H,
    KParam HInfo,
    __global float* err,
    KParam eInfo,
    __global const float* x_src,
    __global const float* y_src,
    __global const float* x_dst,
    __global const float* y_dst,
    __global const float* rnd,
    const unsigned iterations,
    const unsigned nsamples,
    const float inlier_thr)
{
    unsigned tid_x = get_local_id(0);
    unsigned i = get_global_id(0);

    __local unsigned l_inliers[256];
    __local unsigned l_idx[256];

    l_inliers[tid_x] = 0;
    l_idx[tid_x]     = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < iterations) {
        const unsigned Hidx = HInfo.dims[0] * i;
        __global T* H_ptr = H + Hidx;
        T H_tmp[9];
        for (int h = 0; h < 9; h++)
            H_tmp[h] = H_ptr[h];

#ifdef RANSAC
        // Compute inliers
        unsigned inliers_count = 0;
        for (unsigned j = 0; j < nsamples; j++) {
            float z =  H_tmp[6]*x_src[j] + H_tmp[7]*y_src[j] + H_tmp[8];
            float x = (H_tmp[0]*x_src[j] + H_tmp[1]*y_src[j] + H_tmp[2]) / z;
            float y = (H_tmp[3]*x_src[j] + H_tmp[4]*y_src[j] + H_tmp[5]) / z;

            float dist = sq(x_dst[j] - x) + sq(y_dst[j] - y);
            if (dist < inlier_thr*inlier_thr)
                inliers_count++;
        }

        l_inliers[tid_x] = inliers_count;
        l_idx[tid_x]     = i;
#endif
#ifdef LMEDS
        // Compute error
        for (unsigned j = 0; j < nsamples; j++) {
            float z =  H_tmp[6]*x_src[j] + H_tmp[7]*y_src[j] + H_tmp[8];
            float x = (H_tmp[0]*x_src[j] + H_tmp[1]*y_src[j] + H_tmp[2]) / z;
            float y = (H_tmp[3]*x_src[j] + H_tmp[4]*y_src[j] + H_tmp[5]) / z;

            float dist = sq(x_dst[j] - x) + sq(y_dst[j] - y);
            err[i*eInfo.dims[0] + j] = sqrt(dist);
        }
#endif
    }

#ifdef RANSAC
    unsigned bid_x = get_group_id(0);

    // Find sample with most inliers
    for (unsigned tx = 128; tx > 0; tx >>= 1) {
        if (tid_x < tx) {
            if (l_inliers[tid_x + tx] > l_inliers[tid_x]) {
                l_inliers[tid_x] = l_inliers[tid_x + tx];
                l_idx[tid_x]     = l_idx[tid_x + tx];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    inliers[bid_x] = l_inliers[0];
    idx[bid_x]     = l_idx[0];
#endif
}

__kernel void compute_median(
    __global float* median,
    __global unsigned* idx,
    __global const float* err,
    KParam eInfo,
    const unsigned iterations)
{
    const unsigned tid = get_local_id(0);
    const unsigned bid = get_group_id(0);
    const unsigned i = get_global_id(0);

    __local float l_median[256];
    __local unsigned l_idx[256];

    l_median[tid] = FLT_MAX;
    l_idx[tid] = 0;

    if (i < iterations) {
        const int nsamples = eInfo.dims[0];
        float m = err[i*nsamples + nsamples / 2];
        if (nsamples % 2 == 0)
            m = (m + err[i*nsamples + nsamples / 2 - 1]) * 0.5f;

        l_idx[tid] = i;
        l_median[tid] = m;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned t = 128; t > 0; t >>= 1) {
        if (tid < t) {
            if (l_median[tid + t] < l_median[tid]) {
                l_median[tid] = l_median[tid + t];
                l_idx[tid]    = l_idx[tid + t];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    median[bid] = l_median[0];
    idx[bid] = l_idx[0];
}

#define DIVUP(A, B) (((A) + (B) - 1) / (B))

__kernel void find_min_median(
    __global float* minMedian,
    __global unsigned* minIdx,
    __global const float* median,
    KParam mInfo,
    __global const unsigned* idx)
{
    const unsigned tid = get_local_id(0);

    __local float l_minMedian[256];
    __local unsigned l_minIdx[256];

    l_minMedian[tid] = FLT_MAX;
    l_minIdx[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const int loop = DIVUP(mInfo.dims[0], get_local_size(0));

    for (int i = 0; i < loop; i++) {
        int j = i * get_local_size(0) + tid;
        if (j < mInfo.dims[0] && median[j] < l_minMedian[tid]) {
            l_minMedian[tid] = median[j];
            l_minIdx[tid] = idx[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (unsigned t = 128; t > 0; t >>= 1) {
        if (tid < t) {
            if (l_minMedian[tid + t] < l_minMedian[tid]) {
                l_minMedian[tid] = l_minMedian[tid + t];
                l_minIdx[tid]    = l_minIdx[tid + t];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    *minMedian = l_minMedian[0];
    *minIdx = l_minIdx[0];
}

#undef DIVUP

__kernel void compute_lmeds_inliers(
    __global unsigned* inliers,
    __global const T* H,
    __global const float* x_src,
    __global const float* y_src,
    __global const float* x_dst,
    __global const float* y_dst,
    const float minMedian,
    const unsigned nsamples)
{
    unsigned tid = get_local_id(0);
    unsigned bid = get_group_id(0);
    unsigned i = get_global_id(0);

    __local T l_H[9];
    __local unsigned l_inliers[256];

    l_inliers[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 9)
        l_H[tid] = H[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    float sigma = fmax(1.4826f * (1 + 5.f/(nsamples - 4)) * (float)sqrt(minMedian), 1e-6f);
    float dist_thr = sq(2.5f * sigma);

    if (i < nsamples) {
        float z =  l_H[6]*x_src[i] + l_H[7]*y_src[i] + l_H[8];
        float x = (l_H[0]*x_src[i] + l_H[1]*y_src[i] + l_H[2]) / z;
        float y = (l_H[3]*x_src[i] + l_H[4]*y_src[i] + l_H[5]) / z;

        float dist = sq(x_dst[i] - x) + sq(y_dst[i] - y);
        if (dist <= dist_thr)
            l_inliers[tid] = 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned t = 128; t > 0; t >>= 1) {
        if (tid < t)
            l_inliers[tid] += l_inliers[tid + t];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    inliers[bid] = l_inliers[0];
}

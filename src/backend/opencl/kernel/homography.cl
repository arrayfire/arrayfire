/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

inline T sq(T a) { return a * a; }

inline void jacobi_svd(local T* l_V, __local T* l_S, __local T* l_d,
                       local T* l_acc1, __local T* l_acc2, int m, int n) {
    const int iterations = 30;

    int tid_x = get_local_id(0);
    int bsz_x = get_local_size(0);
    int tid_y = get_local_id(1);
    int gid_y = get_global_id(1);

    int doff = tid_y * n;
    int soff = tid_y * 81;

    if (tid_x < n) {
        T acc1 = 0;
        for (int i = 0; i < m; i++) {
            int stid = soff + tid_x * m + i;
            T t      = l_S[stid];
            acc1 += t * t;
            l_V[stid] = (tid_x == i) ? 1 : 0;
        }
        l_d[doff + tid_x] = acc1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if defined(IS_CPU)
    // All threads do the same work
    // FIXME: Figure out why code below doesnt work
    int tst = 0, toff = 1, tcond = tid_x == 0;
#define BARRIER  // nothing
#else
    // Split work across subgroup
    int tst = tid_x, toff = bsz_x, tcond = 1;
#define BARRIER barrier(CLK_LOCAL_MEM_FENCE)
#endif

    for (int it = 0; tcond && it < iterations; it++) {
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                local T* Si = l_S + soff + i * m;
                local T* Sj = l_S + soff + j * m;

                local T* Vi = l_V + soff + i * n;
                local T* Vj = l_V + soff + j * n;

                T p = (T)0;
                for (int k = 0; k < m; k++) p += Si[k] * Sj[k];

                T di = l_d[doff + i];
                T dj = l_d[doff + j];
                BARRIER;

                T c = 0, s = 0;
                T t0 = 0, t1 = 0;
                int cond = (fabs(p) > m * EPS * sqrt(di * dj));
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

                    for (int k = tst; k < m; k += toff) {
                        t0                        = c * Si[k] + s * Sj[k];
                        t1                        = c * Sj[k] - s * Si[k];
                        Si[k]                     = t0;
                        Sj[k]                     = t1;
                        l_acc1[tid_y * bsz_x + k] = t0 * t0;
                        l_acc2[tid_y * bsz_x + k] = t1 * t1;
                    }
                }
                BARRIER;

                if (cond) {
                    a = 0;
                    b = 0;
                    for (int k = 0; k < m; k++) {
                        a += l_acc1[tid_y * bsz_x + k];
                        b += l_acc2[tid_y * bsz_x + k];
                    }
                    l_d[doff + i] = a;
                    l_d[doff + j] = b;
                }
                BARRIER;

                if (cond) {
                    for (int l = tst; l < n; l += toff) {
                        T t0 = Vi[l] * c + Vj[l] * s;
                        T t1 = Vj[l] * c - Vi[l] * s;

                        Vi[l] = t0;
                        Vj[l] = t1;
                    }
                }
                BARRIER;
            }
        }
    }
}

inline int compute_mean_scale(float* x_src_mean, float* y_src_mean,
                              float* x_dst_mean, float* y_dst_mean,
                              float* src_scale, float* dst_scale,
                              float* src_pt_x, float* src_pt_y, float* dst_pt_x,
                              float* dst_pt_y, global const float* x_src,
                              global const float* y_src,
                              global const float* x_dst,
                              global const float* y_dst,
                              global const float* rnd, KParam rInfo, int i) {
    const unsigned ridx = rInfo.dims[0] * i;
    unsigned r[4]       = {(unsigned)rnd[ridx], (unsigned)rnd[ridx + 1],
                     (unsigned)rnd[ridx + 2], (unsigned)rnd[ridx + 3]};

    // If one of the points is repeated, it's a bad samples, will still
    // compute homography to ensure all threads pass barrier()
    int bad = (r[0] == r[1] || r[0] == r[2] || r[0] == r[3] || r[1] == r[2] ||
               r[1] == r[3] || r[2] == r[3]);

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
        src_var +=
            sq(src_pt_x[j] - *x_src_mean) + sq(src_pt_y[j] - *y_src_mean);
        dst_var +=
            sq(dst_pt_x[j] - *x_dst_mean) + sq(dst_pt_y[j] - *y_dst_mean);
    }

    src_var /= 4.f;
    dst_var /= 4.f;

    *src_scale = sqrt(2.0f) / sqrt(src_var);
    *dst_scale = sqrt(2.0f) / sqrt(dst_var);

    return bad;
}

#define LSPTR(Z, Y, X) (l_S[(Z)*81 + (Y)*9 + (X)])

kernel void compute_homography(global T* H, KParam HInfo,
                                 global const float* x_src,
                                 global const float* y_src,
                                 global const float* x_dst,
                                 global const float* y_dst,
                                 global const float* rnd, KParam rInfo,
                                 const unsigned iterations) {
    unsigned i     = get_global_id(1);
    unsigned tid_y = get_local_id(1);
    unsigned tid_x = get_local_id(0);

    float x_src_mean, y_src_mean;
    float x_dst_mean, y_dst_mean;
    float src_scale, dst_scale;
    float src_pt_x[4], src_pt_y[4], dst_pt_x[4], dst_pt_y[4];

    int bad =
        compute_mean_scale(&x_src_mean, &y_src_mean, &x_dst_mean, &y_dst_mean,
                           &src_scale, &dst_scale, src_pt_x, src_pt_y, dst_pt_x,
                           dst_pt_y, x_src, y_src, x_dst, y_dst, rnd, rInfo, i);

    local T l_acc1[256];
    local T l_acc2[256];

    local T l_S[16 * 81];
    local T l_V[16 * 81];
    local T l_d[16 * 9];

    // Compute input matrix
    if (tid_x < 4) {
        float srcx = (src_pt_x[tid_x] - x_src_mean) * src_scale;
        float srcy = (src_pt_y[tid_x] - y_src_mean) * src_scale;
        float dstx = (dst_pt_x[tid_x] - x_dst_mean) * dst_scale;
        float dsty = (dst_pt_y[tid_x] - y_dst_mean) * dst_scale;

        LSPTR(tid_y, 0, tid_x * 2) = 0.0f;
        LSPTR(tid_y, 1, tid_x * 2) = 0.0f;
        LSPTR(tid_y, 2, tid_x * 2) = 0.0f;
        LSPTR(tid_y, 3, tid_x * 2) = -srcx;
        LSPTR(tid_y, 4, tid_x * 2) = -srcy;
        LSPTR(tid_y, 5, tid_x * 2) = -1.0f;
        LSPTR(tid_y, 6, tid_x * 2) = dsty * srcx;
        LSPTR(tid_y, 7, tid_x * 2) = dsty * srcy;
        LSPTR(tid_y, 8, tid_x * 2) = dsty;

        LSPTR(tid_y, 0, tid_x * 2 + 1) = srcx;
        LSPTR(tid_y, 1, tid_x * 2 + 1) = srcy;
        LSPTR(tid_y, 2, tid_x * 2 + 1) = 1.0f;
        LSPTR(tid_y, 3, tid_x * 2 + 1) = 0.0f;
        LSPTR(tid_y, 4, tid_x * 2 + 1) = 0.0f;
        LSPTR(tid_y, 5, tid_x * 2 + 1) = 0.0f;
        LSPTR(tid_y, 6, tid_x * 2 + 1) = -dstx * srcx;
        LSPTR(tid_y, 7, tid_x * 2 + 1) = -dstx * srcy;
        LSPTR(tid_y, 8, tid_x * 2 + 1) = -dstx;

        if (tid_x == 4) {
            LSPTR(tid_y, 0, 8) = 0.0f;
            LSPTR(tid_y, 1, 8) = 0.0f;
            LSPTR(tid_y, 2, 8) = 0.0f;
            LSPTR(tid_y, 3, 8) = 0.0f;
            LSPTR(tid_y, 4, 8) = 0.0f;
            LSPTR(tid_y, 5, 8) = 0.0f;
            LSPTR(tid_y, 6, 8) = 0.0f;
            LSPTR(tid_y, 7, 8) = 0.0f;
            LSPTR(tid_y, 8, 8) = 0.0f;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    jacobi_svd(l_V, l_S, l_d, l_acc1, l_acc2, 9, 9);

    if (i < HInfo.dims[1] && tid_x == 0) {
        T vH[9], H_tmp[9];
        for (unsigned j = 0; j < 9; j++) vH[j] = l_V[tid_y * 81 + 8 * 9 + j];

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

        const unsigned Hidx = HInfo.dims[0] * i;
        global T* H_ptr   = H + Hidx;
        for (int h = 0; h < 9; h++) H_ptr[h] = bad ? 0 : H_tmp[h];
    }
}

#undef APTR

// LMedS:
// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node25.html
kernel void eval_homography(
    global unsigned* inliers, __global unsigned* idx, __global T* H,
    KParam HInfo, global float* err, KParam eInfo,
    global const float* x_src, __global const float* y_src,
    global const float* x_dst, __global const float* y_dst,
    global const float* rnd, const unsigned iterations,
    const unsigned nsamples, const float inlier_thr) {
    unsigned tid_x = get_local_id(0);
    unsigned i     = get_global_id(0);

    local unsigned l_inliers[256];
    local unsigned l_idx[256];

    l_inliers[tid_x] = 0;
    l_idx[tid_x]     = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < iterations) {
        const unsigned Hidx = HInfo.dims[0] * i;
        global T* H_ptr   = H + Hidx;
        T H_tmp[9];
        for (int h = 0; h < 9; h++) H_tmp[h] = H_ptr[h];

#ifdef RANSAC
        // Compute inliers
        unsigned inliers_count = 0;
        for (unsigned j = 0; j < nsamples; j++) {
            float z = H_tmp[6] * x_src[j] + H_tmp[7] * y_src[j] + H_tmp[8];
            float x =
                (H_tmp[0] * x_src[j] + H_tmp[1] * y_src[j] + H_tmp[2]) / z;
            float y =
                (H_tmp[3] * x_src[j] + H_tmp[4] * y_src[j] + H_tmp[5]) / z;

            float dist = sq(x_dst[j] - x) + sq(y_dst[j] - y);
            if (dist < inlier_thr * inlier_thr) inliers_count++;
        }

        l_inliers[tid_x] = inliers_count;
        l_idx[tid_x]     = i;
#endif
#ifdef LMEDS
        // Compute error
        for (unsigned j = 0; j < nsamples; j++) {
            float z = H_tmp[6] * x_src[j] + H_tmp[7] * y_src[j] + H_tmp[8];
            float x =
                (H_tmp[0] * x_src[j] + H_tmp[1] * y_src[j] + H_tmp[2]) / z;
            float y =
                (H_tmp[3] * x_src[j] + H_tmp[4] * y_src[j] + H_tmp[5]) / z;

            float dist                 = sq(x_dst[j] - x) + sq(y_dst[j] - y);
            err[i * eInfo.dims[0] + j] = sqrt(dist);
        }
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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

    if (tid_x == 0) {
        inliers[bid_x] = l_inliers[0];
        idx[bid_x]     = l_idx[0];
    }
#endif
}

kernel void compute_median(global float* median, __global unsigned* idx,
                             global const float* err, KParam eInfo,
                             const unsigned iterations) {
    const unsigned tid = get_local_id(0);
    const unsigned bid = get_group_id(0);
    const unsigned i   = get_global_id(0);

    local float l_median[256];
    local unsigned l_idx[256];

    l_median[tid] = FLT_MAX;
    l_idx[tid]    = 0;

    if (i < iterations) {
        const int nsamples = eInfo.dims[0];
        float m            = err[i * nsamples + nsamples / 2];
        if (nsamples % 2 == 0)
            m = (m + err[i * nsamples + nsamples / 2 - 1]) * 0.5f;

        l_idx[tid]    = i;
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
    idx[bid]    = l_idx[0];
}

#define DIVUP(A, B) (((A) + (B)-1) / (B))

kernel void find_min_median(global float* minMedian,
                              global unsigned* minIdx,
                              global const float* median, KParam mInfo,
                              global const unsigned* idx) {
    const unsigned tid = get_local_id(0);

    local float l_minMedian[256];
    local unsigned l_minIdx[256];

    l_minMedian[tid] = FLT_MAX;
    l_minIdx[tid]    = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const int loop = DIVUP(mInfo.dims[0], get_local_size(0));

    for (int i = 0; i < loop; i++) {
        int j = i * get_local_size(0) + tid;
        if (j < mInfo.dims[0] && median[j] < l_minMedian[tid]) {
            l_minMedian[tid] = median[j];
            l_minIdx[tid]    = idx[j];
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
    *minIdx    = l_minIdx[0];
}

#undef DIVUP

kernel void compute_lmeds_inliers(
    global unsigned* inliers, __global const T* H,
    global const float* x_src, __global const float* y_src,
    global const float* x_dst, __global const float* y_dst,
    const float minMedian, const unsigned nsamples) {
    unsigned tid = get_local_id(0);
    unsigned bid = get_group_id(0);
    unsigned i   = get_global_id(0);

    local T l_H[9];
    local unsigned l_inliers[256];

    l_inliers[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 9) l_H[tid] = H[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    float sigma = fmax(
        1.4826f * (1 + 5.f / (nsamples - 4)) * (float)sqrt(minMedian), 1e-6f);
    float dist_thr = sq(2.5f * sigma);

    if (i < nsamples) {
        float z = l_H[6] * x_src[i] + l_H[7] * y_src[i] + l_H[8];
        float x = (l_H[0] * x_src[i] + l_H[1] * y_src[i] + l_H[2]) / z;
        float y = (l_H[3] * x_src[i] + l_H[4] * y_src[i] + l_H[5]) / z;

        float dist = sq(x_dst[i] - x) + sq(y_dst[i] - y);
        if (dist <= dist_thr) l_inliers[tid] = 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned t = 128; t > 0; t >>= 1) {
        if (tid < t) l_inliers[tid] += l_inliers[tid + t];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    inliers[bid] = l_inliers[0];
}

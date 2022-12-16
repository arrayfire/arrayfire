/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <arith.hpp>
#include <err_cpu.hpp>
#include <homography.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using af::dim4;
using std::abs;
using std::array;
using std::log;
using std::max;
using std::min;
using std::numeric_limits;
using std::pow;
using std::round;
using std::sqrt;
using std::vector;

namespace arrayfire {
namespace cpu {

template<typename T>
T sq(T a) {
    return a * a;
}

#define APTR(Y, X) (A_ptr[(Y)*Adims[0] + (X)])

static const float RANSACConfidence  = 0.99f;
static const float LMEDSConfidence   = 0.99f;
static const float LMEDSOutlierRatio = 0.4f;

template<typename T>
struct EPS {
    T eps() { return numeric_limits<float>::epsilon(); }
};

template<>
struct EPS<float> {
    static float eps() { return numeric_limits<float>::epsilon(); }
};

template<>
struct EPS<double> {
    static double eps() { return numeric_limits<double>::epsilon(); }
};

template<typename T, int M, int N>
void JacobiSVD(T* S, T* V) {
    const int iterations = 30;
    array<T, N> d{};

    for (int i = 0; i < N; i++) {
        T sd = 0;
        for (int j = 0; j < M; j++) {
            T t = S[i * M + j];
            sd += t * t;
        }
        d[i] = sd;

        V[i * N + i] = 1;
    }

    for (int it = 0; it < iterations; it++) {
        bool converged = false;

        for (int i = 0; i < N - 1; i++) {
            for (int j = i + 1; j < N; j++) {
                T* Si = S + i * M;
                T* Sj = S + j * M;
                T* Vi = V + i * N;
                T* Vj = V + j * N;

                T p = static_cast<T>(0);
                for (int k = 0; k < M; k++) { p += Si[k] * Sj[k]; }

                if (abs(p) <= M * EPS<T>::eps() * sqrt(d[i] * d[j])) {
                    continue;
                }

                T y  = d[i] - d[j];
                T r  = hypot(p * 2, y);
                T r2 = r * 2;
                T c, s;
                if (y >= 0) {
                    c = sqrt((r + y) / r2);
                    s = p / (r2 * c);
                } else {
                    s = sqrt((r - y) / r2);
                    c = p / (r2 * s);
                }

                T a = 0, b = 0;
                for (int k = 0; k < M; k++) {
                    T t0  = c * Si[k] + s * Sj[k];
                    T t1  = c * Sj[k] - s * Si[k];
                    Si[k] = t0;
                    Sj[k] = t1;

                    a += t0 * t0;
                    b += t1 * t1;
                }
                d[i] = a;
                d[j] = b;

                for (int l = 0; l < N; l++) {
                    T t0 = Vi[l] * c + Vj[l] * s;
                    T t1 = Vj[l] * c - Vi[l] * s;

                    Vi[l] = t0;
                    Vj[l] = t1;
                }

                converged = true;
            }
            if (!converged) { break; }
        }
    }
}

unsigned updateIterations(float inlier_ratio, unsigned iter) {
    float w  = min(max(inlier_ratio, 0.0f), 1.0f);
    float wn = pow(1 - w, 4.f);

    float d = 1.f - wn;
    if (d < numeric_limits<float>::min()) { return 0; }

    d = log(d);

    float p = min(max(RANSACConfidence, 0.0f), 1.0f);
    float n = log(1.f - p);

    return n <= d * static_cast<float>(iter)
               ? iter
               : static_cast<unsigned>(round(n / d));
}

template<typename T>
int computeHomography(T* H_ptr, const float* rnd_ptr, const float* x_src_ptr,
                      const float* y_src_ptr, const float* x_dst_ptr,
                      const float* y_dst_ptr) {
    if (static_cast<unsigned>(rnd_ptr[0]) ==
            static_cast<unsigned>(rnd_ptr[1]) ||
        static_cast<unsigned>(rnd_ptr[0]) ==
            static_cast<unsigned>(rnd_ptr[2]) ||
        static_cast<unsigned>(rnd_ptr[0]) ==
            static_cast<unsigned>(rnd_ptr[3]) ||
        static_cast<unsigned>(rnd_ptr[1]) ==
            static_cast<unsigned>(rnd_ptr[2]) ||
        static_cast<unsigned>(rnd_ptr[1]) ==
            static_cast<unsigned>(rnd_ptr[3]) ||
        static_cast<unsigned>(rnd_ptr[2]) ==
            static_cast<unsigned>(rnd_ptr[3])) {
        return 1;
    }

    float src_pt_x[4], src_pt_y[4], dst_pt_x[4], dst_pt_y[4];
    for (unsigned j = 0; j < 4; j++) {
        src_pt_x[j] = x_src_ptr[static_cast<unsigned>(rnd_ptr[j])];
        src_pt_y[j] = y_src_ptr[static_cast<unsigned>(rnd_ptr[j])];
        dst_pt_x[j] = x_dst_ptr[static_cast<unsigned>(rnd_ptr[j])];
        dst_pt_y[j] = y_dst_ptr[static_cast<unsigned>(rnd_ptr[j])];
    }

    float x_src_mean =
        (src_pt_x[0] + src_pt_x[1] + src_pt_x[2] + src_pt_x[3]) / 4.f;
    float y_src_mean =
        (src_pt_y[0] + src_pt_y[1] + src_pt_y[2] + src_pt_y[3]) / 4.f;
    float x_dst_mean =
        (dst_pt_x[0] + dst_pt_x[1] + dst_pt_x[2] + dst_pt_x[3]) / 4.f;
    float y_dst_mean =
        (dst_pt_y[0] + dst_pt_y[1] + dst_pt_y[2] + dst_pt_y[3]) / 4.f;

    float src_var = 0.0f, dst_var = 0.0f;
    for (unsigned j = 0; j < 4; j++) {
        src_var += sq(src_pt_x[j] - x_src_mean) + sq(src_pt_y[j] - y_src_mean);
        dst_var += sq(dst_pt_x[j] - x_dst_mean) + sq(dst_pt_y[j] - y_dst_mean);
    }

    src_var /= 4.f;
    dst_var /= 4.f;

    float src_scale = sqrt(2.0f) / sqrt(src_var);
    float dst_scale = sqrt(2.0f) / sqrt(dst_var);

    Array<T> A     = createValueArray<T>(af::dim4(9, 9), static_cast<T>(0));
    af::dim4 Adims = A.dims();
    T* A_ptr       = A.get();
    getQueue().sync();

    for (unsigned j = 0; j < 4; j++) {
        float srcx = (src_pt_x[j] - x_src_mean) * src_scale;
        float srcy = (src_pt_y[j] - y_src_mean) * src_scale;
        float dstx = (dst_pt_x[j] - x_dst_mean) * dst_scale;
        float dsty = (dst_pt_y[j] - y_dst_mean) * dst_scale;

        APTR(3, j * 2) = -srcx;
        APTR(4, j * 2) = -srcy;
        APTR(5, j * 2) = -1.0f;
        APTR(6, j * 2) = dsty * srcx;
        APTR(7, j * 2) = dsty * srcy;
        APTR(8, j * 2) = dsty;

        APTR(0, j * 2 + 1) = srcx;
        APTR(1, j * 2 + 1) = srcy;
        APTR(2, j * 2 + 1) = 1.0f;
        APTR(6, j * 2 + 1) = -dstx * srcx;
        APTR(7, j * 2 + 1) = -dstx * srcy;
        APTR(8, j * 2 + 1) = -dstx;
    }

    Array<T> V =
        createValueArray<T>(af::dim4(Adims[1], Adims[1]), static_cast<T>(0));
    V.eval();
    getQueue().sync();
    JacobiSVD<T, 9, 9>(A.get(), V.get());

    dim4 Vdims = V.dims();
    T* V_ptr   = V.get();

    array<T, 9> vH{};
    for (unsigned j = 0; j < 9; j++) { vH[j] = V_ptr[8 * Vdims[0] + j]; }

    H_ptr[0] = src_scale * x_dst_mean * vH[6] + src_scale * vH[0] / dst_scale;
    H_ptr[1] = src_scale * x_dst_mean * vH[7] + src_scale * vH[1] / dst_scale;
    H_ptr[2] = x_dst_mean * (vH[8] - src_scale * y_src_mean * vH[7] -
                             src_scale * x_src_mean * vH[6]) +
               (vH[2] - src_scale * y_src_mean * vH[1] -
                src_scale * x_src_mean * vH[0]) /
                   dst_scale;

    H_ptr[3] = src_scale * y_dst_mean * vH[6] + src_scale * vH[3] / dst_scale;
    H_ptr[4] = src_scale * y_dst_mean * vH[7] + src_scale * vH[4] / dst_scale;
    H_ptr[5] = y_dst_mean * (vH[8] - src_scale * y_src_mean * vH[7] -
                             src_scale * x_src_mean * vH[6]) +
               (vH[5] - src_scale * y_src_mean * vH[4] -
                src_scale * x_src_mean * vH[3]) /
                   dst_scale;

    H_ptr[6] = src_scale * vH[6];
    H_ptr[7] = src_scale * vH[7];
    H_ptr[8] =
        vH[8] - src_scale * y_src_mean * vH[7] - src_scale * x_src_mean * vH[6];

    return 0;
}

// LMedS:
// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node25.html
template<typename T>
int findBestHomography(Array<T>& bestH, const Array<float>& x_src,
                       const Array<float>& y_src, const Array<float>& x_dst,
                       const Array<float>& y_dst, const Array<float>& rnd,
                       const unsigned iterations, const unsigned nsamples,
                       const float inlier_thr, const af_homography_type htype) {
    const float* x_src_ptr = x_src.get();
    const float* y_src_ptr = y_src.get();
    const float* x_dst_ptr = x_dst.get();
    const float* y_dst_ptr = y_dst.get();

    Array<T> H =
        createValueArray<T>(af::dim4(9, iterations), static_cast<T>(0));
    H.eval();
    getQueue().sync();

    const af::dim4& rdims = rnd.dims();
    const af::dim4& Hdims = H.dims();

    unsigned iter    = iterations;
    unsigned bestIdx = 0;
    int bestInliers  = 0;
    float minMedian  = numeric_limits<float>::max();

    for (unsigned i = 0; i < iter; i++) {
        const unsigned Hidx = Hdims[0] * i;
        T* H_ptr            = H.get() + Hidx;

        const unsigned ridx  = rdims[0] * i;
        const float* rnd_ptr = rnd.get() + ridx;

        if (computeHomography<T>(H_ptr, rnd_ptr, x_src_ptr, y_src_ptr,
                                 x_dst_ptr, y_dst_ptr)) {
            continue;
        }

        if (htype == AF_HOMOGRAPHY_RANSAC) {
            int inliers_count = 0;
            for (unsigned j = 0; j < nsamples; j++) {
                float z = H_ptr[6] * x_src_ptr[j] + H_ptr[7] * y_src_ptr[j] +
                          H_ptr[8];
                float x = (H_ptr[0] * x_src_ptr[j] + H_ptr[1] * y_src_ptr[j] +
                           H_ptr[2]) /
                          z;
                float y = (H_ptr[3] * x_src_ptr[j] + H_ptr[4] * y_src_ptr[j] +
                           H_ptr[5]) /
                          z;

                float dist = sq(x_dst_ptr[j] - x) + sq(y_dst_ptr[j] - y);
                if (dist < (inlier_thr * inlier_thr)) { inliers_count++; }
            }
            iter =
                updateIterations(static_cast<float>(nsamples - inliers_count) /
                                     static_cast<float>(nsamples),
                                 iter);
            if (inliers_count > bestInliers) {
                bestIdx     = i;
                bestInliers = inliers_count;
            }
        } else if (htype == AF_HOMOGRAPHY_LMEDS) {
            vector<float> err(nsamples);
            for (unsigned j = 0; j < nsamples; j++) {
                float z = H_ptr[6] * x_src_ptr[j] + H_ptr[7] * y_src_ptr[j] +
                          H_ptr[8];
                float x = (H_ptr[0] * x_src_ptr[j] + H_ptr[1] * y_src_ptr[j] +
                           H_ptr[2]) /
                          z;
                float y = (H_ptr[3] * x_src_ptr[j] + H_ptr[4] * y_src_ptr[j] +
                           H_ptr[5]) /
                          z;

                float dist = sq(x_dst_ptr[j] - x) + sq(y_dst_ptr[j] - y);
                err[j]     = sqrt(dist);
            }

            stable_sort(err.begin(), err.end());

            float median = err[nsamples / 2];
            if (nsamples % 2 == 0) {
                median = (median + err[nsamples / 2 - 1]) * 0.5f;
            }

            if (median < minMedian &&
                median > numeric_limits<float>::epsilon()) {
                minMedian = median;
                bestIdx   = i;
            }
        }
    }

    memcpy(bestH.get(), H.get() + bestIdx * 9, 9 * sizeof(T));

    if (htype == AF_HOMOGRAPHY_LMEDS) {
        float sigma =
            max(1.4826f * (1.f + 5.f / (static_cast<float>(nsamples) - 4.f)) *
                    static_cast<float>(sqrt(minMedian)),
                1e-6f);
        float dist_thr = sq(2.5f * sigma);
        T* bestH_ptr   = bestH.get();

        for (unsigned j = 0; j < nsamples; j++) {
            float z = bestH_ptr[6] * x_src_ptr[j] +
                      bestH_ptr[7] * y_src_ptr[j] + bestH_ptr[8];
            float x = (bestH_ptr[0] * x_src_ptr[j] +
                       bestH_ptr[1] * y_src_ptr[j] + bestH_ptr[2]) /
                      z;
            float y = (bestH_ptr[3] * x_src_ptr[j] +
                       bestH_ptr[4] * y_src_ptr[j] + bestH_ptr[5]) /
                      z;

            float dist = sq(x_dst_ptr[j] - x) + sq(y_dst_ptr[j] - y);
            if (dist <= dist_thr) { bestInliers++; }
        }
    }

    return bestInliers;
}

template<typename T>
int homography(Array<T>& bestH, const Array<float>& x_src,
               const Array<float>& y_src, const Array<float>& x_dst,
               const Array<float>& y_dst, const Array<float>& initial,
               const af_homography_type htype, const float inlier_thr,
               const unsigned iterations) {
    const dim4& idims       = x_src.dims();
    const unsigned nsamples = idims[0];

    unsigned iter = iterations;
    if (htype == AF_HOMOGRAPHY_LMEDS) {
        iter = min(iter, static_cast<unsigned>(
                             log(1.f - LMEDSConfidence) /
                             log(1.f - pow(1.f - LMEDSOutlierRatio, 4.f))));
    }

    af::dim4 rdims(4, iter);
    Array<float> fctr =
        createValueArray<float>(rdims, static_cast<float>(nsamples));
    Array<float> rnd = arithOp<float, af_mul_t>(initial, fctr, rdims);
    rnd.eval();
    getQueue().sync();

    return findBestHomography<T>(bestH, x_src, y_src, x_dst, y_dst, rnd, iter,
                                 nsamples, inlier_thr, htype);
}

#define INSTANTIATE(T)                                          \
    template int homography<T>(                                 \
        Array<T> & bestH, const Array<float>& x_src,            \
        const Array<float>& y_src, const Array<float>& x_dst,   \
        const Array<float>& y_dst, const Array<float>& initial, \
        const af_homography_type htype, const float inlier_thr, \
        const unsigned iterations);

INSTANTIATE(float)
INSTANTIATE(double)

}  // namespace cpu
}  // namespace arrayfire

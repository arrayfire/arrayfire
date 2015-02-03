/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/features.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <err_cpu.hpp>
#include <handle.hpp>
#include <fast.hpp>

using af::dim4;

namespace cpu
{

inline int clamp(int f, int a, int b)
{
    return std::max(a, std::min(f, b));
}

inline int idx_y(int i)
{
    if (i >= 8)
        return clamp(-(i-8-4), -3, 3);

    return clamp(i-4, -3, 3);
}

inline int idx_x(int i)
{
    if (i < 12)
        return idx_y(i+4);

    return idx_y(i-12);
}

inline int idx(int y, int x, unsigned idim0)
{
    return x * idim0 + y;
}

// test_greater()
// Tests if a pixel x > p + thr
template<typename T>
inline int test_greater(T x, T p, float thr)
{
    return (x >= p + thr);
}

// test_smaller()
// Tests if a pixel x < p - thr
template<typename T>
inline int test_smaller(T x, T p, float thr)
{
    return (x <= p - thr);
}

// test_pixel()
// Returns -1 when x < p - thr
// Returns  0 when x >= p - thr && x <= p + thr
// Returns  1 when x > p + thr
template<typename T>
inline int test_pixel(const T* image, const T p, float thr, int y, int x, unsigned idim0)
{
    return -test_smaller<T>(image[idx(y,x,idim0)], p, thr) | test_greater<T>(image[idx(y,x,idim0)], p, thr);
}

// abs_diff()
// Returns absolute difference of x and y
inline int abs_diff(int x, int y)
{
    return abs(x - y);
}
inline unsigned abs_diff(unsigned x, unsigned y)
{
    return (unsigned)abs((int)x - (int)y);
}
inline float abs_diff(float x, float y)
{
    return fabs(x - y);
}
inline double abs_diff(double x, double y)
{
    return fabs(x - y);
}

template<typename T>
void locate_features(
    const Array<T> &in,
    Array<T> &score,
    Array<float> &x_out,
    Array<float> &y_out,
    Array<float> &score_out,
    unsigned* count,
    const float thr,
    const unsigned arc_length,
    const unsigned nonmax,
    const unsigned max_feat)
{
    dim4 in_dims = in.dims();
    const T* in_ptr = in.get();

    for (int y = 3; y < (int)(in_dims[0] - 3); y++) {
        for (int x = 3; x < (int)(in_dims[1] - 3); x++) {
            T p = in_ptr[idx(y, x, in_dims[0])];

            // Start by testing opposite pixels of the circle that will result in
            // a non-kepoint
            int d;
            d  = test_pixel<T>(in_ptr, p, thr, y-3,   x, in_dims[0]) | test_pixel<T>(in_ptr, p, thr, y+3,   x, in_dims[0]);
            if (d == 0)
                continue;

            d &= test_pixel<T>(in_ptr, p, thr, y-2, x+2, in_dims[0]) | test_pixel<T>(in_ptr, p, thr, y+2, x-2, in_dims[0]);
            d &= test_pixel<T>(in_ptr, p, thr, y  , x+3, in_dims[0]) | test_pixel<T>(in_ptr, p, thr, y  , x-3, in_dims[0]);
            d &= test_pixel<T>(in_ptr, p, thr, y+2, x+2, in_dims[0]) | test_pixel<T>(in_ptr, p, thr, y-2, x-2, in_dims[0]);
            if (d == 0)
                continue;

            d &= test_pixel<T>(in_ptr, p, thr, y-3, x+1, in_dims[0]) | test_pixel<T>(in_ptr, p, thr, y+3, x-1, in_dims[0]);
            d &= test_pixel<T>(in_ptr, p, thr, y-1, x+3, in_dims[0]) | test_pixel<T>(in_ptr, p, thr, y+1, x-3, in_dims[0]);
            d &= test_pixel<T>(in_ptr, p, thr, y+1, x+3, in_dims[0]) | test_pixel<T>(in_ptr, p, thr, y-1, x-3, in_dims[0]);
            d &= test_pixel<T>(in_ptr, p, thr, y+3, x+1, in_dims[0]) | test_pixel<T>(in_ptr, p, thr, y-3, x-1, in_dims[0]);
            if (d == 0)
                continue;

            int sum = 0;

            // Sum responses [-1, 0 or 1] of first arc_length pixels
            for (int i = 0; i < static_cast<int>(arc_length); i++)
                sum += test_pixel<T>(in_ptr, p, thr, y+idx_y(i), x+idx_x(i), in_dims[0]);

            // Test maximum and mininmum responses of first segment of arc_length
            // pixels
            int max_sum = 0, min_sum = 0;
            max_sum = std::max(max_sum, sum);
            min_sum = std::min(min_sum, sum);

            // Sum responses and test the remaining 16-arc_length pixels of the circle
            for (int i = arc_length; i < 16; i++) {
                sum -= test_pixel<T>(in_ptr, p, thr, y+idx_y(i-arc_length), x+idx_x(i-arc_length), in_dims[0]);
                sum += test_pixel<T>(in_ptr, p, thr, y+idx_y(i), x+idx_x(i), in_dims[0]);
                max_sum = std::max(max_sum, sum);
                min_sum = std::min(min_sum, sum);
            }

            // To completely test all possible segments, it's necessary to test
            // segments that include the top junction of the circle
            for (int i = 0; i < static_cast<int>(arc_length-1); i++) {
                sum -= test_pixel<T>(in_ptr, p, thr, y+idx_y(16-arc_length+i), x+idx_x(16-arc_length+i), in_dims[0]);
                sum += test_pixel<T>(in_ptr, p, thr, y+idx_y(i), x+idx_x(i), in_dims[0]);
                max_sum = std::max(max_sum, sum);
                min_sum = std::min(min_sum, sum);
            }

            T s_bright = 0, s_dark = 0;
            for (int i = 0; i < 16; i++) {
                s_bright += test_greater<T>(in_ptr[idx(y+idx_y(i), x+idx_x(i), in_dims[0])], p, thr) *
                            (abs_diff(in_ptr[idx(y+idx_y(i), x+idx_x(i), in_dims[0])], p) - thr);
                s_dark   += test_smaller<T>(in_ptr[idx(y+idx_y(i), x+idx_x(i), in_dims[0])], p, thr) *
                            (abs_diff(p, in_ptr[idx(y+idx_y(i), x+idx_x(i), in_dims[0])]) - thr);
            }

            // If sum at some point was equal to (+-)arc_length, there is a segment
            // that for which all pixels are much brighter or much brighter than
            // central pixel p.
            if (max_sum == static_cast<int>(arc_length) || min_sum == -static_cast<int>(arc_length)) {
                unsigned j = *count;
                ++*count;
                if (j < max_feat) {
                    float *x_out_ptr = x_out.get();
                    float *y_out_ptr = y_out.get();
                    float *score_out_ptr = score_out.get();
                    x_out_ptr[j]     = static_cast<float>(x);
                    y_out_ptr[j]     = static_cast<float>(y);
                    score_out_ptr[j] = static_cast<float>(std::max(s_bright, s_dark));
                    if (nonmax == 1) {
                        T* score_ptr = score.get();
                        score_ptr[idx(y, x, in_dims[0])] = std::max(s_bright, s_dark);
                    }
                }
            }
        }
    }
}

template<typename T>
void non_maximal(
    const Array<T> &score,
    const Array<float> &x_in,
    const Array<float> &y_in,
    Array<float> &x_out,
    Array<float> &y_out,
    Array<float> &score_out,
    unsigned* count,
    const unsigned total_feat)
{
    const T *score_ptr = score.get();
    const float *x_in_ptr = x_in.get();
    const float *y_in_ptr = y_in.get();

    dim4 score_dims = score.dims();

    for (unsigned k = 0; k < total_feat; k++) {
        unsigned x = static_cast<unsigned>(round(x_in_ptr[k]));
        unsigned y = static_cast<unsigned>(round(y_in_ptr[k]));

        T v = score_ptr[y + score_dims[0] * x];
        T max_v;
        max_v = std::max(score_ptr[y-1 + score_dims[0] * (x-1)], score_ptr[y-1 + score_dims[0] * x]);
        max_v = std::max(max_v, score_ptr[y-1 + score_dims[0] * (x+1)]);
        max_v = std::max(max_v, score_ptr[y   + score_dims[0] * (x-1)]);
        max_v = std::max(max_v, score_ptr[y   + score_dims[0] * (x+1)]);
        max_v = std::max(max_v, score_ptr[y+1 + score_dims[0] * (x-1)]);
        max_v = std::max(max_v, score_ptr[y+1 + score_dims[0] * (x)  ]);
        max_v = std::max(max_v, score_ptr[y+1 + score_dims[0] * (x+1)]);

        // Stores keypoint to feat_out if it's response is maximum compared to
        // its 8-neighborhood
        if (v > max_v) {
            unsigned j = *count;
            ++*count;

            float *x_out_ptr = x_out.get();
            float *y_out_ptr = y_out.get();
            float *score_out_ptr = score_out.get();

            x_out_ptr[j]     = static_cast<float>(x);
            y_out_ptr[j]     = static_cast<float>(y);
            score_out_ptr[j] = static_cast<float>(v);
        }
    }
}

template<typename T>
features fast(const Array<T> &in, const float thr, const unsigned arc_length,
              const bool nonmax, const float feature_ratio)
{
    dim4 in_dims = in.dims();
    const unsigned max_feat = ceil(in.elements() * feature_ratio);

    // Matrix containing scores for detected features, scores are stored in the
    // same coordinates as features, dimensions should be equal to in.
    Array<T> *V = NULL;
    if (nonmax == 1) {
        dim4 V_dims(in_dims[0], in_dims[1]);
        V = createValueArray<T>(V_dims, (T)0);
    }

    // Arrays containing all features detected before non-maximal suppression.
    dim4 max_feat_dims(max_feat);
    Array<float> *x = createEmptyArray<float>(max_feat_dims);
    Array<float> *y = createEmptyArray<float>(max_feat_dims);
    Array<float> *score = createEmptyArray<float>(max_feat_dims);

    // Feature counter
    unsigned count = 0;

    locate_features<T>(in, *V, *x, *y, *score, &count, thr, arc_length, nonmax, max_feat);

    // If more features than max_feat were detected, feat wasn't populated
    // with them anyway, so the real number of features will be that of
    // max_feat and not count.
    unsigned feat_found = std::min(max_feat, count);
    dim4 feat_found_dims(feat_found);

    Array<float> *x_out;
    Array<float> *y_out;
    Array<float> *score_out;
    Array<float> *orientation_out;
    Array<float> *size_out;

    if (nonmax == 1) {
        count = 0;

        Array<float> *x_nonmax = createEmptyArray<float>(feat_found_dims);
        Array<float> *y_nonmax = createEmptyArray<float>(feat_found_dims);
        Array<float> *score_nonmax = createEmptyArray<float>(feat_found_dims);

        non_maximal<T>(*V, *x, *y,
                       *x_nonmax, *y_nonmax, *score_nonmax,
                       &count, feat_found);

        delete V;

        feat_found = std::min(max_feat, count);

        if (feat_found > 0) {
            feat_found_dims = dim4(feat_found);

            // TODO: improve output copy, use af_index?
            x_out = createEmptyArray<float>(feat_found_dims);
            y_out = createEmptyArray<float>(feat_found_dims);
            score_out = createEmptyArray<float>(feat_found_dims);
            orientation_out = createValueArray<float>(feat_found_dims, 0.0f);
            size_out = createValueArray<float>(feat_found_dims, 1.0f);

            float *x_nonmax_ptr = x_nonmax->get();
            float *y_nonmax_ptr = y_nonmax->get();
            float *score_nonmax_ptr = score_nonmax->get();
            float *x_out_ptr = x_out->get();
            float *y_out_ptr = y_out->get();
            float *score_out_ptr = score_out->get();
            for (size_t i = 0; i < feat_found; i++) {
                x_out_ptr[i] = x_nonmax_ptr[i];
                y_out_ptr[i] = y_nonmax_ptr[i];
                score_out_ptr[i] = score_nonmax_ptr[i];
            }
        }

        delete x;
        delete y;
        delete score;
        delete x_nonmax;
        delete y_nonmax;
        delete score_nonmax;
    }
    else {
        if (feat_found > 0) {
            // TODO: improve output copy, use af_index?
            x_out = createEmptyArray<float>(feat_found_dims);
            y_out = createEmptyArray<float>(feat_found_dims);
            score_out = createEmptyArray<float>(feat_found_dims);
            orientation_out = createValueArray<float>(feat_found_dims, 0.0f);
            size_out = createValueArray<float>(feat_found_dims, 1.0f);

            float *x_ptr = x->get();
            float *y_ptr = y->get();
            float *score_ptr = score->get();
            float *x_out_ptr = x_out->get();
            float *y_out_ptr = y_out->get();
            float *score_out_ptr = score_out->get();
            for (size_t i = 0; i < feat_found; i++) {
                x_out_ptr[i] = x_ptr[i];
                y_out_ptr[i] = y_ptr[i];
                score_out_ptr[i] = score_ptr[i];
            }
        }

        delete x;
        delete y;
        delete score;
    }

    features feat;
    if (feat_found == 0) {
        feat.setNumFeatures(0);
        feat.setX(getHandle<float>(*createEmptyArray<float>(af::dim4())));
        feat.setY(getHandle<float>(*createEmptyArray<float>(af::dim4())));
        feat.setScore(getHandle<float>(*createEmptyArray<float>(af::dim4())));
        feat.setOrientation(getHandle<float>(*createEmptyArray<float>(af::dim4())));
        feat.setSize(getHandle<float>(*createEmptyArray<float>(af::dim4())));
    }
    else {
        feat.setNumFeatures(feat_found);
        feat.setX(getHandle<float>(*x_out));
        feat.setY(getHandle<float>(*y_out));
        feat.setScore(getHandle<float>(*score_out));
        feat.setOrientation(getHandle<float>(*orientation_out));
        feat.setSize(getHandle<float>(*size_out));

        // Delete weak copies
        delete x_out;
        delete y_out;
        delete score_out;
        delete orientation_out;
        delete size_out;
    }

    return feat;
}

#define INSTANTIATE(T)\
    template features fast<T>(const Array<T> &in, const float thr, const unsigned arc_length, \
                              const bool nonmax, const float feature_ratio);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}

/*******************************************************
 * Copyright (c) 2021, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// The source code contained in this file is based on the original code by
// Rob Hess. Please note that SIFT is an algorithm patented and protected
// by US law. As of 29-Dec-2020, the patent stands expired. It can be looked
// up here - https://patents.google.com/patent/US6711293B1/en

#pragma once

#include <convolve.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <resize.hpp>
#include <sort_index.hpp>

#include <cstring>
#include <limits>
#include <vector>

using af::dim4;

namespace arrayfire {
namespace cpu {

static const float PI_VAL = 3.14159265358979323846f;

// default width of descriptor histogram array
static const int DescrWidth = 4;

// default number of bins per histogram in descriptor array
static const int DescrHistBins = 8;

// assumed gaussian blur for input image
static const float InitSigma = 0.5f;

// width of border in which to ignore keypoints
static const int ImgBorder = 5;

// maximum steps of keypoint interpolation before failure
static const int MaxInterpSteps = 5;

// default number of bins in histogram for orientation assignment
static const int OriHistBins = 36;

// determines gaussian sigma for orientation assignment
static const float OriSigFctr = 1.5f;

// determines the radius of the region used in orientation assignment */
static const float OriRadius = 3.0f * OriSigFctr;

// number of passes of orientation histogram smoothing
static const int SmoothOriPasses = 2;

// orientation magnitude relative to max that results in new feature
static const float OriPeakRatio = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float DescrSclFctr = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float DescrMagThr = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float IntDescrFctr = 512.f;

// Number of GLOH bins in radial direction
static const unsigned GLOHRadialBins = 3;

// Radiuses of GLOH descriptors
static const float GLOHRadii[GLOHRadialBins] = {6.f, 11.f, 15.f};

// Number of GLOH angular bins (excluding the inner-most radial section)
static const unsigned GLOHAngularBins = 8;

// Number of GLOH bins per histogram in descriptor
static const unsigned GLOHHistBins = 16;

typedef struct {
    float f[4];
    unsigned l;
} feat_t;

bool feat_cmp(feat_t i, feat_t j) {
    for (int k = 0; k < 4; k++)
        if (i.f[k] != j.f[k]) return (i.f[k] < j.f[k]);
    if (i.l != j.l) return (i.l < j.l);

    return false;
}

void array_to_feat(std::vector<feat_t>& feat, float* x, float* y,
                   unsigned* layer, float* resp, float* size, unsigned nfeat) {
    feat.resize(nfeat);
    for (unsigned i = 0; i < feat.size(); i++) {
        feat[i].f[0] = x[i];
        feat[i].f[1] = y[i];
        feat[i].f[2] = resp[i];
        feat[i].f[3] = size[i];
        feat[i].l    = layer[i];
    }
}

template<typename T>
void gaussian1D(T* out, const int dim, double sigma = 0.0) {
    if (!(sigma > 0)) sigma = 0.25 * dim;

    T sum = (T)0;
    for (int i = 0; i < dim; i++) {
        int x = i - (dim - 1) / 2;
        T el  = 1. / sqrt(2 * PI_VAL * sigma * sigma) *
               exp(-((x * x) / (2 * (sigma * sigma))));
        out[i] = el;
        sum += el;
    }

    for (int k = 0; k < dim; k++) out[k] /= sum;
}

template<typename T>
Array<T> gauss_filter(float sigma) {
    // Using 6-sigma rule
    unsigned gauss_len = std::min((unsigned)round(sigma * 6 + 1) | 1, 31u);

    Array<T> filter = createEmptyArray<T>(gauss_len);
    gaussian1D((T*)getDevicePtr(filter), gauss_len, sigma);

    return filter;
}

template<int N>
void gaussianElimination(float* A, float* b, float* x) {
    // forward elimination
    for (int i = 0; i < N - 1; i++) {
        for (int j = i + 1; j < N; j++) {
            float s = A[j * N + i] / A[i * N + i];

            for (int k = i; k < N; k++) A[j * N + k] -= s * A[i * N + k];

            b[j] -= s * b[i];
        }
    }

    for (int i = 0; i < N; i++) x[i] = 0;

    // backward substitution
    float sum = 0;
    for (int i = 0; i <= N - 2; i++) {
        sum = b[i];
        for (int j = i + 1; j < N; j++) sum -= A[i * N + j] * x[j];
        x[i] = sum / A[i * N + i];
    }
}

template<typename T>
void sub(Array<T>& out, const Array<T>& in1, const Array<T>& in2) {
    size_t nel       = in1.elements();
    T* out_ptr       = out.get();
    const T* in1_ptr = in1.get();
    const T* in2_ptr = in2.get();

    for (size_t i = 0; i < nel; i++) { out_ptr[i] = in1_ptr[i] - in2_ptr[i]; }
}

#define CPTR(Y, X) (center_ptr[(Y)*idims[0] + (X)])
#define PPTR(Y, X) (prev_ptr[(Y)*idims[0] + (X)])
#define NPTR(Y, X) (next_ptr[(Y)*idims[0] + (X)])

// Determines whether a pixel is a scale-space extremum by comparing it to its
// 3x3x3 pixel neighborhood.
template<typename T>
void detectExtrema(float* x_out, float* y_out, unsigned* layer_out,
                   unsigned* counter, const Array<T>& prev,
                   const Array<T>& center, const Array<T>& next,
                   const unsigned layer, const unsigned max_feat,
                   const float threshold) {
    const af::dim4 idims = center.dims();
    const T* prev_ptr    = prev.get();
    const T* center_ptr  = center.get();
    const T* next_ptr    = next.get();

    for (int y = ImgBorder; y < idims[1] - ImgBorder; y++) {
        for (int x = ImgBorder; x < idims[0] - ImgBorder; x++) {
            float p = center_ptr[y * idims[0] + x];

            // Find extrema
            if (abs((float)p) > threshold &&
                ((p > 0 && p > CPTR(y - 1, x - 1) && p > CPTR(y - 1, x) &&
                  p > CPTR(y - 1, x + 1) && p > CPTR(y, x - 1) &&
                  p > CPTR(y, x + 1) && p > CPTR(y + 1, x - 1) &&
                  p > CPTR(y + 1, x) && p > CPTR(y + 1, x + 1) &&
                  p > PPTR(y - 1, x - 1) && p > PPTR(y - 1, x) &&
                  p > PPTR(y - 1, x + 1) && p > PPTR(y, x - 1) &&
                  p > PPTR(y, x) && p > PPTR(y, x + 1) &&
                  p > PPTR(y + 1, x - 1) && p > PPTR(y + 1, x) &&
                  p > PPTR(y + 1, x + 1) && p > NPTR(y - 1, x - 1) &&
                  p > NPTR(y - 1, x) && p > NPTR(y - 1, x + 1) &&
                  p > NPTR(y, x - 1) && p > NPTR(y, x) && p > NPTR(y, x + 1) &&
                  p > NPTR(y + 1, x - 1) && p > NPTR(y + 1, x) &&
                  p > NPTR(y + 1, x + 1)) ||
                 (p < 0 && p < CPTR(y - 1, x - 1) && p < CPTR(y - 1, x) &&
                  p < CPTR(y - 1, x + 1) && p < CPTR(y, x - 1) &&
                  p < CPTR(y, x + 1) && p < CPTR(y + 1, x - 1) &&
                  p < CPTR(y + 1, x) && p < CPTR(y + 1, x + 1) &&
                  p < PPTR(y - 1, x - 1) && p < PPTR(y - 1, x) &&
                  p < PPTR(y - 1, x + 1) && p < PPTR(y, x - 1) &&
                  p < PPTR(y, x) && p < PPTR(y, x + 1) &&
                  p < PPTR(y + 1, x - 1) && p < PPTR(y + 1, x) &&
                  p < PPTR(y + 1, x + 1) && p < NPTR(y - 1, x - 1) &&
                  p < NPTR(y - 1, x) && p < NPTR(y - 1, x + 1) &&
                  p < NPTR(y, x - 1) && p < NPTR(y, x) && p < NPTR(y, x + 1) &&
                  p < NPTR(y + 1, x - 1) && p < NPTR(y + 1, x) &&
                  p < NPTR(y + 1, x + 1)))) {
                if (*counter < max_feat) {
                    x_out[*counter]     = (float)y;
                    y_out[*counter]     = (float)x;
                    layer_out[*counter] = layer;
                    (*counter)++;
                }
            }
        }
    }
}

// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
template<typename T>
void interpolateExtrema(float* x_out, float* y_out, unsigned* layer_out,
                        float* response_out, float* size_out, unsigned* counter,
                        const float* x_in, const float* y_in,
                        const unsigned* layer_in, const unsigned extrema_feat,
                        std::vector<Array<T>>& dog_pyr, const unsigned max_feat,
                        const unsigned octave, const unsigned n_layers,
                        const float contrast_thr, const float edge_thr,
                        const float sigma, const float img_scale) {
    for (int f = 0; f < (int)extrema_feat; f++) {
        const float first_deriv_scale  = img_scale * 0.5f;
        const float second_deriv_scale = img_scale;
        const float cross_deriv_scale  = img_scale * 0.25f;

        float xl = 0, xy = 0, xx = 0, contr = 0;
        int i = 0;

        unsigned x     = x_in[f];
        unsigned y     = y_in[f];
        unsigned layer = layer_in[f];

        const T* prev_ptr = dog_pyr[octave * (n_layers + 2) + layer - 1].get();
        const T* center_ptr = dog_pyr[octave * (n_layers + 2) + layer].get();
        const T* next_ptr = dog_pyr[octave * (n_layers + 2) + layer + 1].get();

        af::dim4 idims = dog_pyr[octave * (n_layers + 2)].dims();

        bool converges = true;

        for (i = 0; i < MaxInterpSteps; i++) {
            float dD[3] = {
                (float)(CPTR(x + 1, y) - CPTR(x - 1, y)) * first_deriv_scale,
                (float)(CPTR(x, y + 1) - CPTR(x, y - 1)) * first_deriv_scale,
                (float)(NPTR(x, y) - PPTR(x, y)) * first_deriv_scale};

            float d2 = CPTR(x, y) * 2.f;
            float dxx =
                (CPTR(x + 1, y) + CPTR(x - 1, y) - d2) * second_deriv_scale;
            float dyy =
                (CPTR(x, y + 1) + CPTR(x, y - 1) - d2) * second_deriv_scale;
            float dss = (NPTR(x, y) + PPTR(x, y) - d2) * second_deriv_scale;
            float dxy = (CPTR(x + 1, y + 1) - CPTR(x - 1, y + 1) -
                         CPTR(x + 1, y - 1) + CPTR(x - 1, y - 1)) *
                        cross_deriv_scale;
            float dxs = (NPTR(x + 1, y) - NPTR(x - 1, y) - PPTR(x + 1, y) +
                         PPTR(x - 1, y)) *
                        cross_deriv_scale;
            float dys = (NPTR(x, y + 1) - NPTR(x - 1, y - 1) - PPTR(x, y - 1) +
                         PPTR(x - 1, y - 1)) *
                        cross_deriv_scale;

            float H[9] = {dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss};

            float X[3];
            gaussianElimination<3>(H, dD, X);

            xl = -X[2];
            xy = -X[1];
            xx = -X[0];

            if (fabs(xl) < 0.5f && fabs(xy) < 0.5f && fabs(xx) < 0.5f) break;

            x += round(xx);
            y += round(xy);
            layer += round(xl);

            if (layer < 1 || layer > n_layers || x < ImgBorder ||
                x >= idims[1] - ImgBorder || y < ImgBorder ||
                y >= idims[0] - ImgBorder) {
                converges = false;
                break;
            }
        }

        // ensure convergence of interpolation
        if (i >= MaxInterpSteps || !converges) continue;

        float dD[3] = {
            (float)(CPTR(x + 1, y) - CPTR(x - 1, y)) * first_deriv_scale,
            (float)(CPTR(x, y + 1) - CPTR(x, y - 1)) * first_deriv_scale,
            (float)(NPTR(x, y) - PPTR(x, y)) * first_deriv_scale};
        float X[3] = {xx, xy, xl};

        float P = dD[0] * X[0] + dD[1] * X[1] + dD[2] * X[2];

        contr = center_ptr[x * idims[0] + y] * img_scale + P * 0.5f;
        if (abs(contr) < (contrast_thr / n_layers)) continue;

        // principal curvatures are computed using the trace and det of Hessian
        float d2  = CPTR(x, y) * 2.f;
        float dxx = (CPTR(x + 1, y) + CPTR(x - 1, y) - d2) * second_deriv_scale;
        float dyy = (CPTR(x, y + 1) + CPTR(x, y - 1) - d2) * second_deriv_scale;
        float dxy = (CPTR(x + 1, y + 1) - CPTR(x - 1, y + 1) -
                     CPTR(x + 1, y - 1) + CPTR(x - 1, y - 1)) *
                    cross_deriv_scale;

        float tr  = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        // add FLT_EPSILON for double-precision compatibility
        if (det <= 0 ||
            tr * tr * edge_thr >= (edge_thr + 1) * (edge_thr + 1) * det +
                                      std::numeric_limits<float>::epsilon())
            continue;

        if (*counter < max_feat) {
            x_out[*counter]        = (x + xx) * (1 << octave);
            y_out[*counter]        = (y + xy) * (1 << octave);
            layer_out[*counter]    = layer;
            response_out[*counter] = abs(contr);
            size_out[*counter] =
                sigma * pow(2.f, octave + (layer + xl) / n_layers) * 2.f;
            (*counter)++;
        }
    }
}

#undef CPTR
#undef PPTR
#undef NPTR

// Remove duplicate keypoints
void removeDuplicates(float* x_out, float* y_out, unsigned* layer_out,
                      float* response_out, float* size_out, unsigned* counter,
                      const std::vector<feat_t>& sorted_feat) {
    size_t nfeat = sorted_feat.size();

    for (size_t f = 0; f < nfeat; f++) {
        float prec_fctr = 1e4f;

        if (f < nfeat - 1) {
            if (round(sorted_feat[f].f[0] * prec_fctr) ==
                    round(sorted_feat[f + 1].f[0] * prec_fctr) &&
                round(sorted_feat[f].f[1] * prec_fctr) ==
                    round(sorted_feat[f + 1].f[1] * prec_fctr) &&
                round(sorted_feat[f].f[2] * prec_fctr) ==
                    round(sorted_feat[f + 1].f[2] * prec_fctr) &&
                round(sorted_feat[f].f[3] * prec_fctr) ==
                    round(sorted_feat[f + 1].f[3] * prec_fctr) &&
                sorted_feat[f].l == sorted_feat[f + 1].l)
                continue;
        }

        x_out[*counter]        = sorted_feat[f].f[0];
        y_out[*counter]        = sorted_feat[f].f[1];
        response_out[*counter] = sorted_feat[f].f[2];
        size_out[*counter]     = sorted_feat[f].f[3];
        layer_out[*counter]    = sorted_feat[f].l;
        (*counter)++;
    }
}

#define IPTR(Y, X) (img_ptr[(Y)*idims[0] + (X)])

// Computes a canonical orientation for each image feature in an array.  Based
// on Section 5 of Lowe's paper.  This function adds features to the array when
// there is more than one dominant orientation at a given feature location.
template<typename T>
void calcOrientation(float* x_out, float* y_out, unsigned* layer_out,
                     float* response_out, float* size_out, float* ori_out,
                     unsigned* counter, const float* x_in, const float* y_in,
                     const unsigned* layer_in, const float* response_in,
                     const float* size_in, const unsigned total_feat,
                     const std::vector<Array<T>>& gauss_pyr,
                     const unsigned max_feat, const unsigned octave,
                     const unsigned n_layers, const bool double_input) {
    const int n = OriHistBins;

    float hist[OriHistBins];
    float temphist[OriHistBins];

    for (unsigned f = 0; f < total_feat; f++) {
        // Load keypoint information
        const float real_x   = x_in[f];
        const float real_y   = y_in[f];
        const unsigned layer = layer_in[f];
        const float response = response_in[f];
        const float size     = size_in[f];

        const int pt_x = (int)round(real_x / (1 << octave));
        const int pt_y = (int)round(real_y / (1 << octave));

        // Calculate auxiliary parameters
        const float scl_octv  = size * 0.5f / (1 << octave);
        const int radius      = (int)round(OriRadius * scl_octv);
        const float sigma     = OriSigFctr * scl_octv;
        const int len         = (radius * 2 + 1);
        const float exp_denom = 2.f * sigma * sigma;

        // Points img to correct Gaussian pyramid layer
        const Array<T> img = gauss_pyr[octave * (n_layers + 3) + layer];
        const T* img_ptr   = img.get();

        for (int i = 0; i < OriHistBins; i++) hist[i] = 0.f;

        af::dim4 idims = img.dims();

        // Calculate orientation histogram
        for (int l = 0; l < len * len; l++) {
            int i = l / len - radius;
            int j = l % len - radius;

            int y = pt_y + i;
            int x = pt_x + j;
            if (y < 1 || y >= idims[0] - 1 || x < 1 || x >= idims[1] - 1)
                continue;

            float dx = (float)(IPTR(x + 1, y) - IPTR(x - 1, y));
            float dy = (float)(IPTR(x, y - 1) - IPTR(x, y + 1));

            float mag = sqrt(dx * dx + dy * dy);
            float ori = atan2(dy, dx);
            float w   = exp(-(i * i + j * j) / exp_denom);

            int bin = round(n * (ori + PI_VAL) / (2.f * PI_VAL));
            bin     = bin < n ? bin : 0;

            hist[bin] += w * mag;
        }

        for (int i = 0; i < SmoothOriPasses; i++) {
            for (int j = 0; j < n; j++) { temphist[j] = hist[j]; }
            for (int j = 0; j < n; j++) {
                float prev = (j == 0) ? temphist[n - 1] : temphist[j - 1];
                float next = (j + 1 == n) ? temphist[0] : temphist[j + 1];
                hist[j]    = 0.25f * prev + 0.5f * temphist[j] + 0.25f * next;
            }
        }

        float omax = hist[0];
        for (int i = 1; i < n; i++) omax = max(omax, hist[i]);

        float mag_thr = (float)(omax * OriPeakRatio);
        int l, r;
        for (int j = 0; j < n; j++) {
            l = (j == 0) ? n - 1 : j - 1;
            r = (j + 1) % n;
            if (hist[j] > hist[l] && hist[j] > hist[r] && hist[j] >= mag_thr) {
                if (*counter < max_feat) {
                    float bin = j + 0.5f * (hist[l] - hist[r]) /
                                        (hist[l] - 2.0f * hist[j] + hist[r]);
                    bin = (bin < 0.0f) ? bin + n : (bin >= n) ? bin - n : bin;
                    float ori = 360.f - ((360.f / n) * bin);

                    float new_real_x = real_x;
                    float new_real_y = real_y;
                    float new_size   = size;

                    if (double_input) {
                        float scale = 0.5f;
                        new_real_x *= scale;
                        new_real_y *= scale;
                        new_size *= scale;
                    }

                    x_out[*counter]        = new_real_x;
                    y_out[*counter]        = new_real_y;
                    layer_out[*counter]    = layer;
                    response_out[*counter] = response;
                    size_out[*counter]     = new_size;
                    ori_out[*counter]      = ori;
                    (*counter)++;
                }
            }
        }
    }
}

void normalizeDesc(float* desc, const int histlen) {
    float len_sq = 0.0f;

    for (int i = 0; i < histlen; i++) len_sq += desc[i] * desc[i];

    float len_inv = 1.0f / sqrt(len_sq);

    for (int i = 0; i < histlen; i++) { desc[i] *= len_inv; }
}

// Computes feature descriptors for features in an array.  Based on Section 6
// of Lowe's paper.
template<typename T>
void computeDescriptor(float* desc_out, const unsigned desc_len,
                       const float* x_in, const float* y_in,
                       const unsigned* layer_in, const float* response_in,
                       const float* size_in, const float* ori_in,
                       const unsigned total_feat,
                       const std::vector<Array<T>>& gauss_pyr, const int d,
                       const int n, const float scale, const unsigned octave,
                       const unsigned n_layers) {
    UNUSED(response_in);
    float desc[128];

    for (unsigned f = 0; f < total_feat; f++) {
        const unsigned layer = layer_in[f];
        float ori            = (360.f - ori_in[f]) * PI_VAL / 180.f;
        ori                  = (ori > PI_VAL) ? ori - PI_VAL * 2 : ori;
        const float size     = size_in[f];
        const int fx         = round(x_in[f] * scale);
        const int fy         = round(y_in[f] * scale);

        // Points img to correct Gaussian pyramid layer
        Array<T> img     = gauss_pyr[octave * (n_layers + 3) + layer];
        const T* img_ptr = img.get();
        af::dim4 idims   = img.dims();

        float cos_t        = cos(ori);
        float sin_t        = sin(ori);
        float bins_per_rad = n / (PI_VAL * 2.f);
        float exp_denom    = d * d * 0.5f;
        float hist_width   = DescrSclFctr * size * scale * 0.5f;
        int radius         = hist_width * sqrt(2.f) * (d + 1.f) * 0.5f + 0.5f;

        int len = radius * 2 + 1;

        for (int i = 0; i < (int)desc_len; i++) desc[i] = 0.f;

        // Calculate orientation histogram
        for (int l = 0; l < len * len; l++) {
            int i = l / len - radius;
            int j = l % len - radius;

            int y = fy + i;
            int x = fx + j;

            float x_rot = (j * cos_t - i * sin_t) / hist_width;
            float y_rot = (j * sin_t + i * cos_t) / hist_width;
            float xbin  = x_rot + d / 2 - 0.5f;
            float ybin  = y_rot + d / 2 - 0.5f;

            if (ybin > -1.0f && ybin < d && xbin > -1.0f && xbin < d && y > 0 &&
                y < idims[0] - 1 && x > 0 && x < idims[1] - 1) {
                float dx = (float)(IPTR(x + 1, y) - IPTR(x - 1, y));
                float dy = (float)(IPTR(x, y - 1) - IPTR(x, y + 1));

                float grad_mag = sqrt(dx * dx + dy * dy);
                float grad_ori = atan2(dy, dx) - ori;
                while (grad_ori < 0.0f) grad_ori += PI_VAL * 2;
                while (grad_ori >= PI_VAL * 2) grad_ori -= PI_VAL * 2;

                float w    = exp(-(x_rot * x_rot + y_rot * y_rot) / exp_denom);
                float obin = grad_ori * bins_per_rad;
                float mag  = grad_mag * w;

                int x0 = floor(xbin);
                int y0 = floor(ybin);
                int o0 = floor(obin);
                xbin -= x0;
                ybin -= y0;
                obin -= o0;

                for (int yl = 0; yl <= 1; yl++) {
                    int yb = y0 + yl;
                    if (yb >= 0 && yb < d) {
                        float v_y = mag * ((yl == 0) ? 1.0f - ybin : ybin);
                        for (int xl = 0; xl <= 1; xl++) {
                            int xb = x0 + xl;
                            if (xb >= 0 && xb < d) {
                                float v_x =
                                    v_y * ((xl == 0) ? 1.0f - xbin : xbin);
                                for (int ol = 0; ol <= 1; ol++) {
                                    int ob = (o0 + ol) % n;
                                    float v_o =
                                        v_x * ((ol == 0) ? 1.0f - obin : obin);
                                    desc[(yb * d + xb) * n + ob] += v_o;
                                }
                            }
                        }
                    }
                }
            }
        }

        normalizeDesc(desc, desc_len);

        for (int i = 0; i < (int)desc_len; i++)
            desc[i] = min(desc[i], DescrMagThr);

        normalizeDesc(desc, desc_len);

        // Calculate final descriptor values
        for (int k = 0; k < (int)desc_len; k++) {
            desc_out[f * desc_len + k] =
                round(min(255.f, desc[k] * IntDescrFctr));
        }
    }
}

// Computes GLOH feature descriptors for features in an array. Based on Section
// III-B of Mikolajczyk and Schmid paper.
template<typename T>
void computeGLOHDescriptor(float* desc_out, const unsigned desc_len,
                           const float* x_in, const float* y_in,
                           const unsigned* layer_in, const float* response_in,
                           const float* size_in, const float* ori_in,
                           const unsigned total_feat,
                           const std::vector<Array<T>>& gauss_pyr, const int d,
                           const unsigned rb, const unsigned ab,
                           const unsigned hb, const float scale,
                           const unsigned octave, const unsigned n_layers) {
    UNUSED(response_in);
    float desc[272];

    for (unsigned f = 0; f < total_feat; f++) {
        const unsigned layer = layer_in[f];
        float ori            = (360.f - ori_in[f]) * PI_VAL / 180.f;
        ori                  = (ori > PI_VAL) ? ori - PI_VAL * 2 : ori;
        const float size     = size_in[f];
        const int fx         = round(x_in[f] * scale);
        const int fy         = round(y_in[f] * scale);

        // Points img to correct Gaussian pyramid layer
        Array<T> img     = gauss_pyr[octave * (n_layers + 3) + layer];
        const T* img_ptr = img.get();
        af::dim4 idims   = img.dims();

        float cos_t              = cos(ori);
        float sin_t              = sin(ori);
        float hist_bins_per_rad  = hb / (PI_VAL * 2.f);
        float polar_bins_per_rad = ab / (PI_VAL * 2.f);
        float exp_denom          = GLOHRadii[rb - 1] * 0.5f;

        float hist_width = DescrSclFctr * size * scale * 0.5f;

        // Keep same descriptor radius used for SIFT
        int radius = hist_width * sqrt(2.f) * (d + 1.f) * 0.5f + 0.5f;

        // Alternative radius size calculation, changing the radius weight
        // (rw) in the range of 0.25f-0.75f gives different results,
        // increasing it tends to show a better recall rate but with a
        // smaller amount of correct matches
        // float rw = 0.5f;
        // int radius = hist_width * GLOHRadii[rb-1] * rw + 0.5f;

        int len = radius * 2 + 1;

        for (int i = 0; i < (int)desc_len; i++) desc[i] = 0.f;

        // Calculate orientation histogram
        for (int l = 0; l < len * len; l++) {
            int i = l / len - radius;
            int j = l % len - radius;

            int y = fy + i;
            int x = fx + j;

            float x_rot = (j * cos_t - i * sin_t);
            float y_rot = (j * sin_t + i * cos_t);

            float r = sqrt(x_rot * x_rot + y_rot * y_rot) / radius *
                      GLOHRadii[rb - 1];
            float theta = atan2(y_rot, x_rot);
            while (theta < 0.0f) theta += PI_VAL * 2;
            while (theta >= PI_VAL * 2) theta -= PI_VAL * 2;

            float tbin = theta * polar_bins_per_rad;
            float rbin =
                (r < GLOHRadii[0])
                    ? r / GLOHRadii[0]
                    : ((r < GLOHRadii[1])
                           ? 1 + (r - GLOHRadii[0]) /
                                     (float)(GLOHRadii[1] - GLOHRadii[0])
                           : min(2 + (r - GLOHRadii[1]) /
                                         (float)(GLOHRadii[2] - GLOHRadii[1]),
                                 3.f - std::numeric_limits<float>::epsilon()));

            if (r <= GLOHRadii[rb - 1] && y > 0 && y < idims[0] - 1 && x > 0 &&
                x < idims[1] - 1) {
                float dx = (float)(IPTR(x + 1, y) - IPTR(x - 1, y));
                float dy = (float)(IPTR(x, y - 1) - IPTR(x, y + 1));

                float grad_mag = sqrt(dx * dx + dy * dy);
                float grad_ori = atan2(dy, dx) - ori;
                while (grad_ori < 0.0f) grad_ori += PI_VAL * 2;
                while (grad_ori >= PI_VAL * 2) grad_ori -= PI_VAL * 2;

                float w    = exp(-r / exp_denom);
                float obin = grad_ori * hist_bins_per_rad;
                float mag  = grad_mag * w;

                int t0 = floor(tbin);
                int r0 = floor(rbin);
                int o0 = floor(obin);
                tbin -= t0;
                rbin -= r0;
                obin -= o0;

                for (int rl = 0; rl <= 1; rl++) {
                    int rb    = (rbin > 0.5f) ? (r0 + rl) : (r0 - rl);
                    float v_r = mag * ((rl == 0) ? 1.0f - rbin : rbin);
                    if (rb >= 0 && rb <= 2) {
                        for (int tl = 0; tl <= 1; tl++) {
                            int tb    = (t0 + tl) % ab;
                            float v_t = v_r * ((tl == 0) ? 1.0f - tbin : tbin);
                            for (int ol = 0; ol <= 1; ol++) {
                                int ob = (o0 + ol) % hb;
                                float v_o =
                                    v_t * ((ol == 0) ? 1.0f - obin : obin);
                                unsigned idx =
                                    (rb > 0) *
                                        (hb + ((rb - 1) * ab + tb) * hb) +
                                    ob;
                                desc[idx] += v_o;
                            }
                        }
                    }
                }
            }
        }

        normalizeDesc(desc, desc_len);

        for (int i = 0; i < (int)desc_len; i++)
            desc[i] = min(desc[i], DescrMagThr);

        normalizeDesc(desc, desc_len);

        // Calculate final descriptor values
        for (int k = 0; k < (int)desc_len; k++) {
            desc_out[f * desc_len + k] =
                round(min(255.f, desc[k] * IntDescrFctr));
        }
    }
}

#undef IPTR

template<typename T, typename convAccT>
Array<T> createInitialImage(const Array<T>& img, const float init_sigma,
                            const bool double_input) {
    af::dim4 idims = img.dims();

    Array<T> init_img = createEmptyArray<T>(af::dim4());

    float s = (double_input) ? std::max((float)sqrt(init_sigma * init_sigma -
                                                    InitSigma * InitSigma * 4),
                                        0.1f)
                             : std::max((float)sqrt(init_sigma * init_sigma -
                                                    InitSigma * InitSigma),
                                        0.1f);

    Array<T> filter = gauss_filter<T>(s);

    if (double_input) {
        Array<T> double_img =
            resize<T>(img, idims[0] * 2, idims[1] * 2, AF_INTERP_BILINEAR);
        init_img = convolve2<T, convAccT>(double_img, filter, filter, false);
    } else {
        init_img = convolve2<T, convAccT>(img, filter, filter, false);
    }

    return init_img;
}

template<typename T, typename convAccT>
std::vector<Array<T>> buildGaussPyr(const Array<T>& init_img,
                                    const unsigned n_octaves,
                                    const unsigned n_layers,
                                    const float init_sigma) {
    // Precompute Gaussian sigmas using the following formula:
    // \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    std::vector<float> sig_layers(n_layers + 3);
    sig_layers[0] = init_sigma;
    float k       = std::pow(2.0f, 1.0f / n_layers);
    for (unsigned i = 1; i < n_layers + 3; i++) {
        float sig_prev  = std::pow(k, i - 1) * init_sigma;
        float sig_total = sig_prev * k;
        sig_layers[i] = std::sqrt(sig_total * sig_total - sig_prev * sig_prev);
    }

    // Gaussian Pyramid
    std::vector<Array<T>> gauss_pyr(n_octaves * (n_layers + 3),
                                    createEmptyArray<T>(af::dim4()));
    for (unsigned o = 0; o < n_octaves; o++) {
        for (unsigned l = 0; l < n_layers + 3; l++) {
            unsigned src_idx = (l == 0) ? (o - 1) * (n_layers + 3) + n_layers
                                        : o * (n_layers + 3) + l - 1;
            unsigned idx     = o * (n_layers + 3) + l;

            if (o == 0 && l == 0) {
                gauss_pyr[idx] = init_img;
            } else if (l == 0) {
                af::dim4 sdims = gauss_pyr[src_idx].dims();
                gauss_pyr[idx] = resize<T>(gauss_pyr[src_idx], sdims[0] / 2,
                                           sdims[1] / 2, AF_INTERP_BILINEAR);
            } else {
                Array<T> filter = gauss_filter<T>(sig_layers[l]);

                gauss_pyr[idx] = convolve2<T, convAccT>(gauss_pyr[src_idx],
                                                        filter, filter, false);
            }
        }
    }

    return gauss_pyr;
}

template<typename T>
std::vector<Array<T>> buildDoGPyr(std::vector<Array<T>>& gauss_pyr,
                                  const unsigned n_octaves,
                                  const unsigned n_layers) {
    // DoG Pyramid
    std::vector<Array<T>> dog_pyr(n_octaves * (n_layers + 2),
                                  createEmptyArray<T>(af::dim4()));
    for (unsigned o = 0; o < n_octaves; o++) {
        for (unsigned l = 0; l < n_layers + 2; l++) {
            unsigned idx    = o * (n_layers + 2) + l;
            unsigned bottom = o * (n_layers + 3) + l;
            unsigned top    = o * (n_layers + 3) + l + 1;

            dog_pyr[idx] = createEmptyArray<T>(gauss_pyr[bottom].dims());

            sub<T>(dog_pyr[idx], gauss_pyr[top], gauss_pyr[bottom]);
        }
    }

    return dog_pyr;
}

template<typename T, typename convAccT>
unsigned sift_impl(Array<float>& x, Array<float>& y, Array<float>& score,
                   Array<float>& ori, Array<float>& size, Array<float>& desc,
                   const Array<T>& in, const unsigned n_layers,
                   const float contrast_thr, const float edge_thr,
                   const float init_sigma, const bool double_input,
                   const float img_scale, const float feature_ratio,
                   const bool compute_GLOH) {
    using std::function;
    using std::unique_ptr;
    using std::vector;
    af::dim4 idims = in.dims();

    unsigned min_dim = min(idims[0], idims[1]);
    if (double_input) min_dim *= 2;

    const unsigned n_octaves = floor(log(min_dim) / log(2)) - 2;

    Array<T> init_img =
        createInitialImage<T, convAccT>(in, init_sigma, double_input);

    std::vector<Array<T>> gauss_pyr =
        buildGaussPyr<T, convAccT>(init_img, n_octaves, n_layers, init_sigma);

    std::vector<Array<T>> dog_pyr =
        buildDoGPyr<T>(gauss_pyr, n_octaves, n_layers);

    vector<uptr<float>> x_pyr(n_octaves);
    vector<uptr<float>> y_pyr(n_octaves);
    vector<uptr<float>> response_pyr(n_octaves);
    vector<uptr<float>> size_pyr(n_octaves);
    vector<uptr<float>> ori_pyr(n_octaves);
    vector<uptr<float>> desc_pyr(n_octaves);
    vector<unsigned> feat_pyr(n_octaves, 0);
    unsigned total_feat = 0;

    const unsigned d  = DescrWidth;
    const unsigned n  = DescrHistBins;
    const unsigned rb = GLOHRadialBins;
    const unsigned ab = GLOHAngularBins;
    const unsigned hb = GLOHHistBins;
    const unsigned desc_len =
        (compute_GLOH) ? (1 + (rb - 1) * ab) * hb : d * d * n;

    for (unsigned i = 0; i < n_octaves; i++) {
        af::dim4 ddims = dog_pyr[i * (n_layers + 2)].dims();
        if (ddims[0] - 2 * ImgBorder < 1 || ddims[1] - 2 * ImgBorder < 1)
            continue;

        const unsigned imel     = ddims[0] * ddims[1];
        const unsigned max_feat = ceil(imel * feature_ratio);

        auto extrema_x        = memAlloc<float>(max_feat);
        auto extrema_y        = memAlloc<float>(max_feat);
        auto extrema_layer    = memAlloc<unsigned>(max_feat);
        unsigned extrema_feat = 0;

        for (unsigned j = 1; j <= n_layers; j++) {
            unsigned prev   = i * (n_layers + 2) + j - 1;
            unsigned center = i * (n_layers + 2) + j;
            unsigned next   = i * (n_layers + 2) + j + 1;

            unsigned layer = j;

            float extrema_thr = 0.5f * contrast_thr / n_layers;
            detectExtrema<T>(extrema_x.get(), extrema_y.get(),
                             extrema_layer.get(), &extrema_feat, dog_pyr[prev],
                             dog_pyr[center], dog_pyr[next], layer, max_feat,
                             extrema_thr);
        }

        extrema_feat = min(extrema_feat, max_feat);

        if (extrema_feat == 0) { continue; }

        unsigned interp_feat = 0;

        auto interp_x        = memAlloc<float>(extrema_feat);
        auto interp_y        = memAlloc<float>(extrema_feat);
        auto interp_layer    = memAlloc<unsigned>(extrema_feat);
        auto interp_response = memAlloc<float>(extrema_feat);
        auto interp_size     = memAlloc<float>(extrema_feat);

        interpolateExtrema<T>(interp_x.get(), interp_y.get(),
                              interp_layer.get(), interp_response.get(),
                              interp_size.get(), &interp_feat, extrema_x.get(),
                              extrema_y.get(), extrema_layer.get(),
                              extrema_feat, dog_pyr, max_feat, i, n_layers,
                              contrast_thr, edge_thr, init_sigma, img_scale);

        interp_feat = min(interp_feat, max_feat);

        if (interp_feat == 0) { continue; }

        std::vector<feat_t> sorted_feat;
        array_to_feat(sorted_feat, interp_x.get(), interp_y.get(),
                      interp_layer.get(), interp_response.get(),
                      interp_size.get(), interp_feat);
        std::stable_sort(sorted_feat.begin(), sorted_feat.end(), feat_cmp);

        unsigned nodup_feat = 0;

        auto nodup_x        = memAlloc<float>(interp_feat);
        auto nodup_y        = memAlloc<float>(interp_feat);
        auto nodup_layer    = memAlloc<unsigned>(interp_feat);
        auto nodup_response = memAlloc<float>(interp_feat);
        auto nodup_size     = memAlloc<float>(interp_feat);

        removeDuplicates(nodup_x.get(), nodup_y.get(), nodup_layer.get(),
                         nodup_response.get(), nodup_size.get(), &nodup_feat,
                         sorted_feat);

        const unsigned max_oriented_feat = nodup_feat * 3;

        auto oriented_x        = memAlloc<float>(max_oriented_feat);
        auto oriented_y        = memAlloc<float>(max_oriented_feat);
        auto oriented_layer    = memAlloc<unsigned>(max_oriented_feat);
        auto oriented_response = memAlloc<float>(max_oriented_feat);
        auto oriented_size     = memAlloc<float>(max_oriented_feat);
        auto oriented_ori      = memAlloc<float>(max_oriented_feat);

        unsigned oriented_feat = 0;

        calcOrientation<T>(
            oriented_x.get(), oriented_y.get(), oriented_layer.get(),
            oriented_response.get(), oriented_size.get(), oriented_ori.get(),
            &oriented_feat, nodup_x.get(), nodup_y.get(), nodup_layer.get(),
            nodup_response.get(), nodup_size.get(), nodup_feat, gauss_pyr,
            max_oriented_feat, i, n_layers, double_input);

        if (oriented_feat == 0) { continue; }

        auto desc = memAlloc<float>(oriented_feat * desc_len);

        float scale = 1.f / (1 << i);
        if (double_input) scale *= 2.f;

        if (compute_GLOH)
            computeGLOHDescriptor<T>(
                desc.get(), desc_len, oriented_x.get(), oriented_y.get(),
                oriented_layer.get(), oriented_response.get(),
                oriented_size.get(), oriented_ori.get(), oriented_feat,
                gauss_pyr, d, rb, ab, hb, scale, i, n_layers);
        else
            computeDescriptor<T>(desc.get(), desc_len, oriented_x.get(),
                                 oriented_y.get(), oriented_layer.get(),
                                 oriented_response.get(), oriented_size.get(),
                                 oriented_ori.get(), oriented_feat, gauss_pyr,
                                 d, n, scale, i, n_layers);

        total_feat += oriented_feat;
        feat_pyr[i] = oriented_feat;

        if (oriented_feat > 0) {
            x_pyr[i]        = std::move(oriented_x);
            y_pyr[i]        = std::move(oriented_y);
            response_pyr[i] = std::move(oriented_response);
            ori_pyr[i]      = std::move(oriented_ori);
            size_pyr[i]     = std::move(oriented_size);
            desc_pyr[i]     = std::move(desc);
        }
    }

    if (total_feat > 0) {
        const af::dim4 total_feat_dims(total_feat);
        const af::dim4 desc_dims(desc_len, total_feat);

        // Allocate output memory
        x     = createEmptyArray<float>(total_feat_dims);
        y     = createEmptyArray<float>(total_feat_dims);
        score = createEmptyArray<float>(total_feat_dims);
        ori   = createEmptyArray<float>(total_feat_dims);
        size  = createEmptyArray<float>(total_feat_dims);
        desc  = createEmptyArray<float>(desc_dims);

        float* x_ptr     = x.get();
        float* y_ptr     = y.get();
        float* score_ptr = score.get();
        float* ori_ptr   = ori.get();
        float* size_ptr  = size.get();
        float* desc_ptr  = desc.get();

        unsigned offset = 0;
        for (unsigned i = 0; i < n_octaves; i++) {
            if (feat_pyr[i] == 0) continue;

            memcpy(x_ptr + offset, x_pyr[i].get(), feat_pyr[i] * sizeof(float));
            memcpy(y_ptr + offset, y_pyr[i].get(), feat_pyr[i] * sizeof(float));
            memcpy(score_ptr + offset, response_pyr[i].get(),
                   feat_pyr[i] * sizeof(float));
            memcpy(ori_ptr + offset, ori_pyr[i].get(),
                   feat_pyr[i] * sizeof(float));
            memcpy(size_ptr + offset, size_pyr[i].get(),
                   feat_pyr[i] * sizeof(float));

            memcpy(desc_ptr + (offset * desc_len), desc_pyr[i].get(),
                   feat_pyr[i] * desc_len * sizeof(float));
            offset += feat_pyr[i];
        }
    }

    return total_feat;
}

}  // namespace cpu
}  // namespace arrayfire

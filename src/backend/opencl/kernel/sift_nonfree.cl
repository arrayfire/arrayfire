/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// The source code contained in this file is based on the original code by
// Rob Hess. Please note that SIFT is an algorithm patented and protected
// by US law, before using this code or any binary forms generated from it,
// verify that you have permission to do so. The original license by Rob Hess
// can be read below:
//
// Copyright (c) 2006-2012, Rob Hess <rob@iqengines.com>
// All rights reserved.
//
// The following patent has been issued for methods embodied in this
// software: "Method and apparatus for identifying scale invariant features
// in an image and use of same for locating an object in an image," David
// G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
// filed March 8, 1999. Asignee: The University of British Columbia. For
// further details, contact David Lowe (lowe@cs.ubc.ca) or the
// University-Industry Liaison Office of the University of British
// Columbia.
//
// Note that restrictions imposed by this patent (and possibly others)
// exist independently of and may be in conflict with the freedoms granted
// in this license, which refers to copyright of the program, not patents
// for any methods that it implements.  Both copyright and patent law must
// be obeyed to legally use and redistribute this program and it is not the
// purpose of this license to induce you to infringe any patents or other
// property right claims or to contest validity of any such claims.  If you
// redistribute or use the program, then this license merely protects you
// from committing copyright infringement.  It does not protect you from
// committing patent infringement.  So, before you do anything with this
// program, make sure that you have permission to do so not merely in terms
// of copyright, but also in terms of patent law.
//
// Please note that this license is not to be understood as a guarantee
// either.  If you use the program according to this license, but in
// conflict with patent law, it does not mean that the licensor will refund
// you for any losses that you incur if you are sued for your patent
// infringement.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//     * Redistributions of source code must retain the above copyright and
//       patent notices, this list of conditions and the following
//       disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in
//       the documentation and/or other materials provided with the
//       distribution.
//     * Neither the name of Oregon State University nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// width of border in which to ignore keypoints
#define IMG_BORDER 5

// maximum steps of keypoint interpolation before failure
#define MAX_INTERP_STEPS 5

// default number of bins in histogram for orientation assignment
#define ORI_HIST_BINS 36

// determines gaussian sigma for orientation assignment
#define ORI_SIG_FCTR 1.5f

// determines the radius of the region used in orientation assignment
#define ORI_RADIUS (3 * ORI_SIG_FCTR)

// number of passes of orientation histogram smoothing
#define SMOOTH_ORI_PASSES 2

// orientation magnitude relative to max that results in new feature
#define ORI_PEAK_RATIO 0.8f

// determines the size of a single descriptor orientation histogram
#define DESCR_SCL_FCTR 3.f

// threshold on magnitude of elements of descriptor vector
#define DESCR_MAG_THR 0.2f

// factor used to convert floating-point descriptor to unsigned char
#define INT_DESCR_FCTR 512.f

__constant float GLOHRadii[3] = {6.f, 11.f, 15.f};

#define PI_VAL 3.14159265358979323846f

void gaussianElimination(float* A, float* b, float* x, const int n) {
    // forward elimination
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            float s = A[j * n + i] / A[i * n + i];

            // for (int k = i+1; k < n; k++)
            for (int k = i; k < n; k++) A[j * n + k] -= s * A[i * n + k];

            b[j] -= s * b[i];
        }
    }

    for (int i = 0; i < n; i++) x[i] = 0;

    // backward substitution
    float sum = 0;
    for (int i = 0; i <= n - 2; i++) {
        sum = b[i];
        for (int j = i + 1; j < n; j++) sum -= A[i * n + j] * x[j];
        x[i] = sum / A[i * n + i];
    }
}

inline void fatomic_add(volatile local float* source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal  = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile local unsigned int*)source,
                            prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

inline void normalizeDesc(local float* desc, __local float* accum,
                          const int histlen, int lid_x, int lid_y, int lsz_x) {
    for (int i = lid_x; i < histlen; i += lsz_x)
        accum[i] = desc[lid_y * histlen + i] * desc[lid_y * histlen + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;
    for (int i = 0; i < histlen; i++)
        sum += desc[lid_y * histlen + i] * desc[lid_y * histlen + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid_x < 64) accum[lid_x] += accum[lid_x + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 32) accum[lid_x] += accum[lid_x + 32];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 16) accum[lid_x] += accum[lid_x + 16];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 8) accum[lid_x] += accum[lid_x + 8];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 4) accum[lid_x] += accum[lid_x + 4];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 2) accum[lid_x] += accum[lid_x + 2];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 1) accum[lid_x] += accum[lid_x + 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    float len_sq  = accum[0];
    float len_inv = 1.0f / sqrt(len_sq);

    for (int i = lid_x; i < histlen; i += lsz_x) {
        desc[lid_y * histlen + i] *= len_inv;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void normalizeGLOHDesc(local float* desc, __local float* accum,
                              const int histlen, int lid_x, int lid_y,
                              int lsz_x) {
    for (int i = lid_x; i < histlen; i += lsz_x)
        accum[i] = desc[lid_y * histlen + i] * desc[lid_y * histlen + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;
    for (int i = 0; i < histlen; i++)
        sum += desc[lid_y * histlen + i] * desc[lid_y * histlen + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid_x < 128) accum[lid_x] += accum[lid_x + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 64) accum[lid_x] += accum[lid_x + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 32) accum[lid_x] += accum[lid_x + 32];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 16)
        // GLOH is 272-dimensional, accumulating last 16 descriptors
        accum[lid_x] += accum[lid_x + 16] + accum[lid_x + 256];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 8) accum[lid_x] += accum[lid_x + 8];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 4) accum[lid_x] += accum[lid_x + 4];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 2) accum[lid_x] += accum[lid_x + 2];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 1) accum[lid_x] += accum[lid_x + 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    float len_sq  = accum[0];
    float len_inv = 1.0f / sqrt(len_sq);

    for (int i = lid_x; i < histlen; i += lsz_x) {
        desc[lid_y * histlen + i] *= len_inv;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void sub(global T* out, __global const T* in, unsigned nel,
                  unsigned n_layers) {
    unsigned i = get_global_id(0);

    if (i < nel) {
        for (unsigned l = 0; l < n_layers; l++)
            out[l * nel + i] = in[l * nel + i] - in[(l + 1) * nel + i];
    }
}

#define LCPTR(Y, X) (l_center[(Y)*l_i + (X)])
#define LPPTR(Y, X) (l_prev[(Y)*l_i + (X)])
#define LNPTR(Y, X) (l_next[(Y)*l_i + (X)])

// Determines whether a pixel is a scale-space extremum by comparing it to its
// 3x3x3 pixel neighborhood.
kernel void detectExtrema(global float* x_out, __global float* y_out,
                            global unsigned* layer_out,
                            global unsigned* counter, __global const T* dog,
                            KParam iDoG, const unsigned max_feat,
                            const float threshold, local float* l_mem) {
    const int dim0 = iDoG.dims[0];
    const int dim1 = iDoG.dims[1];
    const int imel = iDoG.dims[0] * iDoG.dims[1];

    const int lid_i = get_local_id(0);
    const int lid_j = get_local_id(1);
    const int lsz_i = get_local_size(0);
    const int lsz_j = get_local_size(1);
    const int i     = get_group_id(0) * lsz_i + lid_i + IMG_BORDER;
    const int j     = get_group_id(1) * lsz_j + lid_j + IMG_BORDER;

    // One pixel border for each side
    const int l_i = lsz_i + 2;
    const int l_j = lsz_j + 2;

    local float* l_prev   = l_mem;
    local float* l_center = l_mem + l_i * l_j;
    local float* l_next   = l_mem + l_i * l_j * 2;

    const int x = lid_i + 1;
    const int y = lid_j + 1;

    for (int l = 1; l < iDoG.dims[2] - 1; l++) {
        const int l_i_half = l_i / 2;
        const int l_j_half = l_j / 2;
        if (lid_i < l_i_half && lid_j < l_j_half && i < dim0 - IMG_BORDER + 1 &&
            j < dim1 - IMG_BORDER + 1) {
            l_next[lid_j * l_i + lid_i] =
                (float)dog[(l + 1) * imel + (j - 1) * dim0 + i - 1];
            l_center[lid_j * l_i + lid_i] =
                (float)dog[(l)*imel + (j - 1) * dim0 + i - 1];
            l_prev[lid_j * l_i + lid_i] =
                (float)dog[(l - 1) * imel + (j - 1) * dim0 + i - 1];

            l_next[lid_j * l_i + lid_i + l_i_half] =
                (float)dog[(l + 1) * imel + (j - 1) * dim0 + i - 1 + l_i_half];
            l_center[lid_j * l_i + lid_i + l_i_half] =
                (float)dog[(l)*imel + (j - 1) * dim0 + i - 1 + l_i_half];
            l_prev[lid_j * l_i + lid_i + l_i_half] =
                (float)dog[(l - 1) * imel + (j - 1) * dim0 + i - 1 + l_i_half];

            l_next[(lid_j + l_j_half) * l_i + lid_i] =
                (float)dog[(l + 1) * imel + (j - 1 + l_j_half) * dim0 + i - 1];
            l_center[(lid_j + l_j_half) * l_i + lid_i] =
                (float)dog[(l)*imel + (j - 1 + l_j_half) * dim0 + i - 1];
            l_prev[(lid_j + l_j_half) * l_i + lid_i] =
                (float)dog[(l - 1) * imel + (j - 1 + l_j_half) * dim0 + i - 1];

            l_next[(lid_j + l_j_half) * l_i + lid_i + l_i_half] =
                (float)dog[(l + 1) * imel + (j - 1 + l_j_half) * dim0 + i - 1 +
                           l_i_half];
            l_center[(lid_j + l_j_half) * l_i + lid_i + l_i_half] = (float)
                dog[(l)*imel + (j - 1 + l_j_half) * dim0 + i - 1 + l_i_half];
            l_prev[(lid_j + l_j_half) * l_i + lid_i + l_i_half] =
                (float)dog[(l - 1) * imel + (j - 1 + l_j_half) * dim0 + i - 1 +
                           l_i_half];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (i < dim0 - IMG_BORDER && j < dim1 - IMG_BORDER) {
            const int l_i_half = l_i / 2;
            float p            = l_center[y * l_i + x];

            if (fabs((float)p) > threshold &&
                ((p > 0 && p > LCPTR(y - 1, x - 1) && p > LCPTR(y - 1, x) &&
                  p > LCPTR(y - 1, x + 1) && p > LCPTR(y, x - 1) &&
                  p > LCPTR(y, x + 1) && p > LCPTR(y + 1, x - 1) &&
                  p > LCPTR(y + 1, x) && p > LCPTR(y + 1, x + 1) &&
                  p > LPPTR(y - 1, x - 1) && p > LPPTR(y - 1, x) &&
                  p > LPPTR(y - 1, x + 1) && p > LPPTR(y, x - 1) &&
                  p > LPPTR(y, x) && p > LPPTR(y, x + 1) &&
                  p > LPPTR(y + 1, x - 1) && p > LPPTR(y + 1, x) &&
                  p > LPPTR(y + 1, x + 1) && p > LNPTR(y - 1, x - 1) &&
                  p > LNPTR(y - 1, x) && p > LNPTR(y - 1, x + 1) &&
                  p > LNPTR(y, x - 1) && p > LNPTR(y, x) &&
                  p > LNPTR(y, x + 1) && p > LNPTR(y + 1, x - 1) &&
                  p > LNPTR(y + 1, x) && p > LNPTR(y + 1, x + 1)) ||
                 (p < 0 && p < LCPTR(y - 1, x - 1) && p < LCPTR(y - 1, x) &&
                  p < LCPTR(y - 1, x + 1) && p < LCPTR(y, x - 1) &&
                  p < LCPTR(y, x + 1) && p < LCPTR(y + 1, x - 1) &&
                  p < LCPTR(y + 1, x) && p < LCPTR(y + 1, x + 1) &&
                  p < LPPTR(y - 1, x - 1) && p < LPPTR(y - 1, x) &&
                  p < LPPTR(y - 1, x + 1) && p < LPPTR(y, x - 1) &&
                  p < LPPTR(y, x) && p < LPPTR(y, x + 1) &&
                  p < LPPTR(y + 1, x - 1) && p < LPPTR(y + 1, x) &&
                  p < LPPTR(y + 1, x + 1) && p < LNPTR(y - 1, x - 1) &&
                  p < LNPTR(y - 1, x) && p < LNPTR(y - 1, x + 1) &&
                  p < LNPTR(y, x - 1) && p < LNPTR(y, x) &&
                  p < LNPTR(y, x + 1) && p < LNPTR(y + 1, x - 1) &&
                  p < LNPTR(y + 1, x) && p < LNPTR(y + 1, x + 1)))) {
                unsigned idx = atomic_inc(counter);
                if (idx < max_feat) {
                    x_out[idx]     = (float)j;
                    y_out[idx]     = (float)i;
                    layer_out[idx] = l;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

#undef LCPTR
#undef LPPTR
#undef LNPTR
#define CPTR(Y, X) (center[(Y)*dim0 + (X)])
#define PPTR(Y, X) (prev[(Y)*dim0 + (X)])
#define NPTR(Y, X) (next[(Y)*dim0 + (X)])

// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
kernel void interpolateExtrema(
    global float* x_out, __global float* y_out, __global unsigned* layer_out,
    global float* response_out, __global float* size_out,
    global unsigned* counter, __global const float* x_in,
    global const float* y_in, __global const unsigned* layer_in,
    const unsigned extrema_feat, global const T* dog_octave, KParam iDoG,
    const unsigned max_feat, const unsigned octave, const unsigned n_layers,
    const float contrast_thr, const float edge_thr, const float sigma,
    const float img_scale) {
    const unsigned f = get_global_id(0);

    if (f < extrema_feat) {
        const float first_deriv_scale  = img_scale * 0.5f;
        const float second_deriv_scale = img_scale;
        const float cross_deriv_scale  = img_scale * 0.25f;

        float xl = 0, xy = 0, xx = 0, contr = 0;
        int i = 0;

        unsigned x     = x_in[f];
        unsigned y     = y_in[f];
        unsigned layer = layer_in[f];

        const int dim0 = iDoG.dims[0];
        const int dim1 = iDoG.dims[1];
        const int imel = dim0 * dim1;

        global const T* prev   = dog_octave + (int)((layer - 1) * imel);
        global const T* center = dog_octave + (int)((layer)*imel);
        global const T* next   = dog_octave + (int)((layer + 1) * imel);

        for (i = 0; i < MAX_INTERP_STEPS; i++) {
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
            gaussianElimination(H, dD, X, 3);

            xl = -X[2];
            xy = -X[1];
            xx = -X[0];

            if (fabs(xl) < 0.5f && fabs(xy) < 0.5f && fabs(xx) < 0.5f) break;

            x += round(xx);
            y += round(xy);
            layer += round(xl);

            if (layer < 1 || layer > n_layers || x < IMG_BORDER ||
                x >= dim1 - IMG_BORDER || y < IMG_BORDER ||
                y >= dim0 - IMG_BORDER)
                return;
        }

        // ensure convergence of interpolation
        if (i >= MAX_INTERP_STEPS) return;

        float dD[3] = {
            (float)(CPTR(x + 1, y) - CPTR(x - 1, y)) * first_deriv_scale,
            (float)(CPTR(x, y + 1) - CPTR(x, y - 1)) * first_deriv_scale,
            (float)(NPTR(x, y) - PPTR(x, y)) * first_deriv_scale};
        float X[3] = {xx, xy, xl};

        float P = dD[0] * X[0] + dD[1] * X[1] + dD[2] * X[2];

        contr = center[x * dim0 + y] * img_scale + P * 0.5f;
        if (fabs(contr) < (contrast_thr / n_layers)) return;

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
        if (det <= 0 || tr * tr * edge_thr >=
                            (edge_thr + 1) * (edge_thr + 1) * det + FLT_EPSILON)
            return;

        unsigned ridx = atomic_inc(counter);

        if (ridx < max_feat) {
            x_out[ridx]        = (x + xx) * (1 << octave);
            y_out[ridx]        = (y + xy) * (1 << octave);
            layer_out[ridx]    = layer;
            response_out[ridx] = fabs(contr);
            size_out[ridx] =
                sigma * pow(2.f, octave + (layer + xl) / n_layers) * 2.f;
        }
    }
}

#undef CPTR
#undef PPTR
#undef NPTR

// Remove duplicate keypoints
kernel void removeDuplicates(
    global float* x_out, __global float* y_out, __global unsigned* layer_out,
    global float* response_out, __global float* size_out,
    global unsigned* counter, __global const float* x_in,
    global const float* y_in, __global const unsigned* layer_in,
    global const float* response_in, __global const float* size_in,
    const unsigned total_feat) {
    const unsigned f = get_global_id(0);

    if (f < total_feat) {
        const float prec_fctr = 1e4f;

        bool cond = (f < total_feat - 1)
                        ? !(round(x_in[f] * prec_fctr) ==
                                round(x_in[f + 1] * prec_fctr) &&
                            round(y_in[f] * prec_fctr) ==
                                round(y_in[f + 1] * prec_fctr) &&
                            layer_in[f] == layer_in[f + 1] &&
                            round(response_in[f] * prec_fctr) ==
                                round(response_in[f + 1] * prec_fctr) &&
                            round(size_in[f] * prec_fctr) ==
                                round(size_in[f + 1] * prec_fctr))
                        : true;

        if (cond) {
            unsigned idx = atomic_inc(counter);

            x_out[idx]        = x_in[f];
            y_out[idx]        = y_in[f];
            layer_out[idx]    = layer_in[f];
            response_out[idx] = response_in[f];
            size_out[idx]     = size_in[f];
        }
    }
}

#define IPTR(Y, X) (img[(Y)*dim0 + X])

// Computes a canonical orientation for each image feature in an array.  Based
// on Section 5 of Lowe's paper.  This function adds features to the array when
// there is more than one dominant orientation at a given feature location.
kernel void calcOrientation(
    global float* x_out, __global float* y_out, __global unsigned* layer_out,
    global float* response_out, __global float* size_out,
    global float* ori_out, __global unsigned* counter,
    global const float* x_in, __global const float* y_in,
    global const unsigned* layer_in, __global const float* response_in,
    global const float* size_in, const unsigned total_feat,
    global const T* gauss_octave, KParam iGauss, const unsigned max_feat,
    const unsigned octave, const int double_input, local float* l_mem) {
    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int lsz_x = get_local_size(0);

    const unsigned f = get_global_id(1);

    const int n = ORI_HIST_BINS;

    local float* hist     = l_mem;
    local float* temphist = l_mem + n * 8;

    // Initialize temporary histogram
    for (int i = lid_x; i < n; i += lsz_x) { hist[lid_y * n + i] = 0.f; }
    barrier(CLK_LOCAL_MEM_FENCE);

    float real_x, real_y, response, size;
    unsigned layer;

    if (f < total_feat) {
        // Load keypoint information
        real_x   = x_in[f];
        real_y   = y_in[f];
        layer    = layer_in[f];
        response = response_in[f];
        size     = size_in[f];

        const int pt_x = (int)round(real_x / (1 << octave));
        const int pt_y = (int)round(real_y / (1 << octave));

        // Calculate auxiliary parameters
        const float scl_octv  = size * 0.5f / (1 << octave);
        const int radius      = (int)round(ORI_RADIUS * scl_octv);
        const float sigma     = ORI_SIG_FCTR * scl_octv;
        const int len         = (radius * 2 + 1);
        const float exp_denom = 2.f * sigma * sigma;

        const int dim0 = iGauss.dims[0];
        const int dim1 = iGauss.dims[1];

        // Calculate layer offset
        const int layer_offset = layer * dim0 * dim1;
        global const T* img  = gauss_octave + layer_offset;

        // Calculate orientation histogram
        for (int l = lid_x; l < len * len; l += lsz_x) {
            int i = l / len - radius;
            int j = l % len - radius;

            int y = pt_y + i;
            int x = pt_x + j;
            if (y < 1 || y >= dim0 - 1 || x < 1 || x >= dim1 - 1) continue;

            float dx = (float)(IPTR(x + 1, y) - IPTR(x - 1, y));
            float dy = (float)(IPTR(x, y - 1) - IPTR(x, y + 1));

            float mag = sqrt(dx * dx + dy * dy);
            float ori = atan2(dy, dx);
            float w   = exp(-(i * i + j * j) / exp_denom);

            int bin = round(n * (ori + PI_VAL) / (2.f * PI_VAL));
            bin     = bin < n ? bin : 0;
            bin     = (bin < 0) ? 0 : (bin >= n) ? n - 1 : bin;

            fatomic_add(&hist[lid_y * n + bin], w * mag);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < SMOOTH_ORI_PASSES; i++) {
        for (int j = lid_x; j < n; j += lsz_x) {
            temphist[lid_y * n + j] = hist[lid_y * n + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = lid_x; j < n; j += lsz_x) {
            float prev = (j == 0) ? temphist[lid_y * n + n - 1]
                                  : temphist[lid_y * n + j - 1];
            float next = (j + 1 == n) ? temphist[lid_y * n]
                                      : temphist[lid_y * n + j + 1];
            hist[lid_y * n + j] =
                0.25f * prev + 0.5f * temphist[lid_y * n + j] + 0.25f * next;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = lid_x; i < n; i += lsz_x)
        temphist[lid_y * n + i] = hist[lid_y * n + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid_x < 16)
        temphist[lid_y * n + lid_x] =
            fmax(hist[lid_y * n + lid_x], hist[lid_y * n + lid_x + 16]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 8)
        temphist[lid_y * n + lid_x] =
            fmax(temphist[lid_y * n + lid_x], temphist[lid_y * n + lid_x + 8]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 4) {
        temphist[lid_y * n + lid_x] =
            fmax(temphist[lid_y * n + lid_x], hist[lid_y * n + lid_x + 32]);
        temphist[lid_y * n + lid_x] =
            fmax(temphist[lid_y * n + lid_x], temphist[lid_y * n + lid_x + 4]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 2)
        temphist[lid_y * n + lid_x] =
            fmax(temphist[lid_y * n + lid_x], temphist[lid_y * n + lid_x + 2]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid_x < 1)
        temphist[lid_y * n + lid_x] =
            fmax(temphist[lid_y * n + lid_x], temphist[lid_y * n + lid_x + 1]);
    barrier(CLK_LOCAL_MEM_FENCE);
    float omax = temphist[lid_y * n];

    if (f < total_feat) {
        float mag_thr = (float)(omax * ORI_PEAK_RATIO);
        int l, r;
        float bin;
        for (int j = lid_x; j < n; j += lsz_x) {
            l = (j == 0) ? n - 1 : j - 1;
            r = (j + 1) % n;
            if (hist[lid_y * n + j] > hist[lid_y * n + l] &&
                hist[lid_y * n + j] > hist[lid_y * n + r] &&
                hist[lid_y * n + j] >= mag_thr) {
                unsigned idx = atomic_inc(counter);

                if (idx < max_feat) {
                    float bin =
                        j +
                        0.5f * (hist[lid_y * n + l] - hist[lid_y * n + r]) /
                            (hist[lid_y * n + l] - 2.0f * hist[lid_y * n + j] +
                             hist[lid_y * n + r]);
                    bin = (bin < 0.0f) ? bin + n : (bin >= n) ? bin - n : bin;
                    float ori = 360.f - ((360.f / n) * bin);

                    float new_real_x = real_x;
                    float new_real_y = real_y;
                    float new_size   = size;

                    if (double_input != 0) {
                        float scale = 0.5f;
                        new_real_x *= scale;
                        new_real_y *= scale;
                        new_size *= scale;
                    }

                    x_out[idx]        = new_real_x;
                    y_out[idx]        = new_real_y;
                    layer_out[idx]    = layer;
                    response_out[idx] = response;
                    size_out[idx]     = new_size;
                    ori_out[idx]      = ori;
                }
            }
        }
    }
}

// Computes feature descriptors for features in an array.  Based on Section 6
// of Lowe's paper.
kernel void computeDescriptor(
    global float* desc_out, const unsigned desc_len, const unsigned histsz,
    global const float* x_in, __global const float* y_in,
    global const unsigned* layer_in, __global const float* response_in,
    global const float* size_in, __global const float* ori_in,
    const unsigned total_feat, global const T* gauss_octave, KParam iGauss,
    const int d, const int n, const float scale, const int n_layers,
    local float* l_mem) {
    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int lsz_x = get_local_size(0);

    const int f = get_global_id(1);

    local float* desc  = l_mem;
    local float* accum = l_mem + desc_len * histsz;

    for (int i = lid_x; i < desc_len * histsz; i += lsz_x)
        desc[lid_y * desc_len + i] = 0.f;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (f < total_feat) {
        const unsigned layer = layer_in[f];
        float ori            = (360.f - ori_in[f]) * PI_VAL / 180.f;
        ori                  = (ori > PI_VAL) ? ori - PI_VAL * 2 : ori;
        const float size     = size_in[f];
        const int fx         = round(x_in[f] * scale);
        const int fy         = round(y_in[f] * scale);

        // Points img to correct Gaussian pyramid layer
        const int dim0        = iGauss.dims[0];
        const int dim1        = iGauss.dims[1];
        global const T* img = gauss_octave + (layer * dim0 * dim1);

        float cos_t        = cos(ori);
        float sin_t        = sin(ori);
        float bins_per_rad = n / (PI_VAL * 2.f);
        float exp_denom    = d * d * 0.5f;
        float hist_width   = DESCR_SCL_FCTR * size * scale * 0.5f;
        int radius         = hist_width * sqrt(2.f) * (d + 1.f) * 0.5f + 0.5f;

        int len            = radius * 2 + 1;
        const int hist_off = (lid_x % histsz) * desc_len;

        // Calculate orientation histogram
        for (int l = lid_x; l < len * len; l += lsz_x) {
            int i = l / len - radius;
            int j = l % len - radius;

            int y = fy + i;
            int x = fx + j;

            float x_rot = (j * cos_t - i * sin_t) / hist_width;
            float y_rot = (j * sin_t + i * cos_t) / hist_width;
            float xbin  = x_rot + d / 2 - 0.5f;
            float ybin  = y_rot + d / 2 - 0.5f;

            if (ybin > -1.0f && ybin < d && xbin > -1.0f && xbin < d && y > 0 &&
                y < dim0 - 1 && x > 0 && x < dim1 - 1) {
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
                                    fatomic_add(
                                        &desc[hist_off + lid_y * desc_len +
                                              (yb * d + xb) * n + ob],
                                        v_o);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Combine histograms (reduces previous atomicAdd overhead)
    for (int l = lid_x; l < desc_len * 4; l += lsz_x)
        desc[l] += desc[l + 4 * desc_len];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int l = lid_x; l < desc_len * 2; l += lsz_x)
        desc[l] += desc[l + 2 * desc_len];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int l = lid_x; l < desc_len; l += lsz_x) desc[l] += desc[l + desc_len];
    barrier(CLK_LOCAL_MEM_FENCE);

    normalizeDesc(desc, accum, desc_len, lid_x, lid_y, lsz_x);

    for (int i = lid_x; i < d * d * n; i += lsz_x)
        desc[lid_y * desc_len + i] =
            min(desc[lid_y * desc_len + i], DESCR_MAG_THR);
    barrier(CLK_LOCAL_MEM_FENCE);

    normalizeDesc(desc, accum, desc_len, lid_x, lid_y, lsz_x);

    if (f < total_feat) {
        // Calculate final descriptor values
        for (int k = lid_x; k < d * d * n; k += lsz_x)
            desc_out[f * desc_len + k] =
                round(min(255.f, desc[lid_y * desc_len + k] * INT_DESCR_FCTR));
    }
}

kernel void computeGLOHDescriptor(
    global float* desc_out, const unsigned desc_len, const unsigned histsz,
    global const float* x_in, __global const float* y_in,
    global const unsigned* layer_in, __global const float* response_in,
    global const float* size_in, __global const float* ori_in,
    const unsigned total_feat, global const T* gauss_octave, KParam iGauss,
    const int d, const unsigned rb, const unsigned ab, const unsigned hb,
    const float scale, const int n_layers, local float* l_mem) {
    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int lsz_x = get_local_size(0);

    const int f = get_global_id(1);

    local float* desc  = l_mem;
    local float* accum = l_mem + desc_len * histsz;

    for (int i = lid_x; i < desc_len * histsz; i += lsz_x)
        desc[lid_y * desc_len + i] = 0.f;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (f < total_feat) {
        const unsigned layer = layer_in[f];
        float ori            = (360.f - ori_in[f]) * PI_VAL / 180.f;
        ori                  = (ori > PI_VAL) ? ori - PI_VAL * 2 : ori;
        const float size     = size_in[f];
        const int fx         = round(x_in[f] * scale);
        const int fy         = round(y_in[f] * scale);

        // Points img to correct Gaussian pyramid layer
        const int dim0        = iGauss.dims[0];
        const int dim1        = iGauss.dims[1];
        global const T* img = gauss_octave + (layer * dim0 * dim1);

        float cos_t              = cos(ori);
        float sin_t              = sin(ori);
        float hist_bins_per_rad  = hb / (PI_VAL * 2.f);
        float polar_bins_per_rad = ab / (PI_VAL * 2.f);
        float exp_denom          = GLOHRadii[rb - 1] * 0.5f;

        float hist_width = DESCR_SCL_FCTR * size * scale * 0.5f;

        // Keep same descriptor radius used for SIFT
        int radius = hist_width * sqrt(2.f) * (d + 1.f) * 0.5f + 0.5f;

        // Alternative radius size calculation, changing the radius weight
        // (rw) in the range of 0.25f-0.75f gives different results,
        // increasing it tends to show a better recall rate but with a
        // smaller amount of correct matches
        // float rw = 0.5f;
        // int radius = hist_width * GLOHRadii[rb-1] * rw + 0.5f;

        int len            = radius * 2 + 1;
        const int hist_off = (lid_x % histsz) * desc_len;

        // Calculate orientation histogram
        for (int l = lid_x; l < len * len; l += lsz_x) {
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
                                 3.f - FLT_EPSILON));

            if (r <= GLOHRadii[rb - 1] && y > 0 && y < dim0 - 1 && x > 0 &&
                x < dim1 - 1) {
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
                                fatomic_add(
                                    &desc[hist_off + lid_y * desc_len + idx],
                                    v_o);
                            }
                        }
                    }
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Combine histograms (reduces previous atomicAdd overhead)
    for (int l = lid_x; l < desc_len * 4; l += lsz_x)
        desc[l] += desc[l + 4 * desc_len];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int l = lid_x; l < desc_len * 2; l += lsz_x)
        desc[l] += desc[l + 2 * desc_len];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int l = lid_x; l < desc_len; l += lsz_x) desc[l] += desc[l + desc_len];
    barrier(CLK_LOCAL_MEM_FENCE);

    normalizeGLOHDesc(desc, accum, desc_len, lid_x, lid_y, lsz_x);

    for (int i = lid_x; i < desc_len; i += lsz_x)
        desc[lid_y * desc_len + i] =
            min(desc[lid_y * desc_len + i], DESCR_MAG_THR);
    barrier(CLK_LOCAL_MEM_FENCE);

    normalizeGLOHDesc(desc, accum, desc_len, lid_x, lid_y, lsz_x);

    if (f < total_feat) {
        // Calculate final descriptor values
        for (int k = lid_x; k < desc_len; k += lsz_x)
            desc_out[f * desc_len + k] =
                round(min(255.f, desc[lid_y * desc_len + k] * INT_DESCR_FCTR));
    }
}

#undef IPTR

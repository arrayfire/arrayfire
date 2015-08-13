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

#define PI_VAL 3.14159265358979323846f

void gaussianElimination(float* A, float* b, float* x, const int n)
{
    // forward elimination
    for (int i = 0; i < n-1; i++) {
        for (int j = i+1; j < n; j++) {
            float s = A[j*n+i] / A[i*n+i];

            //for (int k = i+1; k < n; k++)
            for (int k = i; k < n; k++)
                A[j*n+k] -= s * A[i*n+k];

            b[j] -= s * b[i];
        }
    }

    for (int i = 0; i < n; i++)
        x[i] = 0;

    // backward substitution
    float sum = 0;
    for (int i = 0; i <= n-2; i++) {
        sum = b[i];
        for (int j = i+1; j < n; j++)
            sum -= A[i*n+j] * x[j];
        x[i] = sum / A[i*n+i];
    }
}

inline void fatomic_add(volatile __local float *source, const float operand) {
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
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __local unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

inline void normalizeDesc(
    __local float* desc,
    __local float* accum,
    const int histlen,
    int tid_x,
    int tid_y,
    int bsz_y)
{
    for (int i = tid_y; i < histlen; i += bsz_y)
        accum[tid_y] = desc[tid_x*histlen+i]*desc[tid_x*histlen+i];
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;
    for (int i = 0; i < histlen; i++)
        sum += desc[tid_x*histlen+i]*desc[tid_x*histlen+i];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid_y < 64)
        accum[tid_y] += accum[tid_y+64];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid_y < 32)
        accum[tid_y] += accum[tid_y+32];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid_y < 16)
        accum[tid_y] += accum[tid_y+16];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid_y < 8)
        accum[tid_y] += accum[tid_y+8];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid_y < 4)
        accum[tid_y] += accum[tid_y+4];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid_y < 2)
        accum[tid_y] += accum[tid_y+2];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid_y < 1)
        accum[tid_y] += accum[tid_y+1];
    barrier(CLK_LOCAL_MEM_FENCE);

    float len_sq = accum[0];
    float len_inv = 1.0f / sqrt(len_sq);

    for (int i = tid_y; i < histlen; i += bsz_y) {
        desc[tid_x*histlen+i] *= len_inv;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void sub(
    __global T* out,
    __global const T* in,
    unsigned nel,
    unsigned n_layers)
{
    unsigned i = get_global_id(0);

    for (unsigned l = 0; l < n_layers; l++)
        out[l*nel + i] = in[l*nel + i] - in[(l+1)*nel + i];
}

#define LCPTR(Y, X) (l_center[(Y) * l_i + (X)])
#define LPPTR(Y, X) (l_prev[(Y) * l_i + (X)])
#define LNPTR(Y, X) (l_next[(Y) * l_i + (X)])

// Determines whether a pixel is a scale-space extremum by comparing it to its
// 3x3x3 pixel neighborhood.
__kernel void detectExtrema(
    __global float* x_out,
    __global float* y_out,
    __global unsigned* layer_out,
    __global unsigned* counter,
    __global const T* dog,
    KParam iDoG,
    const unsigned max_feat,
    const float threshold)
{
    // One pixel border for each side
    const int l_i = 32+2;
    const int l_j = 8+2;

    __local float l_mem[(32+2)*(8+2)*3];
    __local float* l_prev   = l_mem;
    __local float* l_center = l_mem + (32+2)*(8+2);
    __local float* l_next   = l_mem + (32+2)*(8+2)*2;
    __local float* l_tmp;

    const int dim0 = iDoG.dims[0];
    const int dim1 = iDoG.dims[1];
    const int imel = iDoG.dims[0]*iDoG.dims[1];

    const int lid_i = get_local_id(0);
    const int lid_j = get_local_id(1);
    const int lsz_i = get_local_size(0);
    const int lsz_j = get_local_size(1);
    const int i = get_group_id(0) * lsz_i + lid_i+IMG_BORDER;
    const int j = get_group_id(1) * lsz_j + lid_j+IMG_BORDER;

    const int x = lid_i+1;
    const int y = lid_j+1;

    for (int l = 1; l < iDoG.dims[2]-1; l++) {
        const int l_i_half = l_i/2;
        const int l_j_half = l_j/2;
        if (lid_i < l_i_half && lid_j < l_j_half && i < dim0-IMG_BORDER+1 && j < dim1-IMG_BORDER+1) {
                l_next  [lid_j*l_i + lid_i] = dog[(l+1)*imel+(j-1)*dim0+i-1];
                l_center[lid_j*l_i + lid_i] = dog[(l  )*imel+(j-1)*dim0+i-1];
                l_prev  [lid_j*l_i + lid_i] = dog[(l-1)*imel+(j-1)*dim0+i-1];

                l_next  [lid_j*l_i + lid_i+l_i_half] = dog[(l+1)*imel+(j-1)*dim0+i-1+l_i_half];
                l_center[lid_j*l_i + lid_i+l_i_half] = dog[(l  )*imel+(j-1)*dim0+i-1+l_i_half];
                l_prev  [lid_j*l_i + lid_i+l_i_half] = dog[(l-1)*imel+(j-1)*dim0+i-1+l_i_half];

                l_next  [(lid_j+l_j_half)*l_i + lid_i] = dog[(l+1)*imel+(j-1+l_j_half)*dim0+i-1];
                l_center[(lid_j+l_j_half)*l_i + lid_i] = dog[(l  )*imel+(j-1+l_j_half)*dim0+i-1];
                l_prev  [(lid_j+l_j_half)*l_i + lid_i] = dog[(l-1)*imel+(j-1+l_j_half)*dim0+i-1];

                l_next  [(lid_j+l_j_half)*l_i + lid_i+l_i_half] = dog[(l+1)*imel+(j-1+l_j_half)*dim0+i-1+l_i_half];
                l_center[(lid_j+l_j_half)*l_i + lid_i+l_i_half] = dog[(l  )*imel+(j-1+l_j_half)*dim0+i-1+l_i_half];
                l_prev  [(lid_j+l_j_half)*l_i + lid_i+l_i_half] = dog[(l-1)*imel+(j-1+l_j_half)*dim0+i-1+l_i_half];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (i < dim0-IMG_BORDER && j < dim1-IMG_BORDER) {
            const int l_i_half = l_i/2;
            float p = l_center[y*l_i + x];

            if (fabs((float)p) > threshold &&
                ((p > 0 && p > LCPTR(y-1, x-1) && p > LCPTR(y-1, x) &&
                  p > LCPTR(y-1, x+1) && p > LCPTR(y, x-1) && p > LCPTR(y,   x+1)  &&
                  p > LCPTR(y+1, x-1) && p > LCPTR(y+1, x) && p > LCPTR(y+1, x+1)  &&
                  p > LPPTR(y-1, x-1) && p > LPPTR(y-1, x) && p > LPPTR(y-1, x+1)  &&
                  p > LPPTR(y,   x-1) && p > LPPTR(y  , x) && p > LPPTR(y,   x+1)  &&
                  p > LPPTR(y+1, x-1) && p > LPPTR(y+1, x) && p > LPPTR(y+1, x+1)  &&
                  p > LNPTR(y-1, x-1) && p > LNPTR(y-1, x) && p > LNPTR(y-1, x+1)  &&
                  p > LNPTR(y,   x-1) && p > LNPTR(y  , x) && p > LNPTR(y,   x+1)  &&
                  p > LNPTR(y+1, x-1) && p > LNPTR(y+1, x) && p > LNPTR(y+1, x+1)) ||
                 (p < 0 && p < LCPTR(y-1, x-1) && p < LCPTR(y-1, x) &&
                  p < LCPTR(y-1, x+1) && p < LCPTR(y, x-1) && p < LCPTR(y,   x+1)  &&
                  p < LCPTR(y+1, x-1) && p < LCPTR(y+1, x) && p < LCPTR(y+1, x+1)  &&
                  p < LPPTR(y-1, x-1) && p < LPPTR(y-1, x) && p < LPPTR(y-1, x+1)  &&
                  p < LPPTR(y,   x-1) && p < LPPTR(y  , x) && p < LPPTR(y,   x+1)  &&
                  p < LPPTR(y+1, x-1) && p < LPPTR(y+1, x) && p < LPPTR(y+1, x+1)  &&
                  p < LNPTR(y-1, x-1) && p < LNPTR(y-1, x) && p < LNPTR(y-1, x+1)  &&
                  p < LNPTR(y,   x-1) && p < LNPTR(y  , x) && p < LNPTR(y,   x+1)  &&
                  p < LNPTR(y+1, x-1) && p < LNPTR(y+1, x) && p < LNPTR(y+1, x+1)))) {

                unsigned idx = atomic_inc(counter);
                if (idx < max_feat)
                {
                    x_out[idx] = (float)j;
                    y_out[idx] = (float)i;
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
#define CPTR(Y, X) (center[(Y) * dim0 + (X)])
#define PPTR(Y, X) (prev[(Y) * dim0 + (X)])
#define NPTR(Y, X) (next[(Y) * dim0 + (X)])

// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
__kernel void interpolateExtrema(
    __global float* x_out,
    __global float* y_out,
    __global unsigned* layer_out,
    __global float* response_out,
    __global float* size_out,
    __global unsigned* counter,
    __global const float* x_in,
    __global const float* y_in,
    __global const unsigned* layer_in,
    const unsigned extrema_feat,
    __global const T* dog_octave,
    KParam iDoG,
    const unsigned max_feat,
    const unsigned octave,
    const unsigned n_layers,
    const float contrast_thr,
    const float edge_thr,
    const float sigma,
    const float img_scale)
{
    const unsigned f = get_global_id(0);

    if (f < extrema_feat)
    {
        const float first_deriv_scale = img_scale*0.5f;
        const float second_deriv_scale = img_scale;
        const float cross_deriv_scale = img_scale*0.25f;

        float xl = 0, xy = 0, xx = 0, contr = 0;
        int i = 0;

        unsigned x = x_in[f];
        unsigned y = y_in[f];
        unsigned layer = layer_in[f];

        const int dim0 = iDoG.dims[0];
        const int dim1 = iDoG.dims[1];
        const int imel = dim0 * dim1;

        __global const T* prev   = dog_octave + (int)((layer-1)*imel);
        __global const T* center = dog_octave + (int)((layer  )*imel);
        __global const T* next   = dog_octave + (int)((layer+1)*imel);

        for(i = 0; i < MAX_INTERP_STEPS; i++) {
            float dD[3] = {(float)(CPTR(x+1, y) - CPTR(x-1, y)) * first_deriv_scale,
                           (float)(CPTR(x, y+1) - CPTR(x, y-1)) * first_deriv_scale,
                           (float)(NPTR(x, y)   - PPTR(x, y))   * first_deriv_scale};

            float d2  = CPTR(x, y) * 2.f;
            float dxx = (CPTR(x+1, y) + CPTR(x-1, y) - d2) * second_deriv_scale;
            float dyy = (CPTR(x, y+1) + CPTR(x, y-1) - d2) * second_deriv_scale;
            float dss = (NPTR(x, y  ) + PPTR(x, y  ) - d2) * second_deriv_scale;
            float dxy = (CPTR(x+1, y+1) - CPTR(x-1, y+1) -
                         CPTR(x+1, y-1) + CPTR(x-1, y-1)) * cross_deriv_scale;
            float dxs = (NPTR(x+1, y) - NPTR(x-1, y) -
                         PPTR(x+1, y) + PPTR(x-1, y)) * cross_deriv_scale;
            float dys = (NPTR(x, y+1) - NPTR(x-1, y-1) -
                         PPTR(x, y-1) + PPTR(x-1, y-1)) * cross_deriv_scale;

            float H[9] = {dxx, dxy, dxs,
                          dxy, dyy, dys,
                          dxs, dys, dss};

            float X[3];
            gaussianElimination(H, dD, X, 3);

            xl = -X[2];
            xy = -X[1];
            xx = -X[0];

            if(fabs(xl) < 0.5f && fabs(xy) < 0.5f && fabs(xx) < 0.5f)
                break;

            x += round(xx);
            y += round(xy);
            layer += round(xl);

            if(layer < 1 || layer > n_layers ||
               x < IMG_BORDER || x >= dim1 - IMG_BORDER ||
               y < IMG_BORDER || y >= dim0 - IMG_BORDER)
                return;
        }

        // ensure convergence of interpolation
        if (i >= MAX_INTERP_STEPS)
            return;

        float dD[3] = {(float)(CPTR(x+1, y) - CPTR(x-1, y)) * first_deriv_scale,
                       (float)(CPTR(x, y+1) - CPTR(x, y-1)) * first_deriv_scale,
                       (float)(NPTR(x, y)   - PPTR(x, y))   * first_deriv_scale};
        float X[3] = {xx, xy, xl};

        float P = dD[0]*X[0] + dD[1]*X[1] + dD[2]*X[2];

        contr = center[x*dim0+y]*img_scale + P * 0.5f;
        if(fabs(contr) < (contrast_thr / n_layers))
            return;

        // principal curvatures are computed using the trace and det of Hessian
        float d2  = CPTR(x, y) * 2.f;
        float dxx = (CPTR(x+1, y) + CPTR(x-1, y) - d2) * second_deriv_scale;
        float dyy = (CPTR(x, y+1) + CPTR(x, y-1) - d2) * second_deriv_scale;
        float dxy = (CPTR(x+1, y+1) - CPTR(x-1, y+1) -
                     CPTR(x+1, y-1) + CPTR(x-1, y-1)) * cross_deriv_scale;

        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        // add FLT_EPSILON for double-precision compatibility
        if (det <= 0 || tr*tr*edge_thr >= (edge_thr + 1)*(edge_thr + 1)*det+FLT_EPSILON)
            return;

        unsigned ridx = atomic_inc(counter);

        if (ridx < max_feat)
        {
            x_out[ridx] = (x + xx) * (1 << octave);
            y_out[ridx] = (y + xy) * (1 << octave);
            layer_out[ridx] = layer;
            response_out[ridx] = fabs(contr);
            size_out[ridx] = sigma*pow(2.f, octave + (layer + xl) / n_layers);
        }
    }
}

#undef CPTR
#undef PPTR
#undef NPTR

// Remove duplicate keypoints
__kernel void removeDuplicates(
    __global float* x_out,
    __global float* y_out,
    __global unsigned* layer_out,
    __global float* response_out,
    __global float* size_out,
    __global unsigned* counter,
    __global const float* x_in,
    __global const float* y_in,
    __global const unsigned* layer_in,
    __global const float* response_in,
    __global const float* size_in,
    const unsigned total_feat)
{
    const unsigned f = get_global_id(0);

    if (f < total_feat) {
        const float prec_fctr = 1e4f;

        bool cond = (f < total_feat-1)
                    ? !(round(x_in[f]*prec_fctr) == round(x_in[f+1]*prec_fctr) &&
                        round(y_in[f]*prec_fctr) == round(y_in[f+1]*prec_fctr) &&
                        layer_in[f] == layer_in[f+1] &&
                        round(response_in[f]*prec_fctr) == round(response_in[f+1]*prec_fctr) &&
                        round(size_in[f]*prec_fctr) == round(size_in[f+1]*prec_fctr))
                    : true;

        if (cond) {
            unsigned idx = atomic_inc(counter);

            x_out[idx] = x_in[f];
            y_out[idx] = y_in[f];
            layer_out[idx] = layer_in[f];
            response_out[idx] = response_in[f];
            size_out[idx] = size_in[f];
        }
    }

}

#define IPTR(Y, X) (img[(Y) * dim0 + X])

// Computes a canonical orientation for each image feature in an array.  Based
// on Section 5 of Lowe's paper.  This function adds features to the array when
// there is more than one dominant orientation at a given feature location.
__kernel void calcOrientation(
    __global float* x_out,
    __global float* y_out,
    __global unsigned* layer_out,
    __global float* response_out,
    __global float* size_out,
    __global float* ori_out,
    __global unsigned* counter,
    __global const float* x_in,
    __global const float* y_in,
    __global const unsigned* layer_in,
    __global const float* response_in,
    __global const float* size_in,
    const unsigned total_feat,
    __global const T* gauss_octave,
    KParam iGauss,
    const unsigned max_feat,
    const unsigned octave,
    const int double_input)
{
    const unsigned f = get_global_id(0);
    const int tid_x = get_local_id(0);
    const int tid_y = get_local_id(1);
    const int bsz_y = get_local_size(1);

    const int n = ORI_HIST_BINS;

    const int hdim = ORI_HIST_BINS;
    const int thdim = ORI_HIST_BINS;
    __local float hist[ORI_HIST_BINS*8];
    __local float temphist[ORI_HIST_BINS*8];

    if (f < total_feat) {
        // Load keypoint information
        const float real_x = x_in[f];
        const float real_y = y_in[f];
        const unsigned layer = layer_in[f];
        const float response = response_in[f];
        const float size = size_in[f];

        const int pt_x = (int)round(real_x / (1 << octave));
        const int pt_y = (int)round(real_y / (1 << octave));

        // Calculate auxiliary parameters
        const float scl_octv = size*0.5f / (1 << octave);
        const int radius = (int)round(ORI_RADIUS * scl_octv);
        const float sigma = ORI_SIG_FCTR * scl_octv;
        const int len = (radius*2+1);
        const float exp_denom = 2.f * sigma * sigma;

        // Initialize temporary histogram
        for (int i = tid_y; i < ORI_HIST_BINS; i += bsz_y) {
            hist[tid_x*hdim + i] = 0.f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int dim0 = iGauss.dims[0];
        const int dim1 = iGauss.dims[1];

        // Calculate layer offset
        const int layer_offset = layer * dim0 * dim1;
        __global const T* img = gauss_octave + layer_offset;

        // Calculate orientation histogram
        for (int l = tid_y; l < len*len; l += bsz_y) {
            int i = l / len - radius;
            int j = l % len - radius;

            int y = pt_y + i;
            int x = pt_x + j;
            if (y < 1 || y >= dim0 - 1 ||
                x < 1 || x >= dim1 - 1)
                continue;

            float dx = (float)(IPTR(x+1, y) - IPTR(x-1, y));
            float dy = (float)(IPTR(x, y-1) - IPTR(x, y+1));

            float mag = sqrt(dx*dx+dy*dy);
            float ori = atan2(dy,dx);
            float w = exp(-(i*i + j*j)/exp_denom);

            int bin = round(n*(ori+PI_VAL)/(2.f*PI_VAL));
            bin = bin < n ? bin : 0;
            bin = (bin < 0) ? 0 : (bin >= n) ? n-1 : bin;

            fatomic_add(&hist[tid_x*hdim+bin], w*mag);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < SMOOTH_ORI_PASSES; i++) {
            for (int j = tid_y; j < n; j += bsz_y) {
                temphist[tid_x*hdim+j] = hist[tid_x*hdim+j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int j = tid_y; j < n; j += bsz_y) {
                float prev = (j == 0) ? temphist[tid_x*hdim+n-1] : temphist[tid_x*hdim+j-1];
                float next = (j+1 == n) ? temphist[tid_x*hdim] : temphist[tid_x*hdim+j+1];
                hist[tid_x*hdim+j] = 0.25f * prev + 0.5f * temphist[tid_x*hdim+j] + 0.25f * next;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for (int i = tid_y; i < n; i += bsz_y)
            temphist[tid_x*hdim+i] = hist[tid_x*hdim+i];
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid_y < 16)
            temphist[tid_x*thdim+tid_y] = fmax(hist[tid_x*hdim+tid_y], hist[tid_x*hdim+tid_y+16]);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid_y < 8)
            temphist[tid_x*thdim+tid_y] = fmax(temphist[tid_x*thdim+tid_y], temphist[tid_x*thdim+tid_y+8]);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid_y < 4) {
            temphist[tid_x*thdim+tid_y] = fmax(temphist[tid_x*thdim+tid_y], hist[tid_x*hdim+tid_y+32]);
            temphist[tid_x*thdim+tid_y] = fmax(temphist[tid_x*thdim+tid_y], temphist[tid_x*thdim+tid_y+4]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid_y < 2)
            temphist[tid_x*thdim+tid_y] = fmax(temphist[tid_x*thdim+tid_y], temphist[tid_x*thdim+tid_y+2]);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid_y < 1)
            temphist[tid_x*thdim+tid_y] = fmax(temphist[tid_x*thdim+tid_y], temphist[tid_x*thdim+tid_y+1]);
        barrier(CLK_LOCAL_MEM_FENCE);
        float omax = temphist[tid_x*thdim];

        float mag_thr = (float)(omax * ORI_PEAK_RATIO);
        int l, r;
        float bin;
        for (int j = tid_y; j < n; j+=bsz_y) {
            l = (j == 0) ? n - 1 : j - 1;
            r = (j + 1) % n;
            if (hist[tid_x*hdim+j] > hist[tid_x*hdim+l] &&
                hist[tid_x*hdim+j] > hist[tid_x*hdim+r] &&
                hist[tid_x*hdim+j] >= mag_thr) {
                unsigned idx = atomic_inc(counter);

                if (idx < max_feat) {
                    float bin = j + 0.5f * (hist[tid_x*hdim+l] - hist[tid_x*hdim+r]) /
                                (hist[tid_x*hdim+l] - 2.0f*hist[tid_x*hdim+j] + hist[tid_x*hdim+r]);
                    bin = (bin < 0.0f) ? bin + n : (bin >= n) ? bin - n : bin;
                    float ori = 360.f - ((360.f/n) * bin);

                    float new_real_x = real_x;
                    float new_real_y = real_y;
                    float new_size = size;

                    if (double_input != 0) {
                        float scale = 0.5f;
                        new_real_x *= scale;
                        new_real_y *= scale;
                        new_size *= scale;
                    }

                    x_out[idx] = new_real_x;
                    y_out[idx] = new_real_y;
                    layer_out[idx] = layer;
                    response_out[idx] = response;
                    size_out[idx] = new_size;
                    ori_out[idx] = ori;
                }
            }
        }
    }
}

// Computes feature descriptors for features in an array.  Based on Section 6
// of Lowe's paper.
__kernel void computeDescriptor(
    __global float* desc_out,
    const unsigned desc_len,
    __global const float* x_in,
    __global const float* y_in,
    __global const unsigned* layer_in,
    __global const float* response_in,
    __global const float* size_in,
    __global const float* ori_in,
    const unsigned total_feat,
    __global const T* gauss_octave,
    KParam iGauss,
    const int d,
    const int n,
    const float scale,
    const float sigma,
    const int n_layers)
{
    const int f = get_global_id(0);
    const int tid_x = get_local_id(0);
    const int tid_y = get_local_id(1);
    const int bsz_x = get_local_size(0);
    const int bsz_y = get_local_size(1);

    const int histsz = 8;
    __local float desc[128*8];
    __local float accum[128];

    if (f < total_feat) {
        const unsigned layer = layer_in[f];
        float ori = (360.f - ori_in[f]) * PI_VAL / 180.f;
        ori = (ori > PI_VAL) ? ori - PI_VAL*2 : ori;
        const float size = size_in[f];
        const int fx = round(x_in[f] * scale);
        const int fy = round(y_in[f] * scale);

        // Points img to correct Gaussian pyramid layer
        const int dim0 = iGauss.dims[0];
        const int dim1 = iGauss.dims[1];
        __global const T* img = gauss_octave + (layer * dim0 * dim1);

        float cos_t = cos(ori);
        float sin_t = sin(ori);
        float bins_per_rad = n / (PI_VAL * 2.f);
        float exp_denom = d * d * 0.5f;
        float hist_width = DESCR_SCL_FCTR * sigma * pow(2.f, layer/n_layers);
        int radius = hist_width * sqrt(2.f) * (d + 1.f) * 0.5f + 0.5f;

        int len = radius*2+1;
        const int histlen = d*d*n;
        const int hist_off = (tid_y % histsz) * 128;

        for (int i = tid_y; i < histlen*histsz; i += bsz_y)
            desc[tid_x*histlen+i] = 0.f;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Calculate orientation histogram
        for (int l = tid_y; l < len*len; l += bsz_y) {
            int i = l / len - radius;
            int j = l % len - radius;

            int y = fy + i;
            int x = fx + j;

            float x_rot = (j * cos_t - i * sin_t) / hist_width;
            float y_rot = (j * sin_t + i * cos_t) / hist_width;
            float xbin = x_rot + d/2 - 0.5f;
            float ybin = y_rot + d/2 - 0.5f;

            if (ybin > -1.0f && ybin < d && xbin > -1.0f && xbin < d &&
                y > 0 && y < dim0 - 1 && x > 0 && x < dim1 - 1) {
                float dx = (float)(IPTR(x+1, y) - IPTR(x-1, y));
                float dy = (float)(IPTR(x, y-1) - IPTR(x, y+1));

                float grad_mag = sqrt(dx*dx + dy*dy);
                float grad_ori = atan2(dy, dx) - ori;
                while (grad_ori < 0.0f)
                    grad_ori += PI_VAL*2;
                while (grad_ori >= PI_VAL*2)
                    grad_ori -= PI_VAL*2;

                float w = exp(-(x_rot*x_rot + y_rot*y_rot) / exp_denom);
                float obin = grad_ori * bins_per_rad;
                float mag = grad_mag*w;

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
		                        float v_x = v_y * ((xl == 0) ? 1.0f - xbin : xbin);
		                        for (int ol = 0; ol <= 1; ol++) {
		                            int ob = (o0 + ol) % n;
		                            float v_o = v_x * ((ol == 0) ? 1.0f - obin : obin);
		                            fatomic_add(&desc[hist_off + tid_x*128 + (yb*d + xb)*n + ob], v_o);
		                        }
		                    }
	                    }
	                }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Combine histograms (reduces previous atomicAdd overhead)
        for (int l = tid_y; l < 128*4; l += bsz_y)
            desc[l] += desc[l+4*128];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int l = tid_y; l < 128*2; l += bsz_y)
            desc[l    ] += desc[l+2*128];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int l = tid_y; l < 128; l += bsz_y)
            desc[l] += desc[l+128];
        barrier(CLK_LOCAL_MEM_FENCE);

        normalizeDesc(desc, accum, histlen, tid_x, tid_y, bsz_y);

        for (int i = tid_y; i < d*d*n; i += bsz_y)
            desc[tid_x*128+i] = min(desc[tid_x*128+i], DESCR_MAG_THR);
        barrier(CLK_LOCAL_MEM_FENCE);

        normalizeDesc(desc, accum, histlen, tid_x, tid_y, bsz_y);

        // Calculate final descriptor values
        for (int k = tid_y; k < d*d*n; k += bsz_y) {
            desc_out[f*desc_len+k] = round(min(255.f, desc[tid_x*128+k] * INT_DESCR_FCTR));
        }
    }
}

#undef IPTR

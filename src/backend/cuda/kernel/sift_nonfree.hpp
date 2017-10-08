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

#include <common/dispatch.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <memory.hpp>
#include "shared.hpp"

#include "convolve.hpp"
#include "resize.hpp"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/random.h>

#include <cfloat>

namespace cuda
{

namespace kernel
{

static const dim_t SIFT_THREADS   = 256;
static const dim_t SIFT_THREADS_X = 32;
static const dim_t SIFT_THREADS_Y = 8;

#define PI_VAL 3.14159265358979323846f

// default width of descriptor histogram array
#define DESCR_WIDTH 4

// default number of bins per histogram in descriptor array
#define DESCR_HIST_BINS 8

// assumed gaussian blur for input image
#define INIT_SIGMA 0.5f

// width of border in which to ignore keypoints
#define IMG_BORDER 5

// maximum steps of keypointerpolation before failure
#define MAX_INTERP_STEPS 5

// default number of bins in histogram for orientation assignment
#define ORI_HIST_BINS 36

// determines gaussian sigma for orientation assignment
#define ORI_SIG_FCTR 1.5f

// determines the radius of the region used in orientation assignment */
#define ORI_RADIUS (3.0f * ORI_SIG_FCTR)

// number of passes of orientation histogram smoothing
#define SMOOTH_ORI_PASSES 2

// orientation magnitude relative to max that results in new feature
#define ORI_PEAK_RATIO 0.8f

// determines the size of a single descriptor orientation histogram
#define DESCR_SCL_FCTR 3.f

// threshold on magnitude of elements of descriptor vector
#define DESC_MAG_THR 0.2f

// factor used to convert floating-podescriptor to unsigned char
#define INT_DESCR_FCTR 512.f

// Number of GLOH bins in radial direction
static const unsigned GLOHRadialBins = 3;

// Radii of GLOH descriptors
__constant__ float GLOHRadii[GLOHRadialBins] = {6.f, 11.f, 15.f};

// Number of GLOH angular bins (excluding the inner-most radial section)
static const unsigned GLOHAngularBins = 8;

// Number of GLOH bins per histogram in descriptor
static const unsigned GLOHHistBins = 16;

template<typename T>
void gaussian1D(T* out, const int dim, double sigma=0.0)
{
    if(!(sigma>0)) sigma = 0.25*dim;

    T sum = (T)0;
    for(int i=0;i<dim;i++)
    {
        int x = i-(dim-1)/2;
        T el = 1. / sqrt(2 * PI_VAL * sigma*sigma) * exp(-((x*x)/(2*(sigma*sigma))));
        out[i] = el;
        sum   += el;
    }

    for(int k=0;k<dim;k++)
        out[k] /= sum;
}

template<typename T>
Param<T> gauss_filter(float sigma)
{
    // Using 6-sigma rule
    unsigned gauss_len = std::min((unsigned)round(sigma * 6 + 1) | 1, 31u);

    T* h_gauss = new T[gauss_len];
    gaussian1D(h_gauss, gauss_len, sigma);
    Param<T> gauss_filter;
    gauss_filter.dims[0] = gauss_len;
    gauss_filter.strides[0] = 1;

    for (int k = 1; k < 4; k++) {
        gauss_filter.dims[k] = 1;
        gauss_filter.strides[k] = gauss_filter.dims[k-1] * gauss_filter.strides[k-1];
    }

    dim_t gauss_elem = gauss_filter.strides[3] * gauss_filter.dims[3];
    gauss_filter.ptr = memAlloc<T>(gauss_elem);
    CUDA_CHECK(cudaMemcpyAsync(gauss_filter.ptr, h_gauss, gauss_elem * sizeof(T),
                cudaMemcpyHostToDevice, cuda::getActiveStream()));
    CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));

    delete[] h_gauss;

    return gauss_filter;
}

template<int N>
__inline__ __device__ void gaussianElimination(float* A, float* b, float* x)
{
    // forward elimination
    #pragma unroll
    for (int i = 0; i < N-1; i++) {
        #pragma unroll
        for (int j = i+1; j < N; j++) {
            float s = A[j*N+i] / A[i*N+i];

            #pragma unroll
            for (int k = i; k < N; k++)
                A[j*N+k] -= s * A[i*N+k];

            b[j] -= s * b[i];
        }
    }

    #pragma unroll
    for (int i = 0; i < N; i++)
        x[i] = 0;

    // backward substitution
    float sum = 0;
    #pragma unroll
    for (int i = 0; i <= N-2; i++) {
        sum = b[i];
        #pragma unroll
        for (int j = i+1; j < N; j++)
            sum -= A[i*N+j] * x[j];
        x[i] = sum / A[i*N+i];
    }
}

__inline__ __device__ void normalizeDesc(
    float* desc,
    float* accum,
    const int histlen)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int bsz_x = blockDim.x;

    for (int i = tid_x; i < histlen; i += bsz_x)
        accum[i] = desc[tid_y*histlen+i]*desc[tid_y*histlen+i];
    __syncthreads();

    if (tid_x < 64)
        accum[tid_x] += accum[tid_x+64];
    __syncthreads();
    if (tid_x < 32)
        accum[tid_x] += accum[tid_x+32];
    __syncthreads();
    if (tid_x < 16)
        accum[tid_x] += accum[tid_x+16];
    __syncthreads();
    if (tid_x < 8)
        accum[tid_x] += accum[tid_x+8];
    __syncthreads();
    if (tid_x < 4)
        accum[tid_x] += accum[tid_x+4];
    __syncthreads();
    if (tid_x < 2)
        accum[tid_x] += accum[tid_x+2];
    __syncthreads();
    if (tid_x < 1)
        accum[tid_x] += accum[tid_x+1];
    __syncthreads();

    float len_sq = accum[0];
    float len_inv = 1.0f / sqrtf(len_sq);

    for (int i = tid_x; i < histlen; i += bsz_x) {
        desc[tid_y*histlen+i] *= len_inv;
    }
    __syncthreads();
}

__inline__ __device__ void normalizeGLOHDesc(
    float* desc,
    float* accum,
    const int histlen)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int bsz_x = blockDim.x;

    for (int i = tid_x; i < histlen; i += bsz_x)
        accum[i] = desc[tid_y*histlen+i]*desc[tid_y*histlen+i];
    __syncthreads();

    if (tid_x < 128)
        accum[tid_x] += accum[tid_x+128];
    __syncthreads();
    if (tid_x < 64)
        accum[tid_x] += accum[tid_x+64];
    __syncthreads();
    if (tid_x < 32)
        accum[tid_x] += accum[tid_x+32];
    __syncthreads();
    if (tid_x < 16)
        // GLOH is 272-dimensional, accumulating last 16 descriptors
        accum[tid_x] += accum[tid_x+16] + accum[tid_x+256];
    __syncthreads();
    if (tid_x < 8)
        accum[tid_x] += accum[tid_x+8];
    __syncthreads();
    if (tid_x < 4)
        accum[tid_x] += accum[tid_x+4];
    __syncthreads();
    if (tid_x < 2)
        accum[tid_x] += accum[tid_x+2];
    __syncthreads();
    if (tid_x < 1)
        accum[tid_x] += accum[tid_x+1];
    __syncthreads();

    float len_sq = accum[0];
    float len_inv = 1.0f / sqrtf(len_sq);

    for (int i = tid_x; i < histlen; i += bsz_x) {
        desc[tid_y*histlen+i] *= len_inv;
    }
    __syncthreads();
}

template<typename T>
__global__ void sub(
    Param<T> out,
    CParam<T> in,
    const unsigned nel,
    const unsigned n_layers)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nel) {
        for (unsigned l = 0; l < n_layers; l++)
            out.ptr[l*nel + i] = in.ptr[(l+1)*nel + i] - in.ptr[l*nel + i];
    }
}

#define SCPTR(Y, X) (s_center[(Y) * s_i + (X)])
#define SPPTR(Y, X) (s_prev[(Y) * s_i + (X)])
#define SNPTR(Y, X) (s_next[(Y) * s_i + (X)])
#define DPTR(Z, Y, X) (dog.ptr[(Z) * imel + (Y) * dim0 + (X)])

// Determines whether a pixel is a scale-space extremum by comparing it to its
// 3x3x3 pixel neighborhood.
template<typename T>
__global__ void detectExtrema(
    float* x_out,
    float* y_out,
    unsigned* layer_out,
    unsigned* counter,
    CParam<T> dog,
    const unsigned max_feat,
    const float threshold)
{
    const int dim0 = dog.dims[0];
    const int dim1 = dog.dims[1];
    const int imel = dim0 * dim1;

    const int tid_i = threadIdx.x;
    const int tid_j = threadIdx.y;
    const int bsz_i = blockDim.x;
    const int bsz_j = blockDim.y;
    const int i = blockIdx.x * bsz_i + tid_i+IMG_BORDER;
    const int j = blockIdx.y * bsz_j + tid_j+IMG_BORDER;

    const int x = tid_i+1;
    const int y = tid_j+1;

    // One pixel border for each side
    const int s_i = bsz_i+2;
    const int s_j = bsz_j+2;

    SharedMemory<float> shared;
    float* shrdMem = shared.getPointer();
    float* s_next   = shrdMem;
    float* s_center = shrdMem + s_i * s_j;
    float* s_prev   = shrdMem + s_i * s_j * 2;

    for (int l = 1; l < dog.dims[2]-1; l++) {
        const int s_i_half = s_i/2;
        const int s_j_half = s_j/2;
        if (tid_i < s_i_half && tid_j < s_j_half && i < dim0-IMG_BORDER+1 && j < dim1-IMG_BORDER+1) {
            SNPTR(tid_j, tid_i) = DPTR(l+1, j-1, i-1);
            SCPTR(tid_j, tid_i) = DPTR(l,   j-1, i-1);
            SPPTR(tid_j, tid_i) = DPTR(l-1, j-1, i-1);

            SNPTR(tid_j, tid_i+s_i_half) = DPTR((l+1), j-1, i-1+s_i_half);
            SCPTR(tid_j, tid_i+s_i_half) = DPTR((l  ), j-1, i-1+s_i_half);
            SPPTR(tid_j, tid_i+s_i_half) = DPTR((l-1), j-1, i-1+s_i_half);

            SNPTR(tid_j+s_j_half, tid_i) = DPTR(l+1, j-1+s_j_half, i-1);
            SCPTR(tid_j+s_j_half, tid_i) = DPTR(l,   j-1+s_j_half, i-1);
            SPPTR(tid_j+s_j_half, tid_i) = DPTR(l-1, j-1+s_j_half, i-1);

            SNPTR(tid_j+s_j_half, tid_i+s_i_half) = DPTR(l+1, j-1+s_j_half, i-1+s_i_half);
            SCPTR(tid_j+s_j_half, tid_i+s_i_half) = DPTR(l,   j-1+s_j_half, i-1+s_i_half);
            SPPTR(tid_j+s_j_half, tid_i+s_i_half) = DPTR(l-1, j-1+s_j_half, i-1+s_i_half);
        }
        __syncthreads();

        float p = SCPTR(y, x);

        if (abs(p) > threshold && i < dim0-IMG_BORDER && j < dim1-IMG_BORDER &&
            ((p > 0 && p > SCPTR(y-1, x-1) && p > SCPTR(y-1, x) &&
              p > SCPTR(y-1, x+1) && p > SCPTR(y, x-1) && p > SCPTR(y,   x+1)  &&
              p > SCPTR(y+1, x-1) && p > SCPTR(y+1, x) && p > SCPTR(y+1, x+1)  &&
              p > SPPTR(y-1, x-1) && p > SPPTR(y-1, x) && p > SPPTR(y-1, x+1)  &&
              p > SPPTR(y,   x-1) && p > SPPTR(y  , x) && p > SPPTR(y,   x+1)  &&
              p > SPPTR(y+1, x-1) && p > SPPTR(y+1, x) && p > SPPTR(y+1, x+1)  &&
              p > SNPTR(y-1, x-1) && p > SNPTR(y-1, x) && p > SNPTR(y-1, x+1)  &&
              p > SNPTR(y,   x-1) && p > SNPTR(y  , x) && p > SNPTR(y,   x+1)  &&
              p > SNPTR(y+1, x-1) && p > SNPTR(y+1, x) && p > SNPTR(y+1, x+1)) ||
             (p < 0 && p < SCPTR(y-1, x-1) && p < SCPTR(y-1, x) &&
              p < SCPTR(y-1, x+1) && p < SCPTR(y, x-1) && p < SCPTR(y,   x+1)  &&
              p < SCPTR(y+1, x-1) && p < SCPTR(y+1, x) && p < SCPTR(y+1, x+1)  &&
              p < SPPTR(y-1, x-1) && p < SPPTR(y-1, x) && p < SPPTR(y-1, x+1)  &&
              p < SPPTR(y,   x-1) && p < SPPTR(y  , x) && p < SPPTR(y,   x+1)  &&
              p < SPPTR(y+1, x-1) && p < SPPTR(y+1, x) && p < SPPTR(y+1, x+1)  &&
              p < SNPTR(y-1, x-1) && p < SNPTR(y-1, x) && p < SNPTR(y-1, x+1)  &&
              p < SNPTR(y,   x-1) && p < SNPTR(y  , x) && p < SNPTR(y,   x+1)  &&
              p < SNPTR(y+1, x-1) && p < SNPTR(y+1, x) && p < SNPTR(y+1, x+1)))) {

            unsigned idx = atomicAdd(counter, 1u);
            if (idx < max_feat)
            {
                x_out[idx] = (float)j;
                y_out[idx] = (float)i;
                layer_out[idx] = l;
            }
        }
        __syncthreads();
    }
}

#undef SCPTR
#undef SPPTR
#undef SNPTR
#define CPTR(Y, X) (center_ptr[(Y) * dim0 + (X)])
#define PPTR(Y, X) (prev_ptr[(Y) * dim0 + (X)])
#define NPTR(Y, X) (next_ptr[(Y) * dim0 + (X)])

// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
template<typename T>
__global__ void interpolateExtrema(
    float* x_out,
    float* y_out,
    unsigned* layer_out,
    float* response_out,
    float* size_out,
    unsigned* counter,
    const float* x_in,
    const float* y_in,
    const unsigned* layer_in,
    const unsigned extrema_feat,
    const CParam<T> dog_octave,
    const unsigned max_feat,
    const unsigned octave,
    const unsigned n_layers,
    const float contrast_thr,
    const float edge_thr,
    const float sigma,
    const float img_scale)
{
    const unsigned f = blockIdx.x * blockDim.x + threadIdx.x;

    if (f < extrema_feat) {
        const float first_deriv_scale = img_scale*0.5f;
        const float second_deriv_scale = img_scale;
        const float cross_deriv_scale = img_scale*0.25f;

        float xl = 0, xy = 0, xx = 0, contr = 0;
        int i = 0;

        unsigned x = x_in[f];
        unsigned y = y_in[f];
        unsigned layer = layer_in[f];

        const int dim0 = dog_octave.dims[0];
        const int dim1 = dog_octave.dims[1];
        const int imel = dim0 * dim1;

        const T* prev_ptr   = dog_octave.ptr + (layer-1) * imel;
        const T* center_ptr = dog_octave.ptr + (layer)   * imel;
        const T* next_ptr   = dog_octave.ptr + (layer+1) * imel;

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
            gaussianElimination<3>(H, dD, X);

            xl = -X[2];
            xy = -X[1];
            xx = -X[0];

            if (abs(xl) < 0.5f && abs(xy) < 0.5f && abs(xx) < 0.5f)
                break;

            x += round(xx);
            y += round(xy);
            layer += round(xl);

            if (layer < 1 || layer > n_layers ||
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

        contr = CPTR(x, y)*img_scale + P * 0.5f;
        if (abs(contr) < (contrast_thr / n_layers))
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

        unsigned ridx = atomicAdd(counter, 1u);

        if (ridx < max_feat)
        {
            x_out[ridx] = (x + xx) * (1 << octave);
            y_out[ridx] = (y + xy) * (1 << octave);
            layer_out[ridx] = layer;
            response_out[ridx] = abs(contr);
            size_out[ridx] = sigma*pow(2.f, octave + (layer + xl) / n_layers) * 2.f;
        }
    }
}

#undef CPTR
#undef PPTR
#undef NPTR

// Remove duplicate keypoints
__global__ void removeDuplicates(
    float* x_out,
    float* y_out,
    unsigned* layer_out,
    float* response_out,
    float* size_out,
    unsigned* counter,
    const float* x_in,
    const float* y_in,
    const unsigned* layer_in,
    const float* response_in,
    const float* size_in,
    const unsigned total_feat)
{
    const unsigned f = blockIdx.x * blockDim.x + threadIdx.x;

    if (f >= total_feat)
        return;

    float prec_fctr = 1e4f;

    if (f < total_feat-1) {
        if (round(x_in[f]*prec_fctr) == round(x_in[f+1]*prec_fctr) &&
            round(y_in[f]*prec_fctr) == round(y_in[f+1]*prec_fctr) &&
            layer_in[f] == layer_in[f+1] &&
            round(response_in[f]*prec_fctr) == round(response_in[f+1]*prec_fctr) &&
            round(size_in[f]*prec_fctr) == round(size_in[f+1]*prec_fctr))
            return;
    }

    unsigned idx = atomicAdd(counter, 1);

    x_out[idx] = x_in[f];
    y_out[idx] = y_in[f];
    layer_out[idx] = layer_in[f];
    response_out[idx] = response_in[f];
    size_out[idx] = size_in[f];
}

#define IPTR(Y, X) (img_ptr[(Y) * dim0 + (X)])

// Computes a canonical orientation for each image feature in an array.  Based
// on Section 5 of Lowe's paper.  This function adds features to the array when
// there is more than one dominant orientation at a given feature location.
template<typename T>
__global__ void calcOrientation(
    float* x_out,
    float* y_out,
    unsigned* layer_out,
    float* response_out,
    float* size_out,
    float* ori_out,
    unsigned* counter,
    const float* x_in,
    const float* y_in,
    const unsigned* layer_in,
    const float* response_in,
    const float* size_in,
    const unsigned total_feat,
    const CParam<T> gauss_octave,
    const unsigned max_feat,
    const unsigned octave,
    const bool double_input)
{
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int bsz_x = blockDim.x;
    const int bsz_y = blockDim.y;

    const unsigned f = blockIdx.y * bsz_y + tid_y;

    const int n = ORI_HIST_BINS;

    SharedMemory<float> shared;
    float* shrdMem = shared.getPointer();
    float* hist = shrdMem;
    float* temphist = shrdMem + n*8;

    // Initialize temporary histogram
    for (int i = tid_x; i < ORI_HIST_BINS; i += bsz_x)
        hist[tid_y*n + i] = 0.f;
    __syncthreads();

    float real_x, real_y, response, size;
    unsigned layer;

    if (f < total_feat) {
        // Load keypoint information
        real_x = x_in[f];
        real_y = y_in[f];
        layer = layer_in[f];
        response = response_in[f];
        size = size_in[f];

        const int pt_x = (int)round(real_x / (1 << octave));
        const int pt_y = (int)round(real_y / (1 << octave));

        // Calculate auxiliary parameters
        const float scl_octv = size*0.5f / (1 << octave);
        const int radius = (int)round(ORI_RADIUS * scl_octv);
        const float sigma = ORI_SIG_FCTR * scl_octv;
        const int len = (radius*2+1);
        const float exp_denom = 2.f * sigma * sigma;

        const int dim0 = gauss_octave.dims[0];
        const int dim1 = gauss_octave.dims[1];
        const int imel = dim0 * dim1;

        // Points img to correct Gaussian pyramid layer
        const T* img_ptr = gauss_octave.ptr + layer * imel;

        // Calculate orientation histogram
        for (int l = tid_x; l < len*len; l += bsz_x) {
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

            atomicAdd(&hist[tid_y*n+bin], w*mag);
        }
    }
    __syncthreads();

    for (int i = 0; i < SMOOTH_ORI_PASSES; i++) {
        for (int j = tid_x; j < n; j += bsz_x) {
            temphist[tid_y*n+j] = hist[tid_y*n+j];
        }
        __syncthreads();
        for (int j = tid_x; j < n; j += bsz_x) {
            float prev = (j == 0) ? temphist[tid_y*n+n-1] : temphist[tid_y*n+j-1];
            float next = (j+1 == n) ? temphist[tid_y*n] : temphist[tid_y*n+j+1];
            hist[tid_y*n+j] = 0.25f * prev + 0.5f * temphist[tid_y*n+j] + 0.25f * next;
        }
        __syncthreads();
    }

    for (int i = tid_x; i < n; i += bsz_x)
        temphist[tid_y*n+i] = hist[tid_y*n+i];
    __syncthreads();

    if (tid_x < 16)
        temphist[tid_y*n+tid_x] = fmax(hist[tid_y*n+tid_x], hist[tid_y*n+tid_x+16]);
    __syncthreads();
    if (tid_x < 8)
        temphist[tid_y*n+tid_x] = fmax(temphist[tid_y*n+tid_x], temphist[tid_y*n+tid_x+8]);
    __syncthreads();
    if (tid_x < 4) {
        temphist[tid_y*n+tid_x] = fmax(temphist[tid_y*n+tid_x], hist[tid_y*n+tid_x+32]);
        temphist[tid_y*n+tid_x] = fmax(temphist[tid_y*n+tid_x], temphist[tid_y*n+tid_x+4]);
    }
    __syncthreads();
    if (tid_x < 2)
        temphist[tid_y*n+tid_x] = fmax(temphist[tid_y*n+tid_x], temphist[tid_y*n+tid_x+2]);
    __syncthreads();
    if (tid_x < 1)
        temphist[tid_y*n+tid_x] = fmax(temphist[tid_y*n+tid_x], temphist[tid_y*n+tid_x+1]);
    __syncthreads();
    float omax = temphist[tid_y*n];

    if (f < total_feat) {
        float mag_thr = (float)(omax * ORI_PEAK_RATIO);
        int l, r;
        for (int j = tid_x; j < n; j+=bsz_x) {
            l = (j == 0) ? n - 1 : j - 1;
            r = (j + 1) % n;
            if (hist[tid_y*n+j] > hist[tid_y*n+l] &&
                hist[tid_y*n+j] > hist[tid_y*n+r] &&
                hist[tid_y*n+j] >= mag_thr) {
                int idx = atomicAdd(counter, 1);

                if (idx < max_feat) {
                    float bin = j + 0.5f * (hist[tid_y*n+l] - hist[tid_y*n+r]) /
                                (hist[tid_y*n+l] - 2.0f*hist[tid_y*n+j] + hist[tid_y*n+r]);
                    bin = (bin < 0.0f) ? bin + n : (bin >= n) ? bin - n : bin;
                    float ori = 360.f - ((360.f/n) * bin);

                    float new_real_x = real_x;
                    float new_real_y = real_y;
                    float new_size = size;

                    if (double_input) {
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
template<typename T>
__global__ void computeDescriptor(
    float* desc_out,
    const unsigned desc_len,
    const unsigned histsz,
    const float* x_in,
    const float* y_in,
    const unsigned* layer_in,
    const float* response_in,
    const float* size_in,
    const float* ori_in,
    const unsigned total_feat,
    const CParam<T> gauss_octave,
    const int d,
    const int n,
    const float scale,
    const int n_layers)
{
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int bsz_x = blockDim.x;
    const int bsz_y = blockDim.y;

    const int f = blockIdx.y * bsz_y + tid_y;

    SharedMemory<float> shared;
    float* shrdMem = shared.getPointer();
    float* desc = shrdMem;
    float* accum = shrdMem + desc_len * histsz;

    for (int i = tid_x; i < desc_len*histsz; i += bsz_x)
        desc[tid_y*desc_len+i] = 0.f;
    __syncthreads();

    if (f < total_feat) {
        const unsigned layer = layer_in[f];
        float ori = (360.f - ori_in[f]) * PI_VAL / 180.f;
        ori = (ori > PI_VAL) ? ori - PI_VAL*2 : ori;
        const float size = size_in[f];
        const int fx = round(x_in[f] * scale);
        const int fy = round(y_in[f] * scale);

        const int dim0 = gauss_octave.dims[0];
        const int dim1 = gauss_octave.dims[1];
        const int imel = dim0 * dim1;

        // Points img to correct Gaussian pyramid layer
        const T* img_ptr = gauss_octave.ptr + layer * imel;

        float cos_t = cosf(ori);
        float sin_t = sinf(ori);
        float bins_per_rad = n / (PI_VAL * 2.f);
        float exp_denom = d * d * 0.5f;
        float hist_width = DESCR_SCL_FCTR * size * scale * 0.5f;
        int radius = hist_width * sqrtf(2.f) * (d + 1.f) * 0.5f + 0.5f;

        int len = radius*2+1;
        const int hist_off = (tid_x % histsz) * desc_len;

        // Calculate orientation histogram
        for (int l = tid_x; l < len*len; l += bsz_x) {
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

                float grad_mag = sqrtf(dx*dx + dy*dy);
                float grad_ori = atan2f(dy, dx) - ori;
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
		                            atomicAdd(&desc[hist_off + tid_y*desc_len + (yb*d + xb)*n + ob], v_o);
		                        }
		                    }
	                    }
	                }
                }
            }
        }
    }
    __syncthreads();

    // Combine histograms (reduces previous atomicAdd overhead)
    for (int l = tid_x; l < desc_len*4; l += bsz_x)
        desc[l] += desc[l+4*desc_len];
    __syncthreads();
    for (int l = tid_x; l < desc_len*2; l += bsz_x)
        desc[l    ] += desc[l+2*desc_len];
    __syncthreads();
    for (int l = tid_x; l < desc_len; l += bsz_x)
        desc[l] += desc[l+desc_len];
    __syncthreads();

    normalizeDesc(desc, accum, desc_len);

    for (int i = tid_x; i < desc_len; i += bsz_x)
        desc[tid_y*desc_len+i] = min(desc[tid_y*desc_len+i], DESC_MAG_THR);
    __syncthreads();

    normalizeDesc(desc, accum, desc_len);

    if (f < total_feat) {
        // Calculate final descriptor values
        for (int k = tid_x; k < desc_len; k += bsz_x)
            desc_out[f*desc_len+k] = round(min(255.f, desc[tid_y*desc_len+k] * INT_DESCR_FCTR));
    }
}

// Computes GLOH feature descriptors for features in an array. Based on Section III-B
// of Mikolajczyk and Schmid paper.
template<typename T>
__global__ void computeGLOHDescriptor(
    float* desc_out,
    const unsigned desc_len,
    const unsigned histsz,
    const float* x_in,
    const float* y_in,
    const unsigned* layer_in,
    const float* response_in,
    const float* size_in,
    const float* ori_in,
    const unsigned total_feat,
    const CParam<T> gauss_octave,
    const int d,
    const unsigned rb,
    const unsigned ab,
    const unsigned hb,
    const float scale,
    const int n_layers)
{
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int bsz_x = blockDim.x;
    const int bsz_y = blockDim.y;

    const int f = blockIdx.y * bsz_y + tid_y;

    SharedMemory<float> shared;
    float* shrdMem = shared.getPointer();
    float* desc = shrdMem;
    float* accum = shrdMem + desc_len * histsz;

    for (int i = tid_x; i < desc_len*histsz; i += bsz_x)
        desc[tid_y*desc_len+i] = 0.f;
    __syncthreads();

    if (f < total_feat) {
        const unsigned layer = layer_in[f];
        float ori = (360.f - ori_in[f]) * PI_VAL / 180.f;
        ori = (ori > PI_VAL) ? ori - PI_VAL*2 : ori;
        const float size = size_in[f];
        const int fx = round(x_in[f] * scale);
        const int fy = round(y_in[f] * scale);

        const int dim0 = gauss_octave.dims[0];
        const int dim1 = gauss_octave.dims[1];
        const int imel = dim0 * dim1;

        // Points img to correct Gaussian pyramid layer
        const T* img_ptr = gauss_octave.ptr + layer * imel;

        float cos_t = cosf(ori);
        float sin_t = sinf(ori);
        float hist_bins_per_rad = hb / (PI_VAL * 2.f);
        float polar_bins_per_rad = ab / (PI_VAL * 2.f);
        float exp_denom = GLOHRadii[rb-1] * 0.5f;

        float hist_width = DESCR_SCL_FCTR * size * scale * 0.5f;

        // Keep same descriptor radius used for SIFT
        int radius = hist_width * sqrt(2.f) * (d + 1.f) * 0.5f + 0.5f;

        // Alternative radius size calculation, changing the radius weight
        // (rw) in the range of 0.25f-0.75f gives different results,
        // increasing it tends to show a better recall rate but with a
        // smaller amount of correct matches
        //float rw = 0.5f;
        //int radius = hist_width * GLOHRadii[rb-1] * rw + 0.5f;

        int len = radius*2+1;
        const int hist_off = (tid_x % histsz) * desc_len;

        // Calculate orientation histogram
        for (int l = tid_x; l < len*len; l += bsz_x) {
            int i = l / len - radius;
            int j = l % len - radius;

            int y = fy + i;
            int x = fx + j;

            float x_rot = (j * cos_t - i * sin_t);
            float y_rot = (j * sin_t + i * cos_t);

            float r = sqrt(x_rot*x_rot + y_rot*y_rot) / radius * GLOHRadii[rb-1];
            float theta = atan2(y_rot, x_rot);
            while (theta < 0.0f)
                theta += PI_VAL*2;
            while (theta >= PI_VAL*2)
                theta -= PI_VAL*2;

            float tbin = theta * polar_bins_per_rad;
            float rbin = (r < GLOHRadii[0]) ? r / GLOHRadii[0] :
                         ((r < GLOHRadii[1]) ? 1 + (r - GLOHRadii[0]) / (float)(GLOHRadii[1] - GLOHRadii[0]) :
                         min(2 + (r - GLOHRadii[1]) / (float)(GLOHRadii[2] - GLOHRadii[1]), 3.f-FLT_EPSILON));

            if (r <= GLOHRadii[rb-1] &&
                y > 0 && y < dim0 - 1 && x > 0 && x < dim1 - 1) {
                float dx = (float)(IPTR(x+1, y) - IPTR(x-1, y));
                float dy = (float)(IPTR(x, y-1) - IPTR(x, y+1));

                float grad_mag = sqrtf(dx*dx + dy*dy);
                float grad_ori = atan2f(dy, dx) - ori;
                while (grad_ori < 0.0f)
                    grad_ori += PI_VAL*2;
                while (grad_ori >= PI_VAL*2)
                    grad_ori -= PI_VAL*2;

                float w = exp(-r / exp_denom);
                float obin = grad_ori * hist_bins_per_rad;
                float mag = grad_mag*w;

                int t0 = floor(tbin);
                int r0 = floor(rbin);
                int o0 = floor(obin);
                tbin -= t0;
                rbin -= r0;
                obin -= o0;

                for (int rl = 0; rl <= 1; rl++) {
                    int rb = (rbin > 0.5f) ? (r0 + rl) : (r0 - rl);
                    float v_r = mag * ((rl == 0) ? 1.0f - rbin : rbin);
                    if (rb >= 0 && rb <= 2) {
                        for (int tl = 0; tl <= 1; tl++) {
                            int tb = (t0 + tl) % ab;
                            float v_t = v_r * ((tl == 0) ? 1.0f - tbin : tbin);
                            for (int ol = 0; ol <= 1; ol++) {
                                int ob = (o0 + ol) % hb;
                                float v_o = v_t * ((ol == 0) ? 1.0f - obin : obin);
                                unsigned idx = (rb > 0) * (hb + ((rb-1) * ab + tb)*hb) + ob;
                                atomicAdd(&desc[hist_off + tid_y*desc_len + idx], v_o);
                            }
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    // Combine histograms (reduces previous atomicAdd overhead)
    for (int l = tid_x; l < desc_len*4; l += bsz_x)
        desc[l] += desc[l+4*desc_len];
    __syncthreads();
    for (int l = tid_x; l < desc_len*2; l += bsz_x)
        desc[l    ] += desc[l+2*desc_len];
    __syncthreads();
    for (int l = tid_x; l < desc_len; l += bsz_x)
        desc[l] += desc[l+desc_len];
    __syncthreads();

    normalizeGLOHDesc(desc, accum, desc_len);

    for (int i = tid_x; i < desc_len; i += bsz_x)
        desc[tid_y*desc_len+i] = min(desc[tid_y*desc_len+i], DESC_MAG_THR);
    __syncthreads();

    normalizeGLOHDesc(desc, accum, desc_len);

    if (f < total_feat) {
        // Calculate final descriptor values
        for (int k = tid_x; k < desc_len; k += bsz_x)
            desc_out[f*desc_len+k] = round(min(255.f, desc[tid_y*desc_len+k] * INT_DESCR_FCTR));
    }
}

#undef IPTR

template<typename T, typename convAccT>
Param<T> createInitialImage(
    CParam<T> img,
    const float init_sigma,
    const bool double_input)
{
    Param<T> init_img, init_tmp;
    init_img.dims[0] = init_tmp.dims[0] = (double_input) ? img.dims[0] * 2 : img.dims[0];
    init_img.dims[1] = init_tmp.dims[1] = (double_input) ? img.dims[1] * 2 : img.dims[1];
    init_img.strides[0] = init_tmp.strides[0] = 1;
    init_img.strides[1] = init_tmp.strides[1] = init_img.dims[0];

    for (int k = 2; k < 4; k++) {
        init_img.dims[k] = 1;
        init_img.strides[k] = init_img.dims[k-1] * init_img.strides[k-1];
        init_tmp.dims[k] = 1;
        init_tmp.strides[k] = init_tmp.dims[k-1] * init_tmp.strides[k-1];
    }

    dim_t init_img_el = init_img.strides[3] * init_img.dims[3];
    init_img.ptr = memAlloc<T>(init_img_el);
    init_tmp.ptr = memAlloc<T>(init_img_el);

    float s = (double_input) ? std::max((float)sqrt(init_sigma * init_sigma - INIT_SIGMA * INIT_SIGMA * 4), 0.1f)
                             : std::max((float)sqrt(init_sigma * init_sigma - INIT_SIGMA * INIT_SIGMA), 0.1f);

    Param<convAccT> filter = gauss_filter<convAccT>(s);

    if (double_input) {
        resize<T, AF_INTERP_BILINEAR>(init_img, img);
        convolve2<T, convAccT, 0, false>(init_tmp, init_img, filter);
    }
    else
        convolve2<T, convAccT, 0, false>(init_tmp, img, filter);

    convolve2<T, convAccT, 1, false>(init_img, CParam<T>(init_tmp), filter);

    memFree(init_tmp.ptr);
    memFree(filter.ptr);

    return init_img;
}

template<typename T, typename convAccT>
std::vector< Param<T> > buildGaussPyr(
    Param<T> init_img,
    const unsigned n_octaves,
    const unsigned n_layers,
    const float init_sigma)
{
    // Precompute Gaussian sigmas using the following formula:
    // \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    std::vector<float> sig_layers(n_layers + 3);
    sig_layers[0] = init_sigma;
    float k = std::pow(2.0f, 1.0f / n_layers);
    for (unsigned i = 1; i < n_layers + 3; i++) {
        float sig_prev = std::pow(k, i-1) * init_sigma;
        float sig_total = sig_prev * k;
        sig_layers[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    // Gaussian Pyramid
    std::vector<Param<T> > gauss_pyr(n_octaves);
    std::vector<Param<T> > tmp_pyr(n_octaves * (n_layers+3));
    for (unsigned o = 0; o < n_octaves; o++) {
        gauss_pyr[o].dims[0] = (o == 0) ? init_img.dims[0] : gauss_pyr[o-1].dims[0] / 2;
        gauss_pyr[o].dims[1] = (o == 0) ? init_img.dims[1] : gauss_pyr[o-1].dims[1] / 2;
        gauss_pyr[o].dims[2] = n_layers+3;
        gauss_pyr[o].dims[3] = 1;

        gauss_pyr[o].strides[0] = 1;
        gauss_pyr[o].strides[1] = gauss_pyr[o].dims[0] * gauss_pyr[o].strides[0];
        gauss_pyr[o].strides[2] = gauss_pyr[o].dims[1] * gauss_pyr[o].strides[1];
        gauss_pyr[o].strides[3] = gauss_pyr[o].dims[2] * gauss_pyr[o].strides[2];

        const unsigned nel = gauss_pyr[o].dims[3] * gauss_pyr[o].strides[3];
        gauss_pyr[o].ptr = memAlloc<T>(nel);

        for (unsigned l = 0; l < n_layers+3; l++) {
            unsigned src_idx = (l == 0) ? (o-1)*(n_layers+3) + n_layers : o*(n_layers+3) + l-1;
            unsigned idx = o*(n_layers+3) + l;

            if (o == 0 && l == 0) {
                for (int k = 0; k < 4; k++) {
                    tmp_pyr[idx].dims[k] = init_img.dims[k];
                    tmp_pyr[idx].strides[k] = init_img.strides[k];
                }
                tmp_pyr[idx].ptr = init_img.ptr;
            }
            else if (l == 0) {
                tmp_pyr[idx].dims[0] = tmp_pyr[src_idx].dims[0] / 2;
                tmp_pyr[idx].dims[1] = tmp_pyr[src_idx].dims[1] / 2;
                tmp_pyr[idx].strides[0] = 1;
                tmp_pyr[idx].strides[1] = tmp_pyr[idx].dims[0];

                for (int k = 2; k < 4; k++) {
                    tmp_pyr[idx].dims[k] = 1;
                    tmp_pyr[idx].strides[k] = tmp_pyr[idx].dims[k-1] * tmp_pyr[idx].strides[k-1];
                }

                dim_t lvl_el = tmp_pyr[idx].strides[3] * tmp_pyr[idx].dims[3];
                tmp_pyr[idx].ptr = memAlloc<T>(lvl_el);

                resize<T, AF_INTERP_BILINEAR>(tmp_pyr[idx], tmp_pyr[src_idx]);
            }
            else {
                for (int k = 0; k < 4; k++) {
                    tmp_pyr[idx].dims[k] = tmp_pyr[src_idx].dims[k];
                    tmp_pyr[idx].strides[k] = tmp_pyr[src_idx].strides[k];
                }
                dim_t lvl_el = tmp_pyr[idx].strides[3] * tmp_pyr[idx].dims[3];
                tmp_pyr[idx].ptr = memAlloc<T>(lvl_el);

                Param<T> tmp;
                for (int k = 0; k < 4; k++) {
                    tmp.dims[k] = tmp_pyr[idx].dims[k];
                    tmp.strides[k] = tmp_pyr[idx].strides[k];
                }
                tmp.ptr = memAlloc<T>(lvl_el);

                Param<convAccT> filter = gauss_filter<convAccT>(sig_layers[l]);



                convolve2<T, convAccT, 0, false>(tmp, tmp_pyr[src_idx], filter);
                convolve2<T, convAccT, 1, false>(tmp_pyr[idx], CParam<T>(tmp), filter);

                memFree(tmp.ptr);
                memFree(filter.ptr);
            }

            const unsigned imel = tmp_pyr[idx].dims[3] * tmp_pyr[idx].strides[3];
            const unsigned offset = imel * l;

            CUDA_CHECK(cudaMemcpyAsync(gauss_pyr[o].ptr + offset, tmp_pyr[idx].ptr,
                        imel * sizeof(T), cudaMemcpyDeviceToDevice,
                        cuda::getActiveStream()));
        }
    }

    for (unsigned o = 0; o < n_octaves; o++) {
        for (unsigned l = 0; l < n_layers+3; l++) {
            unsigned idx = o*(n_layers+3) + l;
            memFree(tmp_pyr[idx].ptr);
        }
    }

    return gauss_pyr;
}

template<typename T>
std::vector< Param<T> > buildDoGPyr(
    std::vector< Param<T> >& gauss_pyr,
    const unsigned n_octaves,
    const unsigned n_layers)
{
    // DoG Pyramid
    std::vector< Param<T> > dog_pyr(n_octaves);
    for (unsigned o = 0; o < n_octaves; o++) {
        for (int k = 0; k < 4; k++) {
            dog_pyr[o].dims[k] = (k == 2) ? gauss_pyr[o].dims[k]-1 : gauss_pyr[o].dims[k];
            dog_pyr[o].strides[k] = (k == 0) ? 1 : dog_pyr[o].dims[k-1] * dog_pyr[o].strides[k-1];
        }

        dog_pyr[o].ptr = memAlloc<T>(dog_pyr[o].dims[3] * dog_pyr[o].strides[3]);

        const unsigned nel = dog_pyr[o].dims[1] * dog_pyr[o].strides[1];
        const unsigned dog_layers = n_layers+2;

        dim3 threads(SIFT_THREADS);
        dim3 blocks(divup(nel, threads.x));
        CUDA_LAUNCH((sub<T>), blocks, threads,
                    dog_pyr[o], gauss_pyr[o], nel, dog_layers);
        POST_LAUNCH_CHECK();
    }

    return dog_pyr;
}

template <typename T>
void update_permutation(thrust::device_ptr<T>& keys, cuda::ThrustVector<int>& permutation)
{
    // temporary storage for keys
    cuda::ThrustVector<T> temp(permutation.size());

    // permute the keys with the current reordering
    THRUST_SELECT((thrust::gather), permutation.begin(), permutation.end(), keys, temp.begin());

    // stable_sort the permuted keys and update the permutation
    THRUST_SELECT((thrust::stable_sort_by_key), temp.begin(), temp.end(), permutation.begin());
}

template <typename T>
void apply_permutation(thrust::device_ptr<T>& keys, cuda::ThrustVector<int>& permutation)
{
    // copy keys to temporary vector
    cuda::ThrustVector<T> temp(keys, keys+permutation.size());

    // permute the keys
    THRUST_SELECT((thrust::gather), permutation.begin(), permutation.end(), temp.begin(), keys);
}

template<typename T, typename convAccT>
void sift(unsigned* out_feat,
          unsigned* out_dlen,
          float** d_x,
          float** d_y,
          float** d_score,
          float** d_ori,
          float** d_size,
          float** d_desc,
          CParam<T> img,
          const unsigned n_layers,
          const float contrast_thr,
          const float edge_thr,
          const float init_sigma,
          const bool double_input,
          const float img_scale,
          const float feature_ratio,
          const bool compute_GLOH)
{
    unsigned min_dim = min(img.dims[0], img.dims[1]);
    if (double_input) min_dim *= 2;

    const unsigned n_octaves = floor(log(min_dim) / log(2)) - 2;

    Param<T> init_img = createInitialImage<T, convAccT>(img, init_sigma, double_input);

    std::vector< Param<T> > gauss_pyr = buildGaussPyr<T, convAccT>(init_img, n_octaves, n_layers, init_sigma);

    std::vector< Param<T> > dog_pyr = buildDoGPyr<T>(gauss_pyr, n_octaves, n_layers);

    std::vector<float*> d_x_pyr(n_octaves, NULL);
    std::vector<float*> d_y_pyr(n_octaves, NULL);
    std::vector<float*> d_response_pyr(n_octaves, NULL);
    std::vector<float*> d_size_pyr(n_octaves, NULL);
    std::vector<float*> d_ori_pyr(n_octaves, NULL);
    std::vector<float*> d_desc_pyr(n_octaves, NULL);
    std::vector<unsigned> feat_pyr(n_octaves, 0);
    unsigned total_feat = 0;

    const unsigned d = DESCR_WIDTH;
    const unsigned n = DESCR_HIST_BINS;
    const unsigned rb = GLOHRadialBins;
    const unsigned ab = GLOHAngularBins;
    const unsigned hb = GLOHHistBins;
    const unsigned desc_len = (compute_GLOH) ? (1 + (rb-1) * ab) * hb : d*d*n;

    unsigned* d_count = memAlloc<unsigned>(1);
    for (unsigned i = 0; i < n_octaves; i++) {
        if (dog_pyr[i].dims[0]-2*IMG_BORDER < 1 ||
            dog_pyr[i].dims[1]-2*IMG_BORDER < 1)
            continue;

        const unsigned imel = dog_pyr[i].dims[0] * dog_pyr[i].dims[1];
        const unsigned max_feat = ceil(imel * feature_ratio);

        CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(unsigned),
                                   cuda::getActiveStream()));

        float* d_extrema_x = memAlloc<float>(max_feat);
        float* d_extrema_y = memAlloc<float>(max_feat);
        unsigned* d_extrema_layer = memAlloc<unsigned>(max_feat);

        int dim0 = dog_pyr[i].dims[0];
        int dim1 = dog_pyr[i].dims[1];

        dim3 threads(SIFT_THREADS_X, SIFT_THREADS_Y);
        dim3 blocks(divup(dim0-2*IMG_BORDER, threads.x), divup(dim1-2*IMG_BORDER, threads.y));

        float extrema_thr = 0.5f * contrast_thr / n_layers;
        const size_t extrema_shared_size = (threads.x+2) * (threads.y+2) * 3 * sizeof(float);
        CUDA_LAUNCH_SMEM((detectExtrema<T>), blocks, threads, extrema_shared_size,
                         d_extrema_x, d_extrema_y, d_extrema_layer, d_count,
                         CParam<T>(dog_pyr[i]), max_feat, extrema_thr);
        POST_LAUNCH_CHECK();

        unsigned extrema_feat = 0;
        CUDA_CHECK(cudaMemcpyAsync(&extrema_feat, d_count, sizeof(unsigned), cudaMemcpyDeviceToHost,
                    cuda::getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
        extrema_feat = min(extrema_feat, max_feat);

        if (extrema_feat == 0) {
            memFree(d_extrema_x);
            memFree(d_extrema_y);
            memFree(d_extrema_layer);

            continue;
        }

        CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(unsigned),
                                   cuda::getActiveStream()));

        unsigned interp_feat = 0;

        float* d_interp_x = memAlloc<float>(extrema_feat);
        float* d_interp_y = memAlloc<float>(extrema_feat);
        unsigned* d_interp_layer = memAlloc<unsigned>(extrema_feat);
        float* d_interp_response = memAlloc<float>(extrema_feat);
        float* d_interp_size = memAlloc<float>(extrema_feat);

        threads = dim3(SIFT_THREADS, 1);
        blocks = dim3(divup(extrema_feat, threads.x), 1);

        CUDA_LAUNCH((interpolateExtrema<T>), blocks, threads,
                    d_interp_x, d_interp_y, d_interp_layer,
                    d_interp_response, d_interp_size, d_count,
                    d_extrema_x, d_extrema_y, d_extrema_layer, extrema_feat,
                    dog_pyr[i], max_feat, i, n_layers,
                    contrast_thr, edge_thr, init_sigma, img_scale);
        POST_LAUNCH_CHECK();

        memFree(d_extrema_x);
        memFree(d_extrema_y);
        memFree(d_extrema_layer);

        CUDA_CHECK(cudaMemcpyAsync(&interp_feat, d_count, sizeof(unsigned), cudaMemcpyDeviceToHost,
                    cuda::getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
        interp_feat = min(interp_feat, max_feat);

        CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(unsigned),
                                   cuda::getActiveStream()));

        if (interp_feat == 0) {
            memFree(d_interp_x);
            memFree(d_interp_y);
            memFree(d_interp_layer);
            memFree(d_interp_response);
            memFree(d_interp_size);

            continue;
        }

        thrust::device_ptr<float> interp_x_ptr = thrust::device_pointer_cast(d_interp_x);
        thrust::device_ptr<float> interp_y_ptr = thrust::device_pointer_cast(d_interp_y);
        thrust::device_ptr<unsigned> interp_layer_ptr = thrust::device_pointer_cast(d_interp_layer);
        thrust::device_ptr<float> interp_response_ptr = thrust::device_pointer_cast(d_interp_response);
        thrust::device_ptr<float> interp_size_ptr = thrust::device_pointer_cast(d_interp_size);

        cuda::ThrustVector<int> permutation(interp_feat);
        thrust::sequence(permutation.begin(), permutation.end());

        update_permutation<float>(interp_size_ptr, permutation);
        update_permutation<float>(interp_response_ptr, permutation);
        update_permutation<unsigned>(interp_layer_ptr, permutation);
        update_permutation<float>(interp_y_ptr, permutation);
        update_permutation<float>(interp_x_ptr, permutation);

        apply_permutation<float>(interp_size_ptr, permutation);
        apply_permutation<float>(interp_response_ptr, permutation);
        apply_permutation<unsigned>(interp_layer_ptr, permutation);
        apply_permutation<float>(interp_y_ptr, permutation);
        apply_permutation<float>(interp_x_ptr, permutation);

        float* d_nodup_x = memAlloc<float>(interp_feat);
        float* d_nodup_y = memAlloc<float>(interp_feat);
        unsigned* d_nodup_layer = memAlloc<unsigned>(interp_feat);
        float* d_nodup_response = memAlloc<float>(interp_feat);
        float* d_nodup_size = memAlloc<float>(interp_feat);

        threads = dim3(SIFT_THREADS, 1);
        blocks = dim3(divup(interp_feat, threads.x), 1);

        CUDA_LAUNCH((removeDuplicates), blocks, threads,
                    d_nodup_x, d_nodup_y, d_nodup_layer,
                    d_nodup_response, d_nodup_size, d_count,
                    d_interp_x, d_interp_y, d_interp_layer,
                    d_interp_response, d_interp_size, interp_feat);
        POST_LAUNCH_CHECK();

        memFree(d_interp_x);
        memFree(d_interp_y);
        memFree(d_interp_layer);
        memFree(d_interp_response);
        memFree(d_interp_size);

        unsigned nodup_feat = 0;
        CUDA_CHECK(cudaMemcpyAsync(&nodup_feat, d_count, sizeof(unsigned), cudaMemcpyDeviceToHost,
                    cuda::getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
        CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(unsigned),
                                   cuda::getActiveStream()));

        const unsigned max_oriented_feat = nodup_feat * 3;

        float* d_oriented_x = memAlloc<float>(max_oriented_feat);
        float* d_oriented_y = memAlloc<float>(max_oriented_feat);
        unsigned* d_oriented_layer = memAlloc<unsigned>(max_oriented_feat);
        float* d_oriented_response = memAlloc<float>(max_oriented_feat);
        float* d_oriented_size = memAlloc<float>(max_oriented_feat);
        float* d_oriented_ori = memAlloc<float>(max_oriented_feat);

        threads = dim3(SIFT_THREADS_X, SIFT_THREADS_Y);
        blocks = dim3(1, divup(nodup_feat, threads.y));

        const size_t ori_shared_size = ORI_HIST_BINS * threads.y * 2 * sizeof(float);
        CUDA_LAUNCH_SMEM((calcOrientation<T>), blocks, threads, ori_shared_size,
                         d_oriented_x, d_oriented_y, d_oriented_layer,
                         d_oriented_response, d_oriented_size, d_oriented_ori, d_count,
                         d_nodup_x, d_nodup_y, d_nodup_layer,
                         d_nodup_response, d_nodup_size, nodup_feat,
                         gauss_pyr[i], max_oriented_feat, i, double_input);
        POST_LAUNCH_CHECK();

        memFree(d_nodup_x);
        memFree(d_nodup_y);
        memFree(d_nodup_layer);
        memFree(d_nodup_response);
        memFree(d_nodup_size);

        unsigned oriented_feat = 0;
        CUDA_CHECK(cudaMemcpyAsync(&oriented_feat, d_count, sizeof(unsigned), cudaMemcpyDeviceToHost,
                    cuda::getActiveStream()));
        CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
        oriented_feat = min(oriented_feat, max_oriented_feat);

        if (oriented_feat == 0) {
            memFree(d_oriented_x);
            memFree(d_oriented_y);
            memFree(d_oriented_layer);
            memFree(d_oriented_response);
            memFree(d_oriented_size);
            memFree(d_oriented_ori);

            continue;
        }

        float* d_desc = memAlloc<float>(oriented_feat * desc_len);

        float scale = 1.f/(1 << i);
        if (double_input) scale *= 2.f;

        threads = dim3(SIFT_THREADS, 1);
        blocks  = dim3(1, divup(oriented_feat, threads.y));

        const unsigned histsz = 8;
        const size_t shared_size = desc_len * (histsz+1) * sizeof(float);

        if (compute_GLOH)
            CUDA_LAUNCH_SMEM((computeGLOHDescriptor<T>), blocks, threads, shared_size,
                             d_desc, desc_len, histsz,
                             d_oriented_x, d_oriented_y, d_oriented_layer,
                             d_oriented_response, d_oriented_size, d_oriented_ori,
                             oriented_feat, gauss_pyr[i], d, rb, ab, hb,
                             scale, n_layers);
        else
            CUDA_LAUNCH_SMEM((computeDescriptor<T>), blocks, threads, shared_size,
                             d_desc, desc_len, histsz,
                             d_oriented_x, d_oriented_y, d_oriented_layer,
                             d_oriented_response, d_oriented_size, d_oriented_ori,
                             oriented_feat, gauss_pyr[i], d, n, scale, n_layers);
        POST_LAUNCH_CHECK();

        total_feat += oriented_feat;
        feat_pyr[i] = oriented_feat;

        if (oriented_feat > 0) {
            d_x_pyr[i] = d_oriented_x;
            d_y_pyr[i] = d_oriented_y;
            d_response_pyr[i] = d_oriented_response;
            d_ori_pyr[i] = d_oriented_ori;
            d_size_pyr[i] = d_oriented_size;
            d_desc_pyr[i] = d_desc;
        }
    }

    memFree(d_count);

    for (size_t i = 0; i < gauss_pyr.size(); i++)
        memFree(gauss_pyr[i].ptr);
    for (size_t i = 0; i < dog_pyr.size(); i++)
        memFree(dog_pyr[i].ptr);

    // Allocate output memory
    *d_x     = memAlloc<float>(total_feat);
    *d_y     = memAlloc<float>(total_feat);
    *d_score = memAlloc<float>(total_feat);
    *d_ori   = memAlloc<float>(total_feat);
    *d_size  = memAlloc<float>(total_feat);
    *d_desc  = memAlloc<float>(total_feat * desc_len);

    unsigned offset = 0;
    for (unsigned i = 0; i < n_octaves; i++) {
        if (feat_pyr[i] == 0)
            continue;

        CUDA_CHECK(cudaMemcpyAsync(*d_x+offset, d_x_pyr[i], feat_pyr[i] * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda::getActiveStream()));
        CUDA_CHECK(cudaMemcpyAsync(*d_y+offset, d_y_pyr[i], feat_pyr[i] * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda::getActiveStream()));
        CUDA_CHECK(cudaMemcpyAsync(*d_score+offset, d_response_pyr[i], feat_pyr[i] * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda::getActiveStream()));
        CUDA_CHECK(cudaMemcpyAsync(*d_ori+offset, d_ori_pyr[i], feat_pyr[i] * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda::getActiveStream()));
        CUDA_CHECK(cudaMemcpyAsync(*d_size+offset, d_size_pyr[i], feat_pyr[i] * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda::getActiveStream()));

        CUDA_CHECK(cudaMemcpyAsync(*d_desc+(offset*desc_len), d_desc_pyr[i],
                    feat_pyr[i] * desc_len * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda::getActiveStream()));

        memFree(d_x_pyr[i]);
        memFree(d_y_pyr[i]);
        memFree(d_response_pyr[i]);
        memFree(d_ori_pyr[i]);
        memFree(d_size_pyr[i]);
        memFree(d_desc_pyr[i]);

        offset += feat_pyr[i];
    }

    // Sets number of output features
    *out_feat = total_feat;
    *out_dlen = desc_len;
}

} // namespace kernel

} // namespace cuda

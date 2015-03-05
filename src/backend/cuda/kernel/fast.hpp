/*******************************************************
 * Copyright (c) 2014, ArrayFire
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
#include <kernel/fast_lut.hpp>
#include <memory.hpp>

namespace cuda
{

namespace kernel
{

inline __device__
int clamp(const int f, const int a, const int b)
{
    return max(a, min(f, b));
}

inline __device__
int idx_y(const int i)
{
    int j = i - 4;
    int k = min(j, 8 - j);
    return clamp(k, -3, 3);
}

inline __device__
int idx_x(const int i)
{
    return idx_y((i + 4) & 15);
}

inline __device__
int idx(const int x, const int y)
{
    return ((threadIdx.x + 3 + x) + (blockDim.x + 6) * (threadIdx.y + 3 + y));
}

// test_greater()
// Tests if a pixel x >= p + thr
template<typename T>
inline __device__
int test_greater(const T x, const T p, const float thr)
{
    return (x >= p + thr);
}

// test_smaller()
// Tests if a pixel x <= p - thr
template<typename T>
inline __device__
int test_smaller(const T x, const T p, const float thr)
{
    return (x <= p - thr);
}

// test_pixel()
// Returns -1 when x < p - thr
// Returns  0 when x >= p - thr && x <= p + thr
// Returns  1 when x > p + thr
template<typename T>
inline __device__
int test_pixel(const T* local_image, const T p, const float thr, const int x, const int y)
{
    return -test_smaller<T>(local_image[idx(x,y)], p, thr) | test_greater<T>(local_image[idx(x,y)], p, thr);
}

// max_val()
// Returns max of x and y
inline __device__
int max_val(const int x, const int y)
{
    return max(x, y);
}
inline __device__
unsigned max_val(const unsigned x, const unsigned y)
{
    return max(x, y);
}
inline __device__
float max_val(const float x, const float y)
{
    return fmax(x, y);
}
inline __device__
double max_val(const double x, const double y)
{
    return fmax(x, y);
}

// abs_diff()
// Returns absolute difference of x and y
inline __device__ int abs_diff(const int x, const int y)
{
    int i = x - y;
    return max(-i, i);
}
inline __device__ unsigned abs_diff(const unsigned x, const unsigned y)
{
    int i = (int)x - (int)y;
    return max(-i, i);
}
inline __device__ float abs_diff(const float x, const float y)
{
    return fabs(x - y);
}
inline __device__ double abs_diff(const double x, const double y)
{
    return fabs(x - y);
}

// non-specialized class template
//http://www.naic.edu/~phil/hardware/nvidia/doc/src/simpleTemplates/doc/readme.txt
template <class T>
class ExtSharedMem
{
    public:
        // Ensure that we won't compile any un-specialized types
        __device__ T* getPointer() { extern __shared__ float s_float[]; return s_float; };
};

// specialization for char
template <>
class ExtSharedMem <char>
{
    public:
        __device__ char* getPointer() { extern __shared__ char s_char[]; return s_char; }
};

// specialization for int
template <>
class ExtSharedMem <uchar>
{
    public:
        __device__ uchar* getPointer() { extern __shared__ uchar s_uchar[]; return s_uchar; }
};

// specialization for int
template <>
class ExtSharedMem <int>
{
    public:
        __device__ int* getPointer() { extern __shared__ int s_int[]; return s_int; }
};

// specialization for unsigned
template <>
class ExtSharedMem <unsigned>
{
    public:
        __device__ unsigned* getPointer() { extern __shared__ unsigned s_unsigned[]; return s_unsigned; }
};

// specialization for float
template <>
class ExtSharedMem <float>
{
    public:
        __device__ float* getPointer() { extern __shared__ float s_float[]; return s_float; }
};

// specialization for double
template <>
class ExtSharedMem <double>
{
    public:
        __device__ double* getPointer() { extern __shared__ double s_double[]; return s_double; }
};

template<typename T, int arc_length>
__device__
void locate_features_core(
    T* local_image,
    T* score,
    const unsigned idim0,
    const unsigned idim1,
    const float thr,
    int x, int y,
    const unsigned edge)
{
    if (x >= idim0 - edge || y >= idim1 - edge) return;

    T p = local_image[idx( 0, 0)];

    // Start by testing opposite pixels of the circle that will result in
    // a non-kepoint
    int d = test_pixel<T>(local_image, p, thr, -3,  0) | test_pixel<T>(local_image, p, thr, 3,  0);
    if (d == 0)
        return;

    d &= test_pixel<T>(local_image, p, thr, -2,  2) | test_pixel<T>(local_image, p, thr,  2, -2);
    d &= test_pixel<T>(local_image, p, thr,  0,  3) | test_pixel<T>(local_image, p, thr,  0, -3);
    d &= test_pixel<T>(local_image, p, thr,  2,  2) | test_pixel<T>(local_image, p, thr, -2, -2);
    if (d == 0)
        return;

    d &= test_pixel<T>(local_image, p, thr, -3,  1) | test_pixel<T>(local_image, p, thr,  3, -1);
    d &= test_pixel<T>(local_image, p, thr, -1,  3) | test_pixel<T>(local_image, p, thr,  1, -3);
    d &= test_pixel<T>(local_image, p, thr,  1,  3) | test_pixel<T>(local_image, p, thr, -1, -3);
    d &= test_pixel<T>(local_image, p, thr,  3,  1) | test_pixel<T>(local_image, p, thr, -3, -1);
    if (d == 0)
        return;

    int bright = 0, dark = 0;
    T s_bright = 0, s_dark = 0;

    // Force less loop unrolls to control maximum number of registers and
    // launch more blocks
    #pragma unroll 4
    for (int i = 0; i < 16; i++) {
        // Get pixel from the circle
        T p_x = local_image[idx(idx_x(i),idx_y(i))];

        // Compute binary vectors with responses for each pixel on circle
        bright |= test_greater<T>(p_x, p, thr) << i;
        dark   |= test_smaller<T>(p_x, p, thr) << i;

        // Compute scores for brighter and darker pixels
        T weight = abs_diff(p_x, p) - thr;
        s_bright += test_greater<T>(p_x, p, thr) * weight;
        s_dark   += test_smaller<T>(p_x, p, thr) * weight;
    }

    // Checks LUT to verify if there is a segment for which all pixels are much
    // brighter or much darker than central pixel p.
    if ((int)FAST_LUT[bright] >= arc_length || (int)FAST_LUT[dark] >= arc_length)
        score[x + idim0 * y] = max_val(s_bright, s_dark);
}

template<typename T>
__device__
void load_shared_image(CParam<T> in,
                       T *local_image,
                       unsigned ix, unsigned iy,
                       unsigned bx, unsigned by,
                       unsigned x, unsigned y,
                       unsigned lx, unsigned ly)
{
    // Copy an image patch to shared memory, with a 3-pixel edge
    if (ix < lx && iy < ly && x - 3 < in.dims[0] && y - 3 < in.dims[1]) {
        local_image[(ix)      + (bx+6) * (iy)]    = in.ptr[(x-3)    + in.dims[0] * (y-3)];
        if (x + lx - 3 < in.dims[0])
            local_image[(ix + lx) + (bx+6) * (iy)]    = in.ptr[(x+lx-3) + in.dims[0] * (y-3)];
        if (y + ly - 3 < in.dims[1])
            local_image[(ix)      + (bx+6) * (iy+ly)] = in.ptr[(x-3)    + in.dims[0] * (y+ly-3)];
        if (x + lx - 3 < in.dims[0] && y + ly - 3 < in.dims[1])
            local_image[(ix + lx) + (bx+6) * (iy+ly)] = in.ptr[(x+lx-3) + in.dims[0] * (y+ly-3)];
    }
}

template<typename T, int arc_length>
__global__
void locate_features(
    CParam<T> in,
    T* score,
    const float thr,
    const unsigned edge)
{
    unsigned ix = threadIdx.x;
    unsigned iy = threadIdx.y;
    unsigned bx = blockDim.x;
    unsigned by = blockDim.y;
    unsigned x = bx * blockIdx.x + ix + edge;
    unsigned y = by * blockIdx.y + iy + edge;
    unsigned lx = bx / 2 + 3;
    unsigned ly = by / 2 + 3;

    ExtSharedMem<T> shared;
    T* local_image_curr = shared.getPointer();
    load_shared_image(in, local_image_curr, ix, iy, bx, by, x, y, lx, ly);
    __syncthreads();
    locate_features_core<T, arc_length>(local_image_curr, score,
                                        in.dims[0], in.dims[1], thr, x, y, edge);
}

template<class T, bool nonmax>
__global__
void non_max_counts(
    unsigned *d_counts,
    unsigned *d_offsets,
    unsigned *d_total,
    T *flags,
    const T* score,
    const unsigned idim0,
    const unsigned idim1)
{
    const int xid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    const int yid = blockIdx.y * blockDim.y * 8 + threadIdx.y;
    const int tid = blockDim.x * threadIdx.y + threadIdx.x;

    const int xoff = blockDim.x;
    const int yoff = blockDim.y;

    const int xend = (blockIdx.x + 1) * blockDim.x * 2;
    const int yend = (blockIdx.y + 1) * blockDim.y * 8;

    const int bid = blockIdx.y * gridDim.x + blockIdx.x;
    __shared__ unsigned s_counts[256];

    unsigned count = 0;
    for (int y = yid; y < yend; y += yoff) {
        if (y >= idim1 - 1 || y <= 1) continue;
        for (int x = xid; x < xend; x += xoff) {
            if (x >= idim0 - 1 || x <= 1) continue;

            T v = score[y * idim0 + x];
            if (v == 0) {
                if (nonmax) flags[y * idim0 + x] = 0;
                continue;
            }

            if (nonmax) {
                T max_v = v;
                max_v = max_val(score[x-1 + idim0 * (y-1)], score[x-1 + idim0 * y]);
                max_v = max_val(max_v, score[x-1 + idim0 * (y+1)]);
                max_v = max_val(max_v, score[x   + idim0 * (y-1)]);
                max_v = max_val(max_v, score[x   + idim0 * (y+1)]);
                max_v = max_val(max_v, score[x+1 + idim0 * (y-1)]);
                max_v = max_val(max_v, score[x+1 + idim0 * (y)  ]);
                max_v = max_val(max_v, score[x+1 + idim0 * (y+1)]);

                v = (v > max_v) ? v : 0;
                flags[y * idim0 + x] = v;
                if (v == 0) continue;
            }

            count++;
        }
    }

    s_counts[tid] = count;
    __syncthreads();

    if (tid >= 128) return;
    if (tid < 128) s_counts[tid] += s_counts[tid + 128]; __syncthreads();

    if (tid >= 64) return;
    if (tid <  64) s_counts[tid] += s_counts[tid +  64]; __syncthreads();

    if (tid >= 32) return;
    if (tid <  32) s_counts[tid] += s_counts[tid +  32];
    if (tid <  16) s_counts[tid] += s_counts[tid +  16];
    if (tid <   8) s_counts[tid] += s_counts[tid +   8];
    if (tid <   4) s_counts[tid] += s_counts[tid +   4];
    if (tid <   2) s_counts[tid] += s_counts[tid +   2];
    if (tid <   1) s_counts[tid] += s_counts[tid +   1];

    if (tid == 0) {
        unsigned total = s_counts[0] ? atomicAdd(d_total, s_counts[0]) : 0;
        d_counts [bid] = s_counts[0];
        d_offsets[bid] = total;
    }
}

template<class T>
__global__
void get_features(
    float *x_out,
    float *y_out,
    float *score_out,
    const T* flags,
    const unsigned *d_counts,
    const unsigned *d_offsets,
    const unsigned total,
    const unsigned idim0,
    const unsigned idim1)
{
    const int xid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    const int yid = blockIdx.y * blockDim.y * 8 + threadIdx.y;
    const int tid = blockDim.x * threadIdx.y + threadIdx.x;

    const int xoff = blockDim.x;
    const int yoff = blockDim.y;

    const int xend = (blockIdx.x + 1) * blockDim.x * 2;
    const int yend = (blockIdx.y + 1) * blockDim.y * 8;

    const int bid = blockIdx.y * gridDim.x + blockIdx.x;

    __shared__ unsigned s_count;
    __shared__ unsigned s_idx;

    if (tid == 0) {
        s_count  = d_counts [bid];
        s_idx    = d_offsets[bid];
    }
    __syncthreads();

    // Blocks that are empty, please bail
    if (s_count == 0) return;
    for (int y = yid; y < yend; y += yoff) {
        if (y >= idim1 - 1 || y <= 1) continue;
        for (int x = xid; x < xend; x += xoff) {
            if (x >= idim0 - 1 || x <= 1) continue;

            T v = flags[y * idim0 + x];
            if (v == 0) continue;

            unsigned id = atomicAdd(&s_idx, 1u);
            if (id >= total) return;
            y_out[id] = x;
            x_out[id] = y;
            score_out[id] = v;
        }
    }
}

template<typename T>
void fast(unsigned* out_feat,
          float** x_out,
          float** y_out,
          float** score_out,
          CParam<T> in,
          const float thr,
          const unsigned arc_length,
          const unsigned nonmax,
          const float feature_ratio,
          const unsigned edge)
{
    const unsigned max_feat = ceil(in.dims[0] * in.dims[1] * feature_ratio);

    dim3 threads(16, 16);
    dim3 blocks(divup(in.dims[0]-edge*2, threads.x), divup(in.dims[1]-edge*2, threads.y));

    // Matrix containing scores for detected features, scores are stored in the
    // same coordinates as features, dimensions should be equal to in.
    T *d_score = NULL;
    size_t score_bytes = in.dims[0] * in.dims[1] * sizeof(T) + sizeof(unsigned);
    d_score = (T *)memAlloc<char>(score_bytes);
    CUDA_CHECK(cudaMemset(d_score, 0, score_bytes));

    T *d_flags = d_score;
    if (nonmax) {
        d_flags = memAlloc<T>(in.dims[0] * in.dims[1]);
    }

    // Shared memory size
    size_t shared_size = (threads.x + 6) * (threads.y + 6) * sizeof(T);

    switch(arc_length) {
    case 9:
        locate_features<T, 9><<<blocks, threads, shared_size>>>(in, d_score, thr, edge);
        break;
    case 10:
        locate_features<T,10><<<blocks, threads, shared_size>>>(in, d_score, thr, edge);
        break;
    case 11:
        locate_features<T,11><<<blocks, threads, shared_size>>>(in, d_score, thr, edge);
        break;
    case 12:
        locate_features<T,12><<<blocks, threads, shared_size>>>(in, d_score, thr, edge);
        break;
    case 13:
        locate_features<T,13><<<blocks, threads, shared_size>>>(in, d_score, thr, edge);
        break;
    case 14:
        locate_features<T,14><<<blocks, threads, shared_size>>>(in, d_score, thr, edge);
        break;
    case 15:
        locate_features<T,15><<<blocks, threads, shared_size>>>(in, d_score, thr, edge);
        break;
    case 16:
        locate_features<T,16><<<blocks, threads, shared_size>>>(in, d_score, thr, edge);
        break;
    }

    POST_LAUNCH_CHECK();

    threads.x = 32;
    threads.y =  8;

    blocks.x = divup(in.dims[0], 64);
    blocks.y = divup(in.dims[1], 64);

    unsigned *d_total = (unsigned *)(d_score + in.dims[0] * in.dims[1]);
    unsigned *d_counts  = memAlloc<unsigned>(blocks.x * blocks.y);
    unsigned *d_offsets =memAlloc<unsigned>(blocks.x * blocks.y);

    if (nonmax)
        non_max_counts<T, true ><<<blocks, threads>>>(d_counts, d_offsets, d_total, d_flags,
                                                       d_score, in.dims[0], in.dims[1]);
    else
        non_max_counts<T, false><<<blocks, threads>>>(d_counts, d_offsets, d_total, d_flags,
                                                       d_score, in.dims[0], in.dims[1]);

    POST_LAUNCH_CHECK();

    // Dimensions of output array
    unsigned total;
    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(unsigned), cudaMemcpyDeviceToHost));
    total = total < max_feat ? total : max_feat;

    if (total > 0) {
        *x_out     = memAlloc<float>(total);
        *y_out     = memAlloc<float>(total);
        *score_out = memAlloc<float>(total);

        get_features<T><<<blocks, threads>>>(*x_out, *y_out, *score_out, d_flags, d_counts,
                                             d_offsets, total, in.dims[0], in.dims[1]);

        POST_LAUNCH_CHECK();
    }

    *out_feat = total;

    memFree<uchar>((uchar *)d_score);
    memFree(d_counts);
    memFree(d_offsets);
    if (nonmax) {
        memFree(d_flags);
    }
}

} // namespace kernel

} // namespace cuda

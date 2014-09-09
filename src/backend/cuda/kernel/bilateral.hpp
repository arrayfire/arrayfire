#include <af/defines.h>
#include <backend.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include "shared.hpp"

namespace cuda
{

namespace kernel
{

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

__forceinline__ __device__ dim_type lIdx(dim_type x, dim_type y,
        dim_type stride1, dim_type stride0)
{
    return (y*stride1 + x*stride0);
}

__forceinline__ __device__ dim_type clamp(dim_type f, dim_type a, dim_type b)
{
    return max(a, min(f, b));
}

__forceinline__ __device__ float gaussian1d(float x, float variance)
{
    const float exponent = (x * x) / (-2.f * variance);
    return exp(exponent);
}

template<typename T>
inline __device__ void load2ShrdMem(T * shrd, const T * const in,
                                    dim_type lx, dim_type ly, dim_type shrdStride,
                                    dim_type dim0, dim_type dim1,
                                    dim_type gx, dim_type gy,
                                    dim_type inStride1, dim_type inStride0)
{
    int gx_  = clamp(gx, 0, dim0-1);
    int gy_  = clamp(gy, 0, dim1-1);
    shrd[lIdx(lx, ly, shrdStride, 1)] = in[lIdx(gx_, gy_, inStride1, inStride0)];
}

template<typename T>
static __global__
void bilateralKernel(Param<T> out, CParam<T> in,
                     float sigma_space, float sigma_color,
                     dim_type gaussOff, dim_type nonBatchBlkSize)
{
    SharedMemory<T> shared;
    T *localMem = shared.getPointer();
    T *gauss2d  = localMem + gaussOff;

    const dim_type radius      = max((int)(sigma_space * 1.5f), 1);
    const dim_type padding     = 2 * radius;
    const dim_type window_size = padding + 1;
    const dim_type shrdLen     = blockDim.x + padding;
    const float variance_range = sigma_color * sigma_color;
    const float variance_space = sigma_space * sigma_space;

    // gfor batch offsets
    unsigned batchId = blockIdx.x / nonBatchBlkSize;
    const T* iptr      = (const T *) in.ptr + (batchId * in.strides[2]);
    T*       optr      = (T *      )out.ptr + (batchId * out.strides[2]);

    const dim_type lx = threadIdx.x;
    const dim_type ly = threadIdx.y;

    const dim_type gx = blockDim.x * (blockIdx.x-batchId*nonBatchBlkSize) + lx;
    const dim_type gy = blockDim.y * blockIdx.y + ly;

    dim_type gx2 = gx + blockDim.x;
    dim_type gy2 = gy + blockDim.y;
    dim_type lx2 = lx + blockDim.x;
    dim_type ly2 = ly + blockDim.y;
    dim_type i   = lx + radius;
    dim_type j   = ly + radius;

    // generate gauss2d spatial variance values for block
    if (lx<window_size && ly<window_size) {
        int x = lx - radius;
        int y = ly - radius;
        gauss2d[ly*window_size+lx] = exp( ((x*x) + (y*y)) / (-2.f * variance_space));
    }

    // pull image to local memory
    load2ShrdMem(localMem, iptr, lx, ly, shrdLen,
                 in.dims[0], in.dims[1], gx-radius,
                 gy-radius, in.strides[1], in.strides[0]);
    if (lx<padding) {
        load2ShrdMem(localMem, iptr, lx2, ly, shrdLen,
                     in.dims[0], in.dims[1], gx2-radius,
                     gy-radius, in.strides[1], in.strides[0]);
    }
    if (ly<padding) {
        load2ShrdMem(localMem, iptr, lx, ly2, shrdLen,
                     in.dims[0], in.dims[1], gx-radius,
                     gy2-radius, in.strides[1], in.strides[0]);
    }
    if (lx<padding && ly<padding) {
        load2ShrdMem(localMem, iptr, lx2, ly2, shrdLen,
                     in.dims[0], in.dims[1], gx2-radius,
                     gy2-radius, in.strides[1], in.strides[0]);
    }
    __syncthreads();

    if (gx<in.dims[0] && gy<in.dims[1]) {
        const float center_color = (float)localMem[j*shrdLen+i];
        float res  = 0;
        float norm = 0;
#pragma unroll
        for(dim_type wj=0; wj<window_size; ++wj) {
            dim_type joff = (j+wj-radius)*shrdLen;
            dim_type goff = wj*window_size;
#pragma unroll
            for(dim_type wi=0; wi<window_size; ++wi) {
                const float tmp_color   = (float)localMem[joff+i+wi-radius];
                const float gauss_space = gauss2d[goff+wi];
                const float gauss_range = gaussian1d(center_color - tmp_color, variance_range);
                const float weight      = gauss_space * gauss_range;
                norm += weight;
                res  += tmp_color * weight;
            }
        }
        dim_type oIdx = gy*out.strides[1] + gx*out.strides[0];
        optr[oIdx] = (T)(res / norm);
    }
}

template<typename T, bool isColor>
void bilateral(Param<T> out, CParam<T> in, float s_sigma, float c_sigma)
{
    dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    dim_type blk_x = divup(in.dims[0], THREADS_X);
    dim_type blk_y = divup(in.dims[1], THREADS_Y);

    dim_type bCount = blk_x * in.dims[2];
    if (isColor)
        bCount *= in.dims[3];

    dim3 blocks(bCount, blk_y);

    // calculate shared memory size
    dim_type radius = (dim_type)std::max(s_sigma * 1.5f, 1.f);
    dim_type num_shrd_elems    = (THREADS_X + 2 * radius) * (THREADS_Y + 2 * radius);
    dim_type num_gauss_elems   = (2 * radius + 1)*(2 * radius + 1);
    dim_type total_shrd_size   = sizeof(T) * (num_shrd_elems + num_gauss_elems);

    bilateralKernel<T>
        <<<blocks, threads, total_shrd_size>>>
        (out, in, s_sigma, c_sigma, num_shrd_elems, blk_x);
}

}

}

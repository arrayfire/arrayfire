#include <af/defines.h>
#include <backend.hpp>
#include "../helper.hpp"

namespace cuda
{

namespace kernel
{

static const dim_type MAX_MORPH_FILTER_LEN = 17;
// cFilter is used by both 2d morph and 3d morph
// Maximum kernel size supported for 2d morph is 17x17*8 = 2312
// Maximum kernel size supported for 3d morph is 7x7x7*8 = 2744
__constant__ double cFilter[MAX_MORPH_FILTER_LEN*MAX_MORPH_FILTER_LEN];

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

static const dim_type CUBE_X    =  8;
static const dim_type CUBE_Y    =  8;
static const dim_type CUBE_Z    =  8;

template<typename T>
struct morph_param_t{
    T *             d_dst;
    const T *       d_src;
    dim_type      dims[4];
    dim_type      windLen;
    dim_type  istrides[4];
    dim_type  ostrides[4];
};

__forceinline__ __device__ dim_type lIdx(dim_type x, dim_type y,
        dim_type stride1, dim_type stride0)
{
    return (y*stride1 + x*stride0);
}

__forceinline__ __device__ dim_type clamp(dim_type f, dim_type a, dim_type b)
{
    return max(a, min(f, b));
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
    shrd[ lIdx(lx, ly, shrdStride, 1) ] = in[ lIdx(gx_, gy_, inStride1, inStride0) ];
}

// kernel assumes mask/filter is square and hence does the
// necessary operations accordingly.
template<typename T, bool isDilation>
static __global__ void morphKernel( const morph_param_t<T> params,
                                    dim_type nonBatchBlkSize)
{
    // get shared memory pointer
    SharedMemory<T> shared;
    T * shrdMem = shared.getPointer();

    // calculate necessary offset and window parameters
    const dim_type se_len = params.windLen;
    const dim_type halo   = se_len/2;
    const dim_type padding= 2*halo;
    const dim_type shrdLen= blockDim.x + padding + 1;

    // gfor batch offsets
    unsigned batchId = blockIdx.x / nonBatchBlkSize;
    const T* in      = (const T *)params.d_src + (batchId * params.istrides[2]);
    T* out           = (T *)params.d_dst + (batchId * params.ostrides[2]);

    dim_type gx, gy, i, j;
    { //scopping out unnecessary variables
    // local neighborhood indices
    const dim_type lx = threadIdx.x;
    const dim_type ly = threadIdx.y;

    // global indices
    gx = blockDim.x * (blockIdx.x-batchId*nonBatchBlkSize) + lx;
    gy = blockDim.y * blockIdx.y + ly;

    // offset values for pulling image to local memory
    dim_type lx2      = lx + blockDim.x;
    dim_type ly2      = ly + blockDim.y;
    dim_type gx2      = gx + blockDim.x;
    dim_type gy2      = gy + blockDim.y;

    // pull image to local memory
    load2ShrdMem(shrdMem, in, lx, ly, shrdLen,
                 params.dims[0], params.dims[1],
                 gx-halo, gy-halo,
                 params.istrides[1], params.istrides[0]);
    if (lx<padding) {
        load2ShrdMem(shrdMem, in, lx2, ly, shrdLen,
                     params.dims[0], params.dims[1],
                     gx2-halo, gy-halo,
                     params.istrides[1], params.istrides[0]);
    }
    if (ly<padding) {
        load2ShrdMem(shrdMem, in, lx, ly2, shrdLen,
                     params.dims[0], params.dims[1],
                     gx-halo, gy2-halo,
                     params.istrides[1], params.istrides[0]);
    }
    if (lx<padding && ly<padding) {
        load2ShrdMem(shrdMem, in, lx2, ly2, shrdLen,
                     params.dims[0], params.dims[1],
                     gx2-halo, gy2-halo,
                     params.istrides[1], params.istrides[0]);
    }
    i = lx + halo;
    j = ly + halo;
    }
    __syncthreads();

    const T * d_filt = (const T *)cFilter;
    T acc = shrdMem[ lIdx(i, j, shrdLen, 1) ];
#pragma unroll
    for(dim_type wj=j-halo; wj<=j+halo; ++wj) {
        dim_type joff   = (wj-j+halo)*se_len;
        dim_type w_joff = wj*shrdLen;
#pragma unroll
        for(dim_type wi=i-halo; wi<=i+halo; ++wi) {
            T cur  = shrdMem[w_joff + wi];
            if (d_filt[joff + wi-i+halo]) {
                if (isDilation)
                    acc = max(acc, cur);
                else
                    acc = min(acc, cur);
            }
        }
    }

    if (gx<params.dims[0] && gy<params.dims[1]) {
        dim_type outIdx = lIdx(gx, gy, params.ostrides[1], params.ostrides[0]);
        out[outIdx] = acc;
    }
}

__forceinline__ __device__ dim_type lIdx3D(dim_type x, dim_type y, dim_type z,
        dim_type stride2, dim_type stride1, dim_type stride0)
{
    return (z*stride2 + y*stride1 + x*stride0);
}

template<typename T>
inline __device__ void load2ShrdVolume(T * shrd, const T * const in,
        dim_type lx, dim_type ly, dim_type lz,
        dim_type shrdStride1, dim_type shrdStride2,
        dim_type dim0, dim_type dim1, dim_type dim2,
        dim_type gx, dim_type gy, dim_type gz,
        dim_type inStride2, dim_type inStride1, dim_type inStride0)
{
    int gx_  = clamp(gx,0,dim0-1);
    int gy_  = clamp(gy,0,dim1-1);
    int gz_  = clamp(gz,0,dim2-1);
    dim_type shrdIdx = lx + ly*shrdStride1 + lz*shrdStride2;
    dim_type inIdx   = gx_*inStride0 + gy_*inStride1 + gz_*inStride2;
    shrd[ shrdIdx ] = in[ inIdx ];
}

// kernel assumes mask/filter is square and hence does the
// necessary operations accordingly.
template<typename T, bool isDilation>
static __global__ void morph3DKernel(const morph_param_t<T> params)
{
    // get shared memory pointer
    SharedMemory<T> shared;
    T * shrdMem = shared.getPointer();

    const dim_type se_len    = params.windLen;
    const dim_type halo      = se_len/2;
    const dim_type padding   = 2*halo;

    const dim_type se_area   = se_len*se_len;
    const dim_type shrdLen   = blockDim.x + padding + 1;
    const dim_type shrdArea  = shrdLen * (blockDim.y+padding);

    const T* in = (const T *)params.d_src;
    T* out      = (T *)params.d_dst;

    dim_type gx, gy, gz, i, j, k;
    { // scoping out unnecessary variables
    const dim_type lx = threadIdx.x;
    const dim_type ly = threadIdx.y;
    const dim_type lz = threadIdx.z;

    gx = blockDim.x * blockIdx.x + lx;
    gy = blockDim.y * blockIdx.y + ly;
    gz = blockDim.z * blockIdx.z + lz;

    const dim_type gx2 = gx + blockDim.x;
    const dim_type gy2 = gy + blockDim.y;
    const dim_type gz2 = gz + blockDim.z;
    const dim_type lx2 = lx + blockDim.x;
    const dim_type ly2 = ly + blockDim.y;
    const dim_type lz2 = lz + blockDim.z;

    // pull volume to shared memory
    load2ShrdVolume(shrdMem, in, lx, ly, lz, shrdLen, shrdArea,
                    params.dims[0], params.dims[1], params.dims[2],
                    gx-halo, gy-halo, gz-halo,
                    params.istrides[2], params.istrides[1], params.istrides[0]);
    if (lx<padding) {
        load2ShrdVolume(shrdMem, in, lx2, ly, lz, shrdLen, shrdArea,
                        params.dims[0], params.dims[1], params.dims[2],
                        gx2-halo, gy-halo, gz-halo,
                        params.istrides[2], params.istrides[1], params.istrides[0]);
    }
    if (ly<padding) {
        load2ShrdVolume(shrdMem, in, lx, ly2, lz, shrdLen, shrdArea,
                        params.dims[0], params.dims[1], params.dims[2],
                        gx-halo, gy2-halo, gz-halo,
                        params.istrides[2], params.istrides[1], params.istrides[0]);
    }
    if (lz<padding) {
        load2ShrdVolume(shrdMem, in, lx, ly, lz2, shrdLen, shrdArea,
                        params.dims[0], params.dims[1], params.dims[2],
                        gx-halo, gy-halo, gz2-halo,
                        params.istrides[2], params.istrides[1], params.istrides[0]);
    }
    if (lx<padding && ly<padding) {
        load2ShrdVolume(shrdMem, in, lx2, ly2, lz, shrdLen, shrdArea,
                        params.dims[0], params.dims[1], params.dims[2],
                        gx2-halo, gy2-halo, gz-halo,
                        params.istrides[2], params.istrides[1], params.istrides[0]);
    }
    if (ly<padding && lz<padding) {
        load2ShrdVolume(shrdMem, in, lx, ly2, lz2, shrdLen, shrdArea,
                        params.dims[0], params.dims[1], params.dims[2],
                        gx-halo, gy2-halo, gz2-halo,
                        params.istrides[2], params.istrides[1], params.istrides[0]);
    }
    if (lz<padding && lx<padding) {
        load2ShrdVolume(shrdMem, in, lx2, ly, lz2, shrdLen, shrdArea,
                        params.dims[0], params.dims[1], params.dims[2],
                        gx2-halo, gy-halo, gz2-halo,
                        params.istrides[2], params.istrides[1], params.istrides[0]);
    }
    if (lx<padding && ly<padding && lz<padding) {
        load2ShrdVolume(shrdMem, in, lx2, ly2, lz2, shrdLen, shrdArea,
                        params.dims[0], params.dims[1], params.dims[2],
                        gx2-halo, gy2-halo, gz2-halo,
                        params.istrides[2], params.istrides[1], params.istrides[0]);
    }
    __syncthreads();
    // indices of voxel owned by current thread
    i  = lx + halo;
    j  = ly + halo;
    k  = lz + halo;
    }

    const T * d_filt = (const T *)cFilter;
    T acc = shrdMem[ lIdx3D(i, j, k, shrdArea, shrdLen, 1) ];
#pragma unroll
    for(dim_type wk=k-halo; wk<=k+halo; ++wk) {
        dim_type koff   = (wk-k+halo)*se_area;
        dim_type w_koff = wk*shrdArea;
#pragma unroll
        for(dim_type wj=j-halo; wj<=j+halo; ++wj) {
        dim_type joff   = (wj-j+halo)*se_len;
        dim_type w_joff = wj*shrdLen;
#pragma unroll
            for(dim_type wi=i-halo; wi<=i+halo; ++wi) {
                T cur  = shrdMem[w_koff + w_joff + wi];
                if (d_filt[koff+joff+wi-i+halo]) {
                    if (isDilation)
                        acc = max(acc, cur);
                    else
                        acc = min(acc, cur);
                }
            }
        }
    }

    if (gx<params.dims[0] && gy<params.dims[1] && gz<params.dims[2]) {
        dim_type outIdx = gz * params.ostrides[2] +
                          gy * params.ostrides[1] +
                          gx * params.ostrides[0];
        out[outIdx] = acc;
    }
}

template<typename T, bool isDilation>
void morph(const morph_param_t<T> &kernelParams)
{
    dim3 threads(kernel::THREADS_X, kernel::THREADS_Y);

    dim_type blk_x = divup(kernelParams.dims[0], THREADS_X);
    dim_type blk_y = divup(kernelParams.dims[1], THREADS_Y);
    // launch batch * blk_x blocks along x dimension
    dim3 blocks(blk_x*kernelParams.dims[2], blk_y);

    // calculate shared memory size
    int halo      = kernelParams.windLen/2;
    int padding   = 2*halo;
    int shrdLen   = kernel::THREADS_X + padding + 1; // +1 for to avoid bank conflicts
    int shrdSize  = shrdLen * (kernel::THREADS_Y+padding) * sizeof(T);

    morphKernel<T, isDilation> <<< blocks, threads, shrdSize>>>(kernelParams, blk_x);
}

template<typename T, bool isDilation>
void morph3d(const morph_param_t<T> &kernelParams)
{
    dim3 threads(kernel::CUBE_X, kernel::CUBE_Y, kernel::CUBE_Z);

    dim_type blk_x = divup(kernelParams.dims[0], CUBE_X);
    dim_type blk_y = divup(kernelParams.dims[1], CUBE_Y);
    dim_type blk_z = divup(kernelParams.dims[2], CUBE_Z);
    dim3 blocks(blk_x, blk_y, blk_z);

    // calculate shared memory size
    int halo      = kernelParams.windLen/2;
    int padding   = 2*halo;
    int shrdLen   = kernel::CUBE_X + padding + 1; // +1 for to avoid bank conflicts
    int shrdSize  = shrdLen * (kernel::CUBE_Y+padding) * (kernel::CUBE_Z+padding) * sizeof(T);

    morph3DKernel<T, isDilation> <<< blocks, threads, shrdSize>>>(kernelParams);
}

}
}

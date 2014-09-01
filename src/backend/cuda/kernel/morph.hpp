#include <af/defines.h>
#include <backend.hpp>
#include "../helper.hpp"

namespace cuda
{

namespace kernel
{

static const dim_type MAX_MORPH_FILTER_LEN = 17;
__constant__ double cFilter[MAX_MORPH_FILTER_LEN*MAX_MORPH_FILTER_LEN];

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

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

}
}

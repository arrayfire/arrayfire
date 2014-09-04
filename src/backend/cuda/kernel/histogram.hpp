#include <af/defines.h>
#include <backend.hpp>
#include "../helper.hpp"

namespace cuda
{

namespace kernel
{

static const unsigned MAX_BINS  = 4000;
static const dim_type THREADS_X =  256;
static const dim_type THRD_LOAD =   16;

template<typename inType, typename outType>
struct hist_param_t {
    outType *            d_dst;
    const inType *       d_src;
    const float2 *    d_minmax;
    dim_type          idims[4];
    dim_type       istrides[4];
    dim_type       ostrides[4];
};

__forceinline__ __device__ dim_type minimum(dim_type a, dim_type b)
{
  return (a < b ? a : b);
}

template<typename inType, typename outType>
static __global__
void histogramKernel(const hist_param_t<inType, outType> params,
                     dim_type len, dim_type nbins, dim_type blk_x)
{
    SharedMemory<outType> shared;
    outType * shrdMem = shared.getPointer();

    // offset minmax array to account for batch ops
    const float2 * d_minmax = params.d_minmax + blockIdx.y;

    // offset input and output to account for batch ops
    const inType *in  = params.d_src + blockIdx.y * params.istrides[2];
    outType * out     = params.d_dst + blockIdx.y * params.ostrides[2];

    int start = blockIdx.x * THRD_LOAD * blockDim.x + threadIdx.x;
    int end   = minimum((start + THRD_LOAD * blockDim.x), len);

    __shared__ float min;
    __shared__ float step;

    if (threadIdx.x == 0) {
        float2 minmax = *d_minmax;
        min  = minmax.x;
        step = (minmax.y-minmax.x) / (float)nbins;
    }

    for (int i = threadIdx.x; i < nbins; i += blockDim.x)
        shrdMem[i] = 0;
    __syncthreads();

    for (int row = start; row < end; row += blockDim.x) {
        int bin = (int)((in[row] - min) / step);
        bin     = (bin < 0)      ? 0         : bin;
        bin     = (bin >= nbins) ? (nbins-1) : bin;
        atomicAdd((shrdMem + bin), 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
        atomicAdd((out + i), shrdMem[i]);
    }
}

template<typename inType, typename outType>
void histogram(const hist_param_t<inType, outType> &params, dim_type nbins)
{
    dim3 threads(kernel::THREADS_X, 1);

    dim_type numElements= params.idims[0]*params.idims[1];

    dim_type blk_x = divup(numElements, THRD_LOAD*THREADS_X);

    dim_type batchCount = params.idims[2];

    dim3 blocks(blk_x, batchCount);

    dim_type smem_size = nbins * sizeof(outType);

    histogramKernel<inType, outType>
                 <<<blocks, threads, smem_size>>>
                 (params, numElements, nbins, blk_x);
}

}

}

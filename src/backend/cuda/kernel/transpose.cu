#include <af/defines.h>
#include <kernel/transpose.hpp>
#include "backend.h"
#include <helper.hpp>

namespace cuda
{
namespace kernel
{

    static const size_t TILE_DIM  = 32;
    static const size_t THREADS_X = TILE_DIM;
    static const size_t THREADS_Y = TILE_DIM/4;

    // Kernel is going access original data in colleased format
    template<typename T, bool is32Multiple>
    __global__
    void transpose( T * out, const T * in,
                    dim_type iDim0, dim_type iDim1,
                    dim_type iStride0, dim_type iStride1,
                    dim_type nonBatchBlkSize)
    {
        SharedMemory<T> shared;
        T * shrdMem = shared.getPointer();
        // create variables to hold output dimensions
        const size_t oDim0 = iDim1;
        const size_t oDim1 = iDim0;
        // calculate strides
        const size_t oStride0    = iStride0;
        const size_t oStride1    = oDim0 * oStride0;
        const size_t batchStride = iDim0 * iDim1;
        const size_t shrdStride  = blockDim.x + 1;
        // TODO: Launch multiple blocks along x dimension
        //       to handle batch later, for loop is just for now
        int lx      = threadIdx.x;
        int ly      = threadIdx.y;
        // batch based block Id
        size_t batchId = blockIdx.x / nonBatchBlkSize;
        size_t blkIdx_x= (blockIdx.x-batchId*nonBatchBlkSize);
        // calculate global indices
        int gx      = lx + blockDim.x * blkIdx_x;
        int gy      = ly + TILE_DIM * blockIdx.y;
        // offset in and out based on batch id
        int offset  = batchId*batchStride;
        const T* in_= in  + offset;
        T* out_     = out + offset;

#pragma unroll
        for (int rep = 0; rep < TILE_DIM; rep += blockDim.y) {
            int gy_ = gy+rep;
            if (is32Multiple || (gx<iDim0 && gy_<iDim1))
                shrdMem[(ly+rep)*shrdStride+lx] = in_[gy_*iStride1+gx];
        }
        __syncthreads();

        gx          = lx + blockDim.x * blockIdx.y;
        gy          = ly + TILE_DIM * blkIdx_x;

        for (int rep = 0; rep < TILE_DIM; rep += blockDim.y) {
            int gy_ = gy+rep;
            if (is32Multiple || (gx<oDim0 && gy_<oDim1))
                out_[gy_*oStride1+gx] = shrdMem[lx*shrdStride+(ly+rep)];
        }
    }

    template<typename T>
    void transpose(T * out, const T * in, const dim_type ndims, const dim_type * const dims, const dim_type * const strides)
    {
        // dimensions passed to this function should be input dimensions
        // any necessary transformations and dimension related calculations are
        // carried out here and inside the kernel
        dim3 threads(kernel::THREADS_X,kernel::THREADS_Y);


        size_t blk_x = divup(dims[0],TILE_DIM);
        size_t blk_y = divup(dims[1],TILE_DIM);
        // launch batch * blk_x blocks along x dimension
        dim3 blocks(blk_x*dims[2],blk_y);

        size_t sharedMemSize = (TILE_DIM+1)*TILE_DIM*sizeof(T);

        if (dims[0]%TILE_DIM==0 && dims[1]%TILE_DIM==0)
            transpose < T, true > <<< blocks,threads,sharedMemSize >>> (out,in,dims[0],dims[1],strides[0],strides[1],blk_x);
        else
            transpose < T, false > <<< blocks,threads,sharedMemSize >>> (out,in,dims[0],dims[1],strides[0],strides[1],blk_x);
    }

#define INSTANTIATE(T)                                          \
    template void transpose(T * out, const T * in,              \
            const dim_type ndims, const dim_type * const dims,  \
            const dim_type * const strides);

    INSTANTIATE(float)
    INSTANTIATE(cfloat)
    INSTANTIATE(double)
    INSTANTIATE(cdouble)
    INSTANTIATE(char)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)

}
}

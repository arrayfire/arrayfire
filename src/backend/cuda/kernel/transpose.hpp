#include <af/defines.h>
#include <backend.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_cuda.hpp>

namespace cuda
{

namespace kernel
{

    static const dim_type TILE_DIM  = 32;
    static const dim_type THREADS_X = TILE_DIM;
    static const dim_type THREADS_Y = TILE_DIM/4;

    // Kernel is going access original data in colleased format
    template<typename T, bool is32Multiple>
    __global__
    void transpose(Param<T> out, CParam<T> in,
                   dim_type nonBatchBlkSize)
    {
        __shared__ T shrdMem[TILE_DIM][TILE_DIM+1];

        // create variables to hold output dimensions
        const dim_type oDim0 = in.dims[1];
        const dim_type oDim1 = in.dims[0];

        // calculate strides
        const dim_type oStride1 = oDim0;
        const dim_type iStride1 = in.strides[1];

        const dim_type iDim0 = in.dims[0];
        const dim_type iDim1 = in.dims[1];

        // TODO: Launch multiple blocks along x dimension
        //       to handle batch later, for loop is just for now
        dim_type lx      = threadIdx.x;
        dim_type ly      = threadIdx.y;

        // batch based block Id
        dim_type batchId = blockIdx.x / nonBatchBlkSize;
        dim_type blkIdx_x= (blockIdx.x-batchId*nonBatchBlkSize);

        // calculate global indices
        dim_type gx      = lx + blockDim.x * blkIdx_x;
        dim_type gy      = ly + TILE_DIM * blockIdx.y;

        // offset in and out based on batch id
        in.ptr  += batchId * in.strides[2];
        out.ptr += batchId * oDim0 * oDim1;

#pragma unroll
        for (dim_type repeat = 0; repeat < TILE_DIM; repeat += blockDim.y) {
            dim_type gy_ = gy+repeat;
            if (is32Multiple || (gx<iDim0 && gy_<iDim1))
                shrdMem[ly + repeat][lx] = in.ptr[gy_ * iStride1 + gx];
        }
        __syncthreads();

        gx          = lx + blockDim.x * blockIdx.y;
        gy          = ly + TILE_DIM * blkIdx_x;

        for (dim_type repeat = 0; repeat < TILE_DIM; repeat += blockDim.y) {
            dim_type gy_ = gy+repeat;
            if (is32Multiple || (gx<oDim0 && gy_<oDim1))
                out.ptr[gy_ * oStride1 + gx] = shrdMem[lx][ly + repeat];
        }
    }

    template<typename T>
    void transpose(Param<T> out, CParam<T> in, const dim_type ndims)
    {
        // dimensions passed to this function should be input dimensions
        // any necessary transformations and dimension related calculations are
        // carried out here and inside the kernel
        dim3 threads(kernel::THREADS_X,kernel::THREADS_Y);


        dim_type blk_x = divup(in.dims[0],TILE_DIM);
        dim_type blk_y = divup(in.dims[1],TILE_DIM);
        // launch batch * blk_x blocks along x dimension
        dim3 blocks(blk_x*in.dims[2],blk_y);

        if (in.dims[0]%TILE_DIM==0 && in.dims[1]%TILE_DIM==0)
            (transpose< T, true >)<<< blocks,threads >>>(out, in, blk_x);
        else
            (transpose< T, false>)<<< blocks,threads >>>(out, in, blk_x);

        POST_LAUNCH_CHECK();
    }

}

}

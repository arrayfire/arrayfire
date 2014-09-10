#include <math.hpp>
#include <dispatch.hpp>
#include <Param.hpp>

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const unsigned TX = 16;
        static const unsigned TY = 16;

        template<typename T>
        __global__
        void tile_kernel(Param<T> out, CParam<T> in,
                         const dim_type blocksPerMatX, const dim_type blocksPerMatY)
        {
            const dim_type oz = blockIdx.x / blocksPerMatX;
            const dim_type ow = blockIdx.y / blocksPerMatY;

            const dim_type blockIdx_x = blockIdx.x - oz * blocksPerMatX;
            const dim_type blockIdx_y = blockIdx.y - ow * blocksPerMatY;

            const dim_type ox = threadIdx.x + blockIdx_x * blockDim.x;
            const dim_type oy = threadIdx.y + blockIdx_y * blockDim.y;

            if(ox >= out.dims[0] ||
               oy >= out.dims[1] ||
               oz >= out.dims[2] ||
               ow >= out.dims[3])
                return;

            const dim_type ix = (in.dims[0] == out.dims[0]) ? ox : ox - ((ox / in.dims[0]) * in.dims[0]);
            const dim_type iy = (in.dims[1] == out.dims[1]) ? oy : oy - ((oy / in.dims[1]) * in.dims[1]);
            const dim_type iz = (in.dims[2] == out.dims[2]) ? oz : oz - ((oz / in.dims[2]) * in.dims[2]);
            const dim_type iw = (in.dims[3] == out.dims[3]) ? ow : ow - ((ow / in.dims[3]) * in.dims[3]);

            unsigned iMem = iw * in.strides[3] + iz * in.strides[2] +
                            iy * in.strides[1] + ix;
            unsigned oMem = ow * out.strides[3] + oz * out.strides[2] +
                            oy * out.strides[1] + ox;

            out.ptr[oMem] = in.ptr[iMem];
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        void tile(Param<T> out, CParam<T> in)
        {
            dim3 threads(TX, TY, 1);

            dim_type blocksPerMatX = divup(out.dims[0], TX);
            dim_type blocksPerMatY = divup(out.dims[1], TY);
            dim3 blocks(blocksPerMatX * out.dims[2],
                        blocksPerMatY * out.dims[3],
                        1);

            tile_kernel<T><<<blocks, threads>>>(out, in, blocksPerMatX, blocksPerMatY);
        }
    }
}

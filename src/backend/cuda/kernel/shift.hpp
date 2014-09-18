#include <math.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <cassert>

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const unsigned TX = 32;
        static const unsigned TY = 8;
        static const unsigned TILEX = 128;
        static const unsigned TILEY = 32;

        __host__ __device__
        static inline dim_type simple_mod(const dim_type i, const dim_type dim)
        {
            return (i < dim) ? i : (i - dim);
        }

        template<typename T>
        __global__
        void shift_kernel(Param<T> out, CParam<T> in, const dim_type d0, const dim_type d1,
                            const dim_type d2, const dim_type d3,
                            const dim_type blocksPerMatX, const dim_type blocksPerMatY)
        {
            const dim_type oz = blockIdx.x / blocksPerMatX;
            const dim_type ow = blockIdx.y / blocksPerMatY;

            const dim_type blockIdx_x = blockIdx.x - oz * blocksPerMatX;
            const dim_type blockIdx_y = blockIdx.y - ow * blocksPerMatY;

            const dim_type xx = threadIdx.x + blockIdx_x * blockDim.x;
            const dim_type yy = threadIdx.y + blockIdx_y * blockDim.y;

            if(xx >= out.dims[0] ||
               yy >= out.dims[1] ||
               oz >= out.dims[2] ||
               ow >= out.dims[3])
                return;

            const dim_type incy = blocksPerMatY * blockDim.y;
            const dim_type incx = blocksPerMatX * blockDim.x;

            const dim_type iw = simple_mod((ow + d3), out.dims[3]);
            const dim_type iz = simple_mod((oz + d2), out.dims[2]);

            const dim_type o_off = ow * out.strides[3] + oz * out.strides[2];
            const dim_type i_off = iw *  in.strides[3] + iz *  in.strides[2];

            for(dim_type oy = yy; oy < out.dims[1]; oy += incy) {
                const dim_type iy = simple_mod((oy + d1), out.dims[1]);
                for(dim_type ox = xx; ox < out.dims[0]; ox += incx) {
                    const dim_type ix = simple_mod((ox + d0), out.dims[0]);

                    const dim_type oIdx = o_off + oy * out.strides[1] + ox;
                    const dim_type iIdx = i_off + iy *  in.strides[1] + ix;

                    out.ptr[oIdx] = in.ptr[iIdx];
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        void shift(Param<T> out, CParam<T> in, const dim_type *sdims)
        {
            dim3 threads(TX, TY, 1);

            dim_type blocksPerMatX = divup(out.dims[0], TILEX);
            dim_type blocksPerMatY = divup(out.dims[1], TILEY);
            dim3 blocks(blocksPerMatX * out.dims[2],
                        blocksPerMatY * out.dims[3],
                        1);

            dim_type sdims_[4];
            // Need to do this because we are mapping output to input in the kernel
            for(int i = 0; i < 4; i++) {
                // sdims_[i] will always be positive and always [0, oDims[i]].
                // Negative shifts are converted to position by going the other way round
                sdims_[i] = -(sdims[i] % out.dims[i]) + out.dims[i] * (sdims[i] > 0);
                assert(sdims_[i] >= 0 && sdims_[i] <= out.dims[i]);
            }

            shift_kernel<T><<<blocks, threads>>>(out, in, sdims_[0], sdims_[1], sdims_[2], sdims_[3],
                                                 blocksPerMatX, blocksPerMatY);
            POST_LAUNCH_CHECK();
        }
    }
}

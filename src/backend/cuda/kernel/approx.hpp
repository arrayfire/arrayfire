/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include "interp.hpp"

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const int TX = 16;
        static const int TY = 16;
        static const int THREADS = 256;

        template<typename Ty, typename Tp, int order>
        __global__
        void approx1_kernel(Param<Ty> out, CParam<Ty> in, CParam<Tp> xpos,
                            const float offGrid, const int blocksMatX, const bool batch,
                            af_interp_type method)
        {
            const int idy = blockIdx.x / blocksMatX;
            const int blockIdx_x = blockIdx.x - idy * blocksMatX;
            const int idx = blockIdx_x * blockDim.x + threadIdx.x;

            const int idw = (blockIdx.y + blockIdx.z * gridDim.y) / out.dims[2];
            const int idz = (blockIdx.y + blockIdx.z * gridDim.y) - idw * out.dims[2];

            if (idx >= out.dims[0] || idy >= out.dims[1] ||
                idz >= out.dims[2] || idw >= out.dims[3])
                return;

            const int omId = idw * out.strides[3] + idz * out.strides[2]
                                               + idy * out.strides[1] + idx;
            int xmid = idx;
            if(batch) xmid += idw * xpos.strides[3] + idz * xpos.strides[2] + idy * xpos.strides[1];

            const Tp x = xpos.ptr[xmid];
            if (x < 0 || in.dims[0] < x+1) {
                out.ptr[omId] = scalar<Ty>(offGrid);
                return;
            }

            int ioff = idw * in.strides[3] + idz * in.strides[2] + idy * in.strides[1];

            // FIXME: Only cubic interpolation is doing clamping
            // We need to make it consistent across all methods
            // Not changing the behavior because tests will fail
            bool clamp = order == 3;

            Interp1<Ty, Tp, order> interp;
            interp(out, omId, in, ioff, x, method, 1, clamp);
        }

        template<typename Ty, typename Tp, int order>
        __global__
        void approx2_kernel(Param<Ty> out, CParam<Ty> in,
                            CParam<Tp> xpos, CParam<Tp> ypos, const float offGrid,
                            const int blocksMatX, const int blocksMatY, const bool batch,
                            af_interp_type method)
        {
            const int idz = blockIdx.x / blocksMatX;
            const int blockIdx_x = blockIdx.x - idz * blocksMatX;
            const int idx = threadIdx.x + blockIdx_x * blockDim.x;

            const int idw = (blockIdx.y + blockIdx.z * gridDim.y) / blocksMatY;
            const int blockIdx_y = (blockIdx.y + blockIdx.z * gridDim.y) - idw * blocksMatY;
            const int idy = threadIdx.y + blockIdx_y * blockDim.y;

            if (idx >= out.dims[0] || idy >= out.dims[1] ||
                idz >= out.dims[2] || idw >= out.dims[3])
                return;

            const int omId = idw * out.strides[3] + idz * out.strides[2]
                + idy * out.strides[1] + idx;
            int xmid = idy * xpos.strides[1] + idx;
            int ymid = idy * ypos.strides[1] + idx;
            if(batch) {
                xmid += idw * xpos.strides[3] + idz * xpos.strides[2];
                ymid += idw * ypos.strides[3] + idz * ypos.strides[2];
            }

            const Tp x = xpos.ptr[xmid], y = ypos.ptr[ymid];
            if (x < 0 || y < 0 || in.dims[0] < x+1 || in.dims[1] < y+1) {
                out.ptr[omId] = scalar<Ty>(offGrid);
                return;
            }

            int ioff = idw * in.strides[3] + idz * in.strides[2];

            // FIXME: Only cubic interpolation is doing clamping
            // We need to make it consistent across all methods
            // Not changing the behavior because tests will fail
            bool clamp = order == 3;

            Interp2<Ty, Tp, order> interp;
            interp(out, omId, in, ioff, x, y, method, 1, clamp);
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename Ty, typename Tp, int order>
        void approx1(Param<Ty> out, CParam<Ty> in,
                     CParam<Tp> xpos, const float offGrid,
                     af_interp_type method)
        {
            dim3 threads(THREADS, 1, 1);
            int blocksPerMat = divup(out.dims[0], threads.x);
            dim3 blocks(blocksPerMat * out.dims[1], out.dims[2] * out.dims[3]);

            bool batch = !(xpos.dims[1] == 1 && xpos.dims[2] == 1 && xpos.dims[3] == 1);

            const int maxBlocksY    = cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
            const int blocksPerMatZ = divup(blocks.y, maxBlocksY);
            if(blocksPerMatZ > 1) {
                blocks.y = maxBlocksY;
                blocks.z = blocksPerMatZ;
            }
            CUDA_LAUNCH((approx1_kernel<Ty, Tp, order>), blocks, threads,
                            out, in, xpos, offGrid, blocksPerMat, batch, method);
            POST_LAUNCH_CHECK();
        }

        template <typename Ty, typename Tp, int order>
        void approx2(Param<Ty> out, CParam<Ty> in,
                     CParam<Tp> xpos, CParam<Tp> ypos, const float offGrid,
                     af_interp_type method)
        {
            dim3 threads(TX, TY, 1);
            int blocksPerMatX = divup(out.dims[0], threads.x);
            int blocksPerMatY = divup(out.dims[1], threads.y);
            dim3 blocks(blocksPerMatX * out.dims[2], blocksPerMatY * out.dims[3]);

            bool batch = !(xpos.dims[2] == 1 && xpos.dims[3] == 1);

            const int maxBlocksY    = cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
            const int blocksPerMatZ = divup(blocks.y, maxBlocksY);
            if(blocksPerMatZ > 1) {
                blocks.y = maxBlocksY;
                blocks.z = blocksPerMatZ;
            }
            CUDA_LAUNCH((approx2_kernel<Ty, Tp, order>), blocks, threads,
                        out, in, xpos, ypos, offGrid, blocksPerMatX, blocksPerMatY, batch, method);
            POST_LAUNCH_CHECK();
        }
    }
}

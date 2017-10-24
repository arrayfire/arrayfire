/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>
#include <common/dispatch.hpp>
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
        void approx1_kernel(Param<Ty> yo, CParam<Ty> yi,
                            CParam<Tp> xo, const int xdim,
                            const Tp xi_beg, const Tp xi_step,
                            const float offGrid, const int blocksMatX, const bool batch,
                            af_interp_type method)
        {
            const int idy = blockIdx.x / blocksMatX;
            const int blockIdx_x = blockIdx.x - idy * blocksMatX;
            const int idx = blockIdx_x * blockDim.x + threadIdx.x;

            const int idw = (blockIdx.y + blockIdx.z * gridDim.y) / yo.dims[2];
            const int idz = (blockIdx.y + blockIdx.z * gridDim.y) - idw * yo.dims[2];

            if (idx >= yo.dims[0] || idy >= yo.dims[1] ||
                idz >= yo.dims[2] || idw >= yo.dims[3])
                return;

            const int omId = idw * yo.strides[3] + idz * yo.strides[2] + idy * yo.strides[1] + idx;
            int xmid = idx;
            if(batch) xmid += idw * xo.strides[3] + idz * xo.strides[2] + idy * xo.strides[1];

            const Tp x = (xo.ptr[xmid] - xi_beg) / xi_step;
            if (x < 0 || yi.dims[0] < x+1) {
                yo.ptr[omId] = scalar<Ty>(offGrid);
                return;
            }

            int ioff = idw * yi.strides[3] + idz * yi.strides[2] + idy * yi.strides[1];

            // FIXME: Only cubic interpolation is doing clamping
            // We need to make it consistent across all methods
            // Not changing the behavior because tests will fail
            bool clamp = order == 3;

            Interp1<Ty, Tp, order> interp;
            interp(yo, omId, yi, ioff, x, method, 1, clamp);
        }

        template<typename Ty, typename Tp, int order>
        __global__
        void approx2_kernel(Param<Ty> zo, CParam<Ty> zi,
                            CParam<Tp> xo, const int xdim,
                            CParam<Tp> yo, const int ydim,
                            const Tp xi_beg, const Tp xi_step,
                            const Tp yi_beg, const Tp yi_step,
                            const float offGrid,
                            const int blocksMatX, const int blocksMatY, const bool batch,
                            af_interp_type method)
        {
            const int idz = blockIdx.x / blocksMatX;
            const int blockIdx_x = blockIdx.x - idz * blocksMatX;
            const int idx = threadIdx.x + blockIdx_x * blockDim.x;

            const int idw = (blockIdx.y + blockIdx.z * gridDim.y) / blocksMatY;
            const int blockIdx_y = (blockIdx.y + blockIdx.z * gridDim.y) - idw * blocksMatY;
            const int idy = threadIdx.y + blockIdx_y * blockDim.y;

            if (idx >= zo.dims[0] || idy >= zo.dims[1] ||
                idz >= zo.dims[2] || idw >= zo.dims[3])
                return;

            const int omId = idw * zo.strides[3] + idz * zo.strides[2] + idy * zo.strides[1] + idx;
            int xmid = idy * xo.strides[1] + idx;
            int ymid = idy * yo.strides[1] + idx;
            if(batch) {
                xmid += idw * xo.strides[3] + idz * xo.strides[2];
                ymid += idw * yo.strides[3] + idz * yo.strides[2];
            }

            const Tp x = (xo.ptr[xmid] - xi_beg) / xi_step;
            const Tp y = (yo.ptr[ymid] - yi_beg) / yi_step;
            if (x < 0 || y < 0 || zi.dims[0] < x+1 || zi.dims[1] < y+1) {
                zo.ptr[omId] = scalar<Ty>(offGrid);
                return;
            }

            int ioff = idw * zi.strides[3] + idz * zi.strides[2];

            // FIXME: Only cubic interpolation is doing clamping
            // We need to make it consistent across all methods
            // Not changing the behavior because tests will fail
            bool clamp = order == 3;

            Interp2<Ty, Tp, order> interp;
            interp(zo, omId, zi, ioff, x, y, method, 1, clamp);
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename Ty, typename Tp, int order>
        void approx1(Param<Ty> yo, CParam<Ty> yi,
                     CParam<Tp> xo, const int xdim,
                     const Tp &xi_beg, const Tp &xi_step,
                     const float offGrid,
                     af_interp_type method)
        {
            dim3 threads(THREADS, 1, 1);
            int blocksPerMat = divup(yo.dims[0], threads.x);
            dim3 blocks(blocksPerMat * yo.dims[1], yo.dims[2] * yo.dims[3]);

            bool batch = !(xo.dims[1] == 1 && xo.dims[2] == 1 && xo.dims[3] == 1);

            const int maxBlocksY    = cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
            blocks.z = divup(blocks.y, maxBlocksY);
            blocks.y = divup(blocks.y, blocks.z);

            CUDA_LAUNCH((approx1_kernel<Ty, Tp, order>), blocks, threads,
                        yo, yi, xo, xdim, xi_beg, xi_step, offGrid, blocksPerMat, batch, method);
            POST_LAUNCH_CHECK();
        }

        template <typename Ty, typename Tp, int order>
        void approx2(Param<Ty> zo, CParam<Ty> zi,
                     CParam<Tp> xo, const int xdim,
                     CParam<Tp> yo, const int ydim,
                     const Tp &xi_beg, const Tp &xi_step,
                     const Tp &yi_beg, const Tp &yi_step,
                     const float offGrid,
                     af_interp_type method)
        {
            dim3 threads(TX, TY, 1);
            int blocksPerMatX = divup(zo.dims[0], threads.x);
            int blocksPerMatY = divup(zo.dims[1], threads.y);
            dim3 blocks(blocksPerMatX * zo.dims[2], blocksPerMatY * zo.dims[3]);

            bool batch = !(xo.dims[2] == 1 && xo.dims[3] == 1);

            const int maxBlocksY    = cuda::getDeviceProp(cuda::getActiveDeviceId()).maxGridSize[1];
            blocks.z = divup(blocks.y, maxBlocksY);
            blocks.y = divup(blocks.y, blocks.z);

            CUDA_LAUNCH((approx2_kernel<Ty, Tp, order>), blocks, threads,
                        zo, zi, xo, xdim, yo, ydim, xi_beg, xi_step, yi_beg, yi_step,
                        offGrid, blocksPerMatX, blocksPerMatY, batch, method);
            POST_LAUNCH_CHECK();
        }
    }
}

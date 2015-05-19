/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <backend.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_cuda.hpp>
#include <math.hpp>

namespace cuda
{

    namespace kernel
    {

        static const int MAX_A_SIZE = 1024;

        template<typename T, bool batch_a>
        __global__
        void iir_kernel(Param<T> y, CParam<T> c, CParam<T> a,
                        const int blocks_y)
        {
            __shared__ T s_z[MAX_A_SIZE];
            __shared__ T s_a[MAX_A_SIZE];
            __shared__ T s_y;

            const int idz = blockIdx.x;
            const int idw = blockIdx.y / blocks_y;
            const int idy = blockIdx.y - idw * blocks_y;

            const int tx = threadIdx.x;
            const int num_a = a.dims[0];

            int y_off = idw * y.strides[3] + idz * y.strides[2] + idy * y.strides[1];
            int c_off = idw * c.strides[3] + idz * c.strides[2] + idy * c.strides[1];
            int a_off = 0;

            if (batch_a) a_off = idw * a.strides[3] + idz * a.strides[2] + idy * a.strides[1];

            T *d_y = y.ptr + y_off;
            const T *d_c = c.ptr + c_off;
            const T *d_a = a.ptr + a_off;
            const int repeat = (num_a + blockDim.x - 1) / blockDim.x;

            for (int ii = 0; ii < MAX_A_SIZE / blockDim.x; ii++) {
                int id = ii * blockDim.x + tx;
                s_z[id] = scalar<T>(0);
                s_a[id] = (id < num_a) ? d_a[id] : scalar<T>(0);
            }
            __syncthreads();


            for (int i = 0; i < y.dims[0]; i++) {
                if (tx == 0) {
                    s_y = (d_c[i] + s_z[0]) / s_a[0];
                    d_y[i] = s_y;
                }
                __syncthreads();

#pragma unroll
                for (int ii = 0; ii < repeat; ii++) {
                    int id = ii * blockDim.x + tx + 1;

                    T z = s_z[id] - s_a[id] * s_y;
                    __syncthreads();

                    s_z[id - 1] = z;
                    __syncthreads();
                }
            }
        }

        template<typename T, bool batch_a>
        void iir(Param<T> y, CParam<T> c, CParam<T> a)
        {
            const int blocks_y = y.dims[1];
            const int blocks_x = y.dims[2];

            dim3 blocks(blocks_x,
                        blocks_y * y.dims[3]);

            int threads = 256;
            while (threads > y.dims[0] && threads > 32) threads /= 2;

            (iir_kernel<T, batch_a>)<<<blocks, threads>>>(y, c, a, blocks_y);
        }

    }
}

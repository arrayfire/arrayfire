/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>
#include <Param.hpp>
#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include "transform_interp.hpp"

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const unsigned TX = 16;
        static const unsigned TY = 16;

        typedef struct {
            float tmat[6];
        } tmat_t;

        ///////////////////////////////////////////////////////////////////////////
        // Rotate Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, af_interp_type method>
        __global__ static void
        rotate_kernel(Param<T> out, CParam<T> in, const tmat_t t, const dim_type nimages)
        {
            // Get thread indices
            const dim_type xx = blockIdx.x * blockDim.x + threadIdx.x;
            const dim_type yy = blockIdx.y * blockDim.y + threadIdx.y;

            if(xx >= out.dims[0] || yy >= out.dims[1])
                return;

            // Global offset
            //          Offset for transform channel + Offset for image channel.
                  T *optr = out.ptr;
            const T *iptr = in.ptr;

            switch(method) {
                case AF_INTERP_NEAREST:
                    transform_n(optr, out, iptr, in, t.tmat, xx, yy, nimages); break;
                case AF_INTERP_BILINEAR:
                    transform_b(optr, out, iptr, in, t.tmat, xx, yy, nimages); break;
                default: break;
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename T, af_interp_type method>
        void rotate(Param<T> out, CParam<T> in, const float theta)
        {
            const dim_type nimages = in.dims[2];

            const float c = cos(-theta), s = sin(-theta);
            float tx, ty;
            {
                const float nx = 0.5 * (in.dims[0] - 1);
                const float ny = 0.5 * (in.dims[1] - 1);
                const float mx = 0.5 * (out.dims[0] - 1);
                const float my = 0.5 * (out.dims[1] - 1);
                const float sx = (mx * c + my *-s);
                const float sy = (mx * s + my * c);
                tx = -(sx - nx);
                ty = -(sy - ny);
            }

            tmat_t t;
            t.tmat[0] =  c;
            t.tmat[1] = -s;
            t.tmat[2] = tx;
            t.tmat[3] =  s;
            t.tmat[4] =  c;
            t.tmat[5] = ty;

            dim3 threads(TX, TY, 1);
            dim3 blocks(divup(out.dims[0], threads.x), divup(out.dims[1], threads.y));

            rotate_kernel<T, method><<<blocks, threads>>> (out, in, t, nimages);

            POST_LAUNCH_CHECK();
        }
    }
}


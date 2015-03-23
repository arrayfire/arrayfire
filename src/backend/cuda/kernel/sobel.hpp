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

namespace cuda
{

namespace kernel
{

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

template<typename Ti>
__device__
Ti load2ShrdMem(const Ti * in,
               dim_type dim0, dim_type dim1,
               dim_type gx, dim_type gy,
               dim_type inStride1, dim_type inStride0)
{
    if (gx<0 || gx>=dim0 || gy<0 || gy>=dim1)
        return Ti(0);
    else
        return in[gx*inStride0+gy*inStride1];
}

template<typename Ti, typename To>
__global__
void sobel3x3(Param<To> dx, Param<To> dy, CParam<Ti> in, dim_type nBBS0, dim_type nBBS1)
{
    __shared__ Ti shrdMem[THREADS_X+2][THREADS_Y+2];

    // calculate necessary offset and window parameters
    const dim_type radius  = 1;
    const dim_type padding = 2*radius;

    // batch offsets
    unsigned b2 = blockIdx.x / nBBS0;
    unsigned b3 = blockIdx.y / nBBS1;
    const Ti* iptr     = (const Ti *)in.ptr + (b2 * in.strides[2] + b3 * in.strides[3]);
    To*       dxptr    = (To *      )dx.ptr + (b2 * dx.strides[2] + b3 * dx.strides[3]);
    To*       dyptr    = (To *      )dy.ptr + (b2 * dy.strides[2] + b3 * dy.strides[3]);

    // local neighborhood indices
    dim_type lx = threadIdx.x;
    dim_type ly = threadIdx.y;

    // global indices
    dim_type gx = THREADS_X * (blockIdx.x-b2*nBBS0) + lx;
    dim_type gy = THREADS_Y * (blockIdx.y-b3*nBBS1) + ly;

    // offset values for pulling image to local memory
    dim_type lx2 = lx + THREADS_X;
    dim_type ly2 = ly + THREADS_Y;
    dim_type gx2 = gx + THREADS_X;
    dim_type gy2 = gy + THREADS_Y;

    // pull image to local memory
    shrdMem[lx][ly] = load2ShrdMem<Ti>(iptr, in.dims[0], in.dims[1],
                                      gx-radius, gy-radius,
                                      in.strides[1], in.strides[0]);
    if (lx<padding) {
        shrdMem[lx2][ly] = load2ShrdMem<Ti>(iptr, in.dims[0], in.dims[1],
                                           gx2-radius, gy-radius,
                                           in.strides[1], in.strides[0]);
    }
    if (ly<padding) {
        shrdMem[lx][ly2] = load2ShrdMem<Ti>(iptr, in.dims[0], in.dims[1],
                                           gx-radius, gy2-radius,
                                           in.strides[1], in.strides[0]);
    }
    if (lx<padding && ly<padding) {
        shrdMem[lx2][ly2] = load2ShrdMem<Ti>(iptr, in.dims[0], in.dims[1],
                                            gx2-radius, gy2-radius,
                                            in.strides[1], in.strides[0]);
    }
    __syncthreads();

    // Only continue if we're at a valid location
    if (gx < in.dims[0] && gy < in.dims[1]) {
        dim_type i = lx + radius;
        dim_type j = ly + radius;
        dim_type _i = i-1;
        dim_type i_ = i+1;
        dim_type _j = j-1;
        dim_type j_ = j+1;

        float NW = shrdMem[_i][_j];
        float SW = shrdMem[i_][_j];
        float NE = shrdMem[_i][j_];
        float SE = shrdMem[i_][j_];

        float t1 = shrdMem[i][_j];
        float t2 = shrdMem[i][j_];
        dxptr[gy*dx.strides[1]+gx] = (NW+SW - (NE+SE) + 2*(t1-t2));

        t1 = shrdMem[_i][j];
        t2 = shrdMem[i_][j];
        dyptr[gy*dy.strides[1]+gx] = (NW+NE - (SW+SE) + 2*(t1-t2));

    }
}

template<typename Ti, typename To>
void sobel(Param<To> dx, Param<To> dy, CParam<Ti> in, const unsigned &ker_size)
{
    const dim3 threads(THREADS_X, THREADS_Y);

    dim_type blk_x = divup(in.dims[0], threads.x);
    dim_type blk_y = divup(in.dims[1], threads.y);

    dim3 blocks(blk_x*in.dims[2], blk_y*in.dims[3]);

    //TODO: add more cases when 5x5 and 7x7 kernels are done
    switch(ker_size) {
        case  3:
            (sobel3x3<Ti, To>) <<< blocks, threads >>> (dx, dy, in, blk_x, blk_y);
            break;
    }

    POST_LAUNCH_CHECK();
}

}

}

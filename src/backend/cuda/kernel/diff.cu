#include "clpx.hpp"
#include <kernel/diff.hpp>
#include <stdio.h>

#define divup(a, b) ((a+b-1)/b)

namespace cuda
{
namespace kernel
{
    // Kernel Launch Config Values
    static const unsigned TX = 16;
    static const unsigned TY = 16;
    static const unsigned TPB = TX * TY;

    ///////////////////////////////////////////////////////////////////////////
    // 1st and 2nd Order Differential Specialized for Vectors
    ///////////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__
    void diff1_1D(T *out, const T *in, const unsigned dim, const unsigned oElem)
    {
        unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

        if(idx > oElem)
            return;

        out[idx] = in[idx + 1] - in[idx];
    }

    template<typename T>
    __global__
    void diff2_1D(T *out, const T *in, const unsigned dim, const unsigned oElem)
    {
        unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

        if(idx > oElem)
            return;

        out[idx] = in[idx + 2] + - in[idx + 1] - in[idx + 1] + in[idx];
    }

    ///////////////////////////////////////////////////////////////////////////
    // 1st and 2nd Order Differential Specialized for Matrices along 0th Dim
    ///////////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__
    void diff1_2D(T *out, const T *in, const unsigned dim,
                   const unsigned oElem, const dim_type odims0,
                   const dim_type ostrides1, const dim_type istrides1)
    {
        const bool isDim0 = dim == 0;
        const bool isDim1 = dim == 1;
        unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned idy = threadIdx.y + blockIdx.y * blockDim.y;

        if(idx >= odims0)
            return;
        if(idx + idy * odims0 > oElem)
            return;

        unsigned iMem0 = (idy + 0 * isDim1) * istrides1 + (idx + 0 * isDim0);
        unsigned iMem1 = (idy + 1 * isDim1) * istrides1 + (idx + 1 * isDim0);
        unsigned oMem  =  idy * ostrides1 + idx;

        out[oMem] = in[iMem1] - in[iMem0];
    }

    template<typename T>
    __global__
    void diff2_2D(T *out, const T *in, const unsigned dim,
                   const unsigned oElem, const dim_type odims0,
                   const dim_type ostrides1, const dim_type istrides1)
    {
        const bool isDim0 = dim == 0;
        const bool isDim1 = dim == 1;
        unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned idy = threadIdx.y + blockIdx.y * blockDim.y;

        if(idx >= odims0)
            return;
        if(idx + idy * odims0 > oElem)
            return;

        unsigned iMem0 = (idy + 0 * isDim1) * istrides1 + (idx + 0 * isDim0);
        unsigned iMem1 = (idy + 1 * isDim1) * istrides1 + (idx + 1 * isDim0);
        unsigned iMem2 = (idy + 2 * isDim1) * istrides1 + (idx + 2 * isDim0);
        unsigned oMem =   idy * ostrides1 + idx;

        out[oMem] = in[iMem2] - in[iMem1] - in[iMem1] + in[iMem0];
    }

    /////////////////////////////////////////////////////////////////////////////
    // 1st and 2nd Order Differential for Volumes (3D) along all dimensions
    ///////////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__
    void diff1_3D(T *out, const T *in, const unsigned dim,
                  const unsigned oElem, const dim_type odims0, const dim_type odims1,
                  const dim_type ostrides1, const dim_type ostrides2,
                  const dim_type istrides1, const dim_type istrides2,
                  const unsigned blocksPerMatX)
    {
        const bool isDim0 = dim == 0;
        const bool isDim1 = dim == 1;
        const bool isDim2 = dim == 2;
        unsigned idx = threadIdx.x + (blockIdx.x % blocksPerMatX) * blockDim.x;
        unsigned idy = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned idz = blockIdx.x / blocksPerMatX;

        if(idx >= odims0 || idy >= odims1)
            return;
        if(idx + idy * odims0 + idz * odims0 * odims1 > oElem)
            return;

        unsigned iMem0 = (idz + 0 * isDim2) * istrides2 + (idy + 0 * isDim1) * istrides1 + (idx + 0 * isDim0);
        unsigned iMem1 = (idz + 1 * isDim2) * istrides2 + (idy + 1 * isDim1) * istrides1 + (idx + 1 * isDim0);
        unsigned oMem = idz * ostrides2 + idy * ostrides1 + idx;

        out[oMem] = in[iMem1] - in[iMem0];
    }

    template<typename T>
    __global__
    void diff2_3D(T *out, const T *in, const unsigned dim,
                  const unsigned oElem, const dim_type odims0, const dim_type odims1,
                  const dim_type ostrides1, const dim_type ostrides2,
                  const dim_type istrides1, const dim_type istrides2,
                  const unsigned blocksPerMatX)
    {
        const bool isDim0 = dim == 0;
        const bool isDim1 = dim == 1;
        const bool isDim2 = dim == 2;
        unsigned idx = threadIdx.x + (blockIdx.x % blocksPerMatX) * blockDim.x;
        unsigned idy = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned idz = blockIdx.x / blocksPerMatX;

        if(idx >= odims0 || idy >= odims1)
            return;
        if(idx + idy * odims0 + idz * odims0 * odims1 > oElem)
            return;

        unsigned iMem0 = (idz + 0 * isDim2) * istrides2 + (idy + 0 * isDim1) * istrides1 + (idx + 0 * isDim0);
        unsigned iMem1 = (idz + 1 * isDim2) * istrides2 + (idy + 1 * isDim1) * istrides1 + (idx + 1 * isDim0);
        unsigned iMem2 = (idz + 2 * isDim2) * istrides2 + (idy + 2 * isDim1) * istrides1 + (idx + 2 * isDim0);
        unsigned oMem = idz * ostrides2 + idy * ostrides1 + idx;

        out[oMem] = in[iMem2] - in[iMem1] - in[iMem1] + in[iMem0];
    }

    /////////////////////////////////////////////////////////////////////////////
    // 1st and 2nd Order Differential for 4D along all dimensions
    ///////////////////////////////////////////////////////////////////////////
    template<typename T>
    __global__
    void diff1_4D(T *out, const T *in, const unsigned dim,
                  const unsigned oElem, const dim_type odims0,
                  const dim_type odims1, const dim_type odims2,
                  const dim_type ostrides1, const dim_type ostrides2, const dim_type ostrides3,
                  const dim_type istrides1, const dim_type istrides2, const dim_type istrides3,
                  const unsigned blocksPerMatX, const unsigned blocksPerMatY)
    {
        const bool isDim0 = dim == 0;
        const bool isDim1 = dim == 1;
        const bool isDim2 = dim == 2;
        const bool isDim3 = dim == 3;
        unsigned idx = threadIdx.x + (blockIdx.x % blocksPerMatX) * blockDim.x;
        unsigned idy = threadIdx.y + (blockIdx.y % blocksPerMatY) * blockDim.y;
        unsigned idz = blockIdx.x / blocksPerMatX;
        unsigned idw = blockIdx.y / blocksPerMatY;

        if(idx >= odims0 || idy >= odims1 || idz >= odims2)
            return;
        if(idx + idy * odims0 + idz * odims0 * odims1 + idw * odims0 * odims1 * odims2 > oElem)
            return;

        unsigned iMem0 = (idw + 0 * isDim3) * istrides3 + (idz + 0 * isDim2) * istrides2 +
                         (idy + 0 * isDim1) * istrides1 + (idx + 0 * isDim0);
        unsigned iMem1 = (idw + 1 * isDim3) * istrides3 + (idz + 1 * isDim2) * istrides2 +
                         (idy + 1 * isDim1) * istrides1 + (idx + 1 * isDim0);
        unsigned oMem = idw * ostrides3 + idz * ostrides2 + idy * ostrides1 + idx;

        out[oMem] = in[iMem1] - in[iMem0];
    }

    template<typename T>
    __global__
    void diff2_3D(T *out, const T *in, const unsigned dim,
                  const unsigned oElem, const dim_type odims0,
                  const dim_type odims1, const dim_type odims2,
                  const dim_type ostrides1, const dim_type ostrides2, const dim_type ostrides3,
                  const dim_type istrides1, const dim_type istrides2, const dim_type istrides3,
                  const unsigned blocksPerMatX, const unsigned blocksPerMatY)
    {
        const bool isDim0 = dim == 0;
        const bool isDim1 = dim == 1;
        const bool isDim2 = dim == 2;
        const bool isDim3 = dim == 3;
        unsigned idx = threadIdx.x + (blockIdx.x % blocksPerMatX) * blockDim.x;
        unsigned idy = threadIdx.y + (blockIdx.y % blocksPerMatY) * blockDim.y;
        unsigned idz = blockIdx.x / blocksPerMatX;
        unsigned idw = blockIdx.y / blocksPerMatY;

        if(idx >= odims0 || idy >= odims1 || idz >= odims2)
            return;
        if(idx + idy * odims0 + idz * odims0 * odims1 + idw * odims0 * odims1 * odims2 > oElem)
            return;

        unsigned iMem0 = (idw + 0 * isDim3) * istrides3 + (idz + 0 * isDim2) * istrides2 +
                         (idy + 0 * isDim1) * istrides1 + (idx + 0 * isDim0);
        unsigned iMem1 = (idw + 1 * isDim3) * istrides3 + (idz + 1 * isDim2) * istrides2 +
                         (idy + 1 * isDim1) * istrides1 + (idx + 1 * isDim0);
        unsigned iMem2 = (idw + 2 * isDim3) * istrides3 + (idz + 2 * isDim2) * istrides2 +
                         (idy + 2 * isDim1) * istrides1 + (idx + 2 * isDim0);
        unsigned oMem = idw * ostrides3 + idz * ostrides2 + idy * ostrides1 + idx;

        out[oMem] = in[iMem2] - in[iMem1] - in[iMem1] + in[iMem0];
    }


    ///////////////////////////////////////////////////////////////////////////
    // Wrapper functions
    ///////////////////////////////////////////////////////////////////////////
    template<typename T>
    void diff1(T *out, const T *in, const unsigned dim,
               const unsigned oElem, const unsigned ondims, const dim_type *odims, const dim_type *ostrides,
               const unsigned iElem, const unsigned indims, const dim_type *idims, const dim_type *istrides)
    {
        dim3 threads(TX, TY, 1);
        unsigned blocksPerMatX = divup(odims[0], TX);
        unsigned blocksPerMatY = divup(odims[1], TY);
        dim3 blocks(blocksPerMatX * odims[2],
                    blocksPerMatY * odims[3],
                    1);

        if (indims == 1) {
            threads = dim3(TPB, 1, 1);
            blocks = dim3(divup(odims[0], TPB));
            diff1_1D<<<blocks, threads>>>(out, in, dim, oElem);
        } else if (indims == 2) {
            diff1_2D<<<blocks, threads>>>(out, in, dim, oElem, odims[0], ostrides[1], istrides[1]);
        } else if (indims == 3) {
            diff1_3D<<<blocks, threads>>>(out, in, dim, oElem, odims[0], odims[1],
                    ostrides[1], ostrides[2], istrides[1], istrides[2], blocksPerMatX);
        } else if (indims == 4) {
            diff1_4D<<<blocks, threads>>>(out, in, dim, oElem, odims[0], odims[1], odims[2],
                                          ostrides[1], ostrides[2], ostrides[3],
                                          istrides[1], istrides[2], istrides[3],
                                          blocksPerMatX, blocksPerMatY);
        } else {
            assert(1!=1);
        }
    }

    template<typename T>
    void diff2(T *out, const T *in, const unsigned dim,
               const unsigned oElem, const unsigned ondims, const dim_type *odims, const dim_type *ostrides,
               const unsigned iElem, const unsigned indims, const dim_type *idims, const dim_type *istrides)
    {
        dim3 threads(TX, TY, 1);
        unsigned blocksPerMatX = divup(odims[0], TX);
        unsigned blocksPerMatY = divup(odims[1], TY);
        dim3 blocks(blocksPerMatX * odims[2],
                    blocksPerMatY * odims[3],
                    1);

        if (indims == 1) {
            threads = dim3(TPB, 1, 1);
            blocks = dim3(divup(odims[0], TPB));
            diff2_1D<<<blocks, threads>>>(out, in, dim, oElem);
        } else if (indims == 2) {
            diff2_2D<<<blocks, threads>>>(out, in, dim, oElem, odims[0], ostrides[1], istrides[1]);
        } else if (indims == 3) {
            diff2_3D<<<blocks, threads>>>(out, in, dim, oElem, odims[0], odims[1],
                ostrides[1], ostrides[2], istrides[1], istrides[2], blocksPerMatX);
        } else if (indims == 4) {
            diff1_4D<<<blocks, threads>>>(out, in, dim, oElem, odims[0], odims[1], odims[2],
                                          ostrides[1], ostrides[2], ostrides[3],
                                          istrides[1], istrides[2], istrides[3],
                                          blocksPerMatX, blocksPerMatY);
        } else {
            assert(1!=1);
        }
    }

#define INSTANTIATE(T)                                                      \
    template void diff1<T> (T *out, const T *in, const unsigned dim,        \
               const unsigned oElem, const unsigned ondims, const dim_type *odims, const dim_type *ostrides,  \
               const unsigned iElem, const unsigned indims, const dim_type *idims, const dim_type *istrides); \
    template void diff2<T> (T *out, const T *in, const unsigned dim,        \
               const unsigned oElem, const unsigned ondims, const dim_type *odims, const dim_type *ostrides,  \
               const unsigned iElem, const unsigned indims, const dim_type *idims, const dim_type *istrides); \


    INSTANTIATE(float);
    INSTANTIATE(double);
    INSTANTIATE(cfloat);
    INSTANTIATE(cdouble);
    INSTANTIATE(int);
    INSTANTIATE(uint);
    INSTANTIATE(uchar);
    INSTANTIATE(char);
}
}

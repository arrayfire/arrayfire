#include <complex.hpp>
#include <kernel/diff.hpp>
#include <stdio.h>

#define divup(a, b) ((a+b-1)/b)

namespace cuda
{
    namespace kernel
    {
        typedef struct
        {
            dim_type dim[4];
        } dims_t;

        // Kernel Launch Config Values
        static const unsigned TX = 16;
        static const unsigned TY = 16;

        template<typename T, bool D>
        inline __host__ __device__
        void diff_this(T* out, const T* in, const unsigned oMem, const unsigned iMem0,
                       const unsigned iMem1, const unsigned iMem2)
        {
            //iMem2 can never be 0
            if(D == 0) {                        // Diff1
                out[oMem] = in[iMem1] - in[iMem0];
            } else {                                // Diff2
                out[oMem] = in[iMem2] - in[iMem1] - in[iMem1] + in[iMem0];
            }
        }

        /////////////////////////////////////////////////////////////////////////////
        // 1st and 2nd Order Differential for 4D along all dimensions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, unsigned dim, bool isDiff2>
        __global__
        void diff_kernel(T *out, const T *in,
                         const unsigned oElem, const dims_t odims,
                         const dims_t ostrides, const dims_t istrides,
                         const unsigned blocksPerMatX, const unsigned blocksPerMatY)
        {
            unsigned idz = blockIdx.x / blocksPerMatX;
            unsigned idw = blockIdx.y / blocksPerMatY;

            unsigned blockIdx_x = blockIdx.x - idz * blocksPerMatX;
            unsigned blockIdx_y = blockIdx.y - idw * blocksPerMatY;

            unsigned idx = threadIdx.x + blockIdx_x * blockDim.x;
            unsigned idy = threadIdx.y + blockIdx_y * blockDim.y;

            if(idx >= odims.dim[0] ||
               idy >= odims.dim[1] ||
               idz >= odims.dim[2] ||
               idw >= odims.dim[3])
                return;

            unsigned iMem0 = idw * istrides.dim[3] + idz * istrides.dim[2] + idy * istrides.dim[1] + idx;
            unsigned iMem1 = iMem0 + istrides.dim[dim];
            unsigned iMem2 = iMem1 + istrides.dim[dim];

            unsigned oMem = idw * ostrides.dim[3] + idz * ostrides.dim[2] + idy * ostrides.dim[1] + idx;

            iMem2 *= isDiff2;

            diff_this<T, isDiff2>(out, in, oMem, iMem0, iMem1, iMem2);
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, unsigned dim, bool isDiff2>
        void diff(T *out, const T *in,
                  const unsigned oElem, const unsigned ondims, const dim_type *odims, const dim_type *ostrides,
                  const unsigned iElem, const unsigned indims, const dim_type *idims, const dim_type *istrides)
        {
            dim3 threads(TX, TY, 1);
            unsigned blocksPerMatX = divup(odims[0], TX);
            unsigned blocksPerMatY = divup(odims[1], TY);
            dim3 blocks(blocksPerMatX * odims[2],
                        blocksPerMatY * odims[3],
                        1);

            dims_t _odims = {{odims[0], odims[1], odims[2], odims[3]}};
            dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
            dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};

            diff_kernel<T, dim, isDiff2><<<blocks, threads>>>(out, in, oElem, _odims,
                                                              _ostrides, _istrides,
                                                              blocksPerMatX, blocksPerMatY);
        }


#define INSTANTIATE_D(T, D)                                             \
        template void diff<T, D, false>(T *out, const T *in,            \
                                        const unsigned oElem, const unsigned ondims, const dim_type *odims, const dim_type *ostrides, \
                                        const unsigned iElem, const unsigned indims, const dim_type *idims, const dim_type *istrides); \
        template void diff<T, D, true >(T *out, const T *in,            \
                                        const unsigned oElem, const unsigned ondims, const dim_type *odims, const dim_type *ostrides, \
                                        const unsigned iElem, const unsigned indims, const dim_type *idims, const dim_type *istrides); \


#define INSTANTIATE(T)                          \
        INSTANTIATE_D(T, 0)                     \
        INSTANTIATE_D(T, 1)                     \
        INSTANTIATE_D(T, 2)                     \
        INSTANTIATE_D(T, 3)                     \

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

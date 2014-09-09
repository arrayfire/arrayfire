#include <stdio.h>
#include <dispatch.hpp>

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


        __constant__ float c_tmat[6 * 256];

        template <typename T>
        __host__ __device__
        void calc_affine_inverse(T *txo, const T *txi)
        {
            T det = txi[0]*txi[4] - txi[1]*txi[3];

            txo[0] = txi[4] / det;
            txo[1] = txi[3] / det;
            txo[3] = txi[1] / det;
            txo[4] = txi[0] / det;

            txo[2] = txi[2] * -txo[0] + txi[5] * -txo[1];
            txo[5] = txi[2] * -txo[3] + txi[5] * -txo[4];
        }

        ///////////////////////////////////////////////////////////////////////////
        // Transform Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, bool inverse>
        __global__ static void
        transform_kernel(T *out, const dim_type xo, const dim_type yo,
                         const T *in, const dim_type xi, const dim_type yi,
                         const dims_t ostrides, const dims_t istrides,
                         const dim_type nimages, const dim_type ntransforms)
        {
            // Get thread indices
            int xx = blockIdx.x * blockDim.x + threadIdx.x;
            int yy = blockIdx.y * blockDim.y + threadIdx.y;

            if(xx >= xo * nimages || yy >= yo * ntransforms)
                return;

            // Index of channel of images and transform
            int i_idx = xx / xo;
            int t_idx = yy / yo;

            // Index in local channel -> This is output index
            int xido = xx - i_idx * xo;
            int yido = yy - t_idx * yo;

            // Global offset
            //          Offset for transform channel + Offset for image channel.
            out += t_idx * nimages * ostrides.dim[2] + i_idx * ostrides.dim[2];
            in  += i_idx * istrides.dim[2];

            // Transform is in constant memory.
            const float *tmat_ptr = c_tmat + t_idx * 6;
            float tmat[6];

            // We expect a inverse transform matrix by default
            // If it is an forward transform, then we need its inverse
            if(inverse) {
                #pragma unroll
                for(int i = 0; i < 6; i++)
                    tmat[i] = tmat_ptr[i];
            } else {
                calc_affine_inverse(tmat, tmat_ptr);
            }

            if (xido >= xo && yido >= yo) return;

            // Compute input index
            const dim_type xidi = round(xido * tmat[0]
                                      + yido * tmat[1]
                                             + tmat[2]);
            const dim_type yidi = round(xido * tmat[3]
                                      + yido * tmat[4]
                                             + tmat[5]);

            // Compute memory location of indices
            dim_type loci = (yidi * istrides.dim[1] + xidi);
            dim_type loco = (yido * ostrides.dim[1] + xido);

            // Copy to output
            T val = 0;
            if (xidi < xi && yidi < yi && xidi >= 0 && yidi >= 0) val = in[loci];

            out[loco] = val;
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <class T>
        void transform(T *out, const T *in, const float *tf,
                       const dim_type *odims, const dim_type *idims,
                       const dim_type *ostrides, const dim_type *istrides,
                       const dim_type *tstrides, const bool inverse)
        {
            const dim_type nimages = idims[2];
            // Multiplied in src/backend/transform.cpp
            const dim_type ntransforms = odims[2] / idims[2];

            // Copy transform to constant memory.
            cudaMemcpyToSymbol(c_tmat, tf, ntransforms * 6 * sizeof(float), 0, cudaMemcpyDeviceToDevice);

            dim3 threads(TX, TY, 1);
            dim3 blocks(divup(odims[0], threads.x), divup(odims[1], threads.y));

            if (nimages > 1)     { blocks.x *= nimages;   }
            if (ntransforms > 1) { blocks.y *= ntransforms; }

            dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
            dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};

            if(inverse) {
                transform_kernel<T, true><<<blocks, threads>>>(out, odims[0], odims[1],
                                            in,  idims[0], idims[1], _ostrides, _istrides,
                                           nimages, ntransforms);
            } else {
                transform_kernel<T, false><<<blocks, threads>>>(out, odims[0], odims[1],
                                             in,  idims[0], idims[1], _ostrides, _istrides,
                                             nimages, ntransforms);
            }
        }
    }
}

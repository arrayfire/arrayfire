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


        ///////////////////////////////////////////////////////////////////////////
        // nearest-neighbor resampling
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        __host__ __device__
        void resize_n(      T* d_out, dim_type odim0, dim_type odim1,
                      const T* d_in,  dim_type idim0, dim_type idim1,
                      const dims_t ostrides, const dims_t istrides,
                      const unsigned blockIdx_x, const float xf, const float yf)
        {
            int const ox = threadIdx.x + blockIdx_x * blockDim.x;
            int const oy = threadIdx.y + blockIdx.y * blockDim.y;

            int ix = round(ox * xf);
            int iy = round(oy * yf);

            if (ox >= odim0 || oy >= odim1) { return; }
            if (ix >= idim0) { ix = idim0 - 1; }
            if (iy >= idim1) { iy = idim1 - 1; }

            d_out[ox + oy * ostrides.dim[1]] = d_in[ix + iy * istrides.dim[1]];
        }

        ///////////////////////////////////////////////////////////////////////////
        // bilinear resampling
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        __host__ __device__
        void resize_b(      T* d_out, dim_type odim0, dim_type odim1,
                      const T* d_in,  dim_type idim0, dim_type idim1,
                      const dims_t ostrides, const dims_t istrides,
                      const unsigned blockIdx_x, const float xf_, const float yf_)
        {
            int const ox = threadIdx.x + blockIdx_x * blockDim.x;
            int const oy = threadIdx.y + blockIdx.y * blockDim.y;

            float xf = ox * xf_;
            float yf = oy * yf_;

            int ix   = floorf(xf);
            int iy   = floorf(yf);

            if (ox >= odim0 || oy >= odim1) { return; }
            if (ix >= idim0) { ix = idim0 - 1; }
            if (iy >= idim1) { iy = idim1 - 1; }

            float b = xf - ix;
            float a = yf - iy;

            const int ix2 = ix + 1 <  idim0 ? ix + 1 : ix;
            const int iy2 = iy + 1 <  idim1 ? iy + 1 : iy;

            const T p1 = d_in[ix  + istrides.dim[1] * iy ];
            const T p2 = d_in[ix  + istrides.dim[1] * iy2];
            const T p3 = d_in[ix2 + istrides.dim[1] * iy ] ;
            const T p4 = d_in[ix2 + istrides.dim[1] * iy2];

            T out = (1.0f-a) * (1.0f-b) * p1 +
                    (a)      * (1.0f-b) * p2 +
                    (1.0f-a) * (b)      * p3 +
                    (a)      * (b)      * p4;

            d_out[ox + oy * ostrides.dim[1]] = out;
        }

        ///////////////////////////////////////////////////////////////////////////
        // Resize Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, af_interp_type method>
        __global__
        void resize_kernel(      T* d_out, dim_type odim0, dim_type odim1,
                           const T* d_in,  dim_type idim0, dim_type idim1,
                           const dims_t ostrides, const dims_t istrides,
                           const unsigned b0, const dim_type batch,
                           const float xf, const float yf)
        {
            unsigned id = blockIdx.x / b0;
            // channel adjustment
            int i_off = id * istrides.dim[2];
            int o_off = id * ostrides.dim[2];
            unsigned blockIdx_x =  blockIdx.x - id * b0;

            // core
            if(method == AF_INTERP_NEAREST) {
                resize_n(d_out+o_off, odim0, odim1,
                         d_in +i_off, idim0, idim1,
                         ostrides, istrides,
                         blockIdx_x, xf, yf);
            } else if(method == AF_INTERP_BILINEAR) {
                resize_b(d_out+o_off, odim0, odim1,
                         d_in +i_off, idim0, idim1,
                         ostrides, istrides,
                         blockIdx_x, xf, yf);
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename T, af_interp_type method>
        void resize(T *out, const dim_type odim0, const dim_type odim1,
              const T *in, const dim_type idim0, const dim_type idim1,
              const dim_type channels, const dim_type *ostrides, const dim_type *istrides)
        {
            dim3 threads(TX, TY, 1);
            dim3 blocks(divup(odim0, threads.x), divup(odim1, threads.y));
            unsigned blocksPerMatX = blocks.x;

            if (channels > 1) { blocks.x *= channels; }
            float xf = (float)idim0 / (float)odim0;
            float yf = (float)idim1 / (float)odim1;

            dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
            dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};

            resize_kernel<T, method><<<blocks, threads>>>(out, odim0, odim1,
                                          in,  idim0, idim1, _ostrides, _istrides,
                                          blocksPerMatX, channels, xf, yf);
        }

    }
}

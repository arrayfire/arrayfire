#include <dispatch.hpp>
#include <Param.hpp>

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const unsigned TX = 16;
        static const unsigned TY = 16;

        ///////////////////////////////////////////////////////////////////////////
        // nearest-neighbor resampling
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        __host__ __device__
        void resize_n(Param<T> out, CParam<T> in,
                      const dim_type o_off, const dim_type i_off,
                      const dim_type blockIdx_x, const float xf, const float yf)
        {
            const dim_type ox = threadIdx.x + blockIdx_x * blockDim.x;
            const dim_type oy = threadIdx.y + blockIdx.y * blockDim.y;

            dim_type ix = round(ox * xf);
            dim_type iy = round(oy * yf);

            if (ox >= out.dims[0] || oy >= out.dims[1]) { return; }
            if (ix >= in.dims[0]) { ix = in.dims[0] - 1; }
            if (iy >= in.dims[1]) { iy = in.dims[1] - 1; }

            out.ptr[o_off + ox + oy * out.strides[1]] = in.ptr[i_off + ix + iy * in.strides[1]];
        }

        ///////////////////////////////////////////////////////////////////////////
        // bilinear resampling
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        __host__ __device__
        void resize_b(Param<T> out, CParam<T> in,
                      const dim_type o_off, const dim_type i_off,
                      const dim_type blockIdx_x, const float xf_, const float yf_)
        {
            const dim_type ox = threadIdx.x + blockIdx_x * blockDim.x;
            const dim_type oy = threadIdx.y + blockIdx.y * blockDim.y;

            float xf = ox * xf_;
            float yf = oy * yf_;

            dim_type ix = floorf(xf);
            dim_type iy = floorf(yf);

            if (ox >= out.dims[0] || oy >= out.dims[1]) { return; }
            if (ix >= in.dims[0]) { ix = in.dims[0] - 1; }
            if (iy >= in.dims[1]) { iy = in.dims[1] - 1; }

            float b = xf - ix;
            float a = yf - iy;

            const dim_type ix2 = ix + 1 <  in.dims[0] ? ix + 1 : ix;
            const dim_type iy2 = iy + 1 <  in.dims[1] ? iy + 1 : iy;

            const T *iptr = in.ptr + i_off;

            const T p1 = iptr[ix  + in.strides[1] * iy ];
            const T p2 = iptr[ix  + in.strides[1] * iy2];
            const T p3 = iptr[ix2 + in.strides[1] * iy ] ;
            const T p4 = iptr[ix2 + in.strides[1] * iy2];

            T val = (1.0f-a) * (1.0f-b) * p1 +
                    (a)      * (1.0f-b) * p2 +
                    (1.0f-a) * (b)      * p3 +
                    (a)      * (b)      * p4;

            out.ptr[o_off + ox + oy * out.strides[1]] = val;
        }

        ///////////////////////////////////////////////////////////////////////////
        // Resize Kernel
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, af_interp_type method>
        __global__
        void resize_kernel(Param<T> out, CParam<T> in,
                           const dim_type b0, const float xf, const float yf)
        {
            const dim_type id = blockIdx.x / b0;
            // channel adjustment
            const dim_type i_off = id * in.strides[2];
            const dim_type o_off = id * out.strides[2];
            const dim_type blockIdx_x =  blockIdx.x - id * b0;

            // core
            if(method == AF_INTERP_NEAREST) {
                resize_n(out, in, o_off, i_off, blockIdx_x, xf, yf);
            } else if(method == AF_INTERP_BILINEAR) {
                resize_b(out, in, o_off, i_off, blockIdx_x, xf, yf);
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template <typename T, af_interp_type method>
        void resize(Param<T> out, CParam<T> in)
        {
            dim3 threads(TX, TY, 1);
            dim3 blocks(divup(out.dims[0], threads.x), divup(out.dims[1], threads.y));
            dim_type blocksPerMatX = blocks.x;

            if (in.dims[2] > 1) { blocks.x *= in.dims[2]; }
            float xf = (float)in.dims[0] / (float)out.dims[0];
            float yf = (float)in.dims[1] / (float)out.dims[1];

            resize_kernel<T, method><<<blocks, threads>>>(out, in, blocksPerMatX, xf, yf);
        }

    }
}

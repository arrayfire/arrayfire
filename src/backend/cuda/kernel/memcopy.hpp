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
#include <backend.hpp>
#include <Param.hpp>
#include <debug_cuda.hpp>

namespace cuda
{
namespace kernel
{

    typedef struct
    {
        int dim[4];
    } dims_t;

    static const uint DIMX = 32;
    static const uint DIMY =  8;

    template<typename T>
    __global__ static void
    memcopy_kernel(T *out, const dims_t ostrides,
                   const T *in, const dims_t idims,
                   const dims_t istrides, uint blocks_x, uint blocks_y)
    {
        const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

        const int zid = blockIdx.x / blocks_x;
        const int wid = blockIdx.y / blocks_y;
        const int blockIdx_x = blockIdx.x - (blocks_x) * zid;
        const int blockIdx_y = blockIdx.y - (blocks_y) * wid;
        const int xid = blockIdx_x * blockDim.x + tidx;
        const int yid = blockIdx_y * blockDim.y + tidy;

        // FIXME: Do more work per block
        out += wid * ostrides.dim[3] + zid * ostrides.dim[2] + yid * ostrides.dim[1];
        in  += wid * istrides.dim[3] + zid * istrides.dim[2] + yid * istrides.dim[1];

        int istride0 = istrides.dim[0];
        if (xid < idims.dim[0] &&
            yid < idims.dim[1] &&
            zid < idims.dim[2] &&
            wid < idims.dim[3]) {
            out[xid] = in[xid * istride0];
        }

    }

    template<typename T>
    void memcopy(T *out, const dim_t *ostrides,
                 const T *in, const dim_t *idims,
                 const dim_t *istrides, uint ndims)
    {
        dim3 threads(DIMX, DIMY);

        if (ndims == 1) {
            threads.x *= threads.y;
            threads.y  = 1;
       }

        // FIXME: DO more work per block
        uint blocks_x = divup(idims[0], threads.x);
        uint blocks_y = divup(idims[1], threads.y);

        dim3 blocks(blocks_x * idims[2],
                    blocks_y * idims[3]);

        dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
        dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};
        dims_t _idims = {{idims[0], idims[1], idims[2], idims[3]}};

        CUDA_LAUNCH((memcopy_kernel<T>), blocks, threads,
                out, _ostrides, in, _idims, _istrides, blocks_x, blocks_y);
        POST_LAUNCH_CHECK();
    }


    ////////////////////////////// BEGIN - templated help functions for copy_kernel ////////////////////////////////
    template<typename T>
    __inline__ __device__ static
    T scale(T value, double factor) {
        return (T)(value*factor);
    }

    template<>
    __inline__ __device__
    cfloat scale<cfloat>(cfloat value, double factor) {
        return make_cuFloatComplex(value.x*factor, value.y*factor);
    }

    template<>
    __inline__ __device__
    cdouble scale<cdouble>(cdouble value, double factor) {
        return make_cuDoubleComplex(value.x*factor, value.y*factor);
    }

    template<typename inType, typename outType>
    __inline__ __device__
    outType convertType(inType value) {
        return (outType)value;
    }

    template<>
    __inline__ __device__
    cdouble convertType<cfloat, cdouble>(cfloat value) {
        return cuComplexFloatToDouble(value);
    }

    template<>
    __inline__ __device__
    cfloat convertType<cdouble, cfloat>(cdouble value) {
        return cuComplexDoubleToFloat(value);
    }

#define OTHER_SPECIALIZATIONS(IN_T)                     \
    template<>                                          \
    __inline__ __device__                               \
    cfloat convertType<IN_T, cfloat>(IN_T value) {      \
        return make_cuFloatComplex(value, 0.0f);        \
    }                                                   \
                                                        \
    template<>                                          \
    __inline__ __device__                               \
    cdouble convertType<IN_T, cdouble>(IN_T value) {    \
        return make_cuDoubleComplex(value, 0.0);        \
    }

    OTHER_SPECIALIZATIONS(float )
    OTHER_SPECIALIZATIONS(double)
    OTHER_SPECIALIZATIONS(int   )
    OTHER_SPECIALIZATIONS(uint  )
    OTHER_SPECIALIZATIONS(intl   )
    OTHER_SPECIALIZATIONS(uintl  )
    OTHER_SPECIALIZATIONS(uchar )
    OTHER_SPECIALIZATIONS(char  )
    ////////////////////////////// END - templated help functions for copy_kernel //////////////////////////////////


    template<typename inType, typename outType, bool same_dims>
    __global__ static void
    copy_kernel(Param<outType> dst, CParam<inType> src, outType default_value,
                double factor, const dims_t trgt, uint blk_x, uint blk_y)
    {
        const uint lx = threadIdx.x;
        const uint ly = threadIdx.y;

        const uint gz = blockIdx.x / blk_x;
        const uint gw = blockIdx.y / blk_y;
        const uint blockIdx_x = blockIdx.x - (blk_x) * gz;
        const uint blockIdx_y = blockIdx.y - (blk_y) * gw;
        const uint gx = blockIdx_x * blockDim.x + lx;
        const uint gy = blockIdx_y * blockDim.y + ly;

        const inType * in = src.ptr + (gw * src.strides[3] + gz * src.strides[2] + gy * src.strides[1]);
        outType * out     = dst.ptr + (gw * dst.strides[3] + gz * dst.strides[2] + gy * dst.strides[1]);

        int istride0 = src.strides[0];
        int ostride0 = dst.strides[0];

        if (gy < dst.dims[1] && gz < dst.dims[2] && gw < dst.dims[3]) {
            int loop_offset = blockDim.x * blk_x;
            bool cond = gy < trgt.dim[1] && gz < trgt.dim[2] && gw < trgt.dim[3];
            for(int rep=gx; rep<dst.dims[0]; rep+=loop_offset) {
                outType temp = default_value;
                if (same_dims || (rep < trgt.dim[0] && cond)) {
                    temp = convertType<inType, outType>(scale<inType>(in[rep * istride0], factor));
                }
                out[rep*ostride0] = temp;
            }
        }
    }

    template<typename inType, typename outType>
    void copy(Param<outType> dst, CParam<inType> src, int ndims, outType default_value, double factor)
    {
        dim3 threads(DIMX, DIMY);
        size_t local_size[] = {DIMX, DIMY};

        //FIXME: Why isn't threads being updated??
        local_size[0] *= local_size[1];
        if (ndims == 1) {
            local_size[1] = 1;
        }

        uint blk_x = divup(dst.dims[0], local_size[0]);
        uint blk_y = divup(dst.dims[1], local_size[1]);

        dim3 blocks(blk_x * dst.dims[2],
                    blk_y * dst.dims[3]);

        int trgt_l  = std::min(dst.dims[3], src.dims[3]);
        int trgt_k  = std::min(dst.dims[2], src.dims[2]);
        int trgt_j  = std::min(dst.dims[1], src.dims[1]);
        int trgt_i  = std::min(dst.dims[0], src.dims[0]);
        dims_t trgt_dims = {{trgt_i, trgt_j, trgt_k, trgt_l}};

        bool same_dims = ( (src.dims[0]==dst.dims[0]) &&
                           (src.dims[1]==dst.dims[1]) &&
                           (src.dims[2]==dst.dims[2]) &&
                           (src.dims[3]==dst.dims[3]) );

        if (same_dims)
            CUDA_LAUNCH((copy_kernel<inType, outType, true >), blocks, threads,
                    dst, src, default_value, factor, trgt_dims, blk_x, blk_y);
        else
            CUDA_LAUNCH((copy_kernel<inType, outType, false>), blocks, threads,
                    dst, src, default_value, factor, trgt_dims, blk_x, blk_y);

        POST_LAUNCH_CHECK();
    }

}
}

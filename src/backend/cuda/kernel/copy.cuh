/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/half.hpp>
#include <dims_param.hpp>
#include <types.hpp>

namespace cuda {

template<typename T>
__inline__ __device__ static T scale(T value, double factor) {
    return (T)(double(value) * factor);
}

template<>
__inline__ __device__ cfloat scale<cfloat>(cfloat value, double factor) {
    return make_cuFloatComplex(value.x * factor, value.y * factor);
}

template<>
__inline__ __device__ cdouble scale<cdouble>(cdouble value, double factor) {
    return make_cuDoubleComplex(value.x * factor, value.y * factor);
}

template<typename inType, typename outType>
__inline__ __device__ outType convertType(inType value) {
    return static_cast<outType>(value);
}

template<>
__inline__ __device__ char convertType<compute_t<common::half>, char>(
    compute_t<common::half> value) {
    return (char)((short)value);
}

template<>
__inline__ __device__ compute_t<common::half>
convertType<char, compute_t<common::half>>(char value) {
    return compute_t<common::half>(value);
}

template<>
__inline__ __device__ cuda::uchar
convertType<compute_t<common::half>, cuda::uchar>(
    compute_t<common::half> value) {
    return (cuda::uchar)((short)value);
}

template<>
__inline__ __device__ compute_t<common::half>
convertType<cuda::uchar, compute_t<common::half>>(cuda::uchar value) {
    return compute_t<common::half>(value);
}

template<>
__inline__ __device__ cdouble convertType<cfloat, cdouble>(cfloat value) {
    return cuComplexFloatToDouble(value);
}

template<>
__inline__ __device__ cfloat convertType<cdouble, cfloat>(cdouble value) {
    return cuComplexDoubleToFloat(value);
}

#define OTHER_SPECIALIZATIONS(IN_T)                                        \
    template<>                                                             \
    __inline__ __device__ cfloat convertType<IN_T, cfloat>(IN_T value) {   \
        return make_cuFloatComplex(static_cast<float>(value), 0.0f);       \
    }                                                                      \
                                                                           \
    template<>                                                             \
    __inline__ __device__ cdouble convertType<IN_T, cdouble>(IN_T value) { \
        return make_cuDoubleComplex(static_cast<double>(value), 0.0);      \
    }

OTHER_SPECIALIZATIONS(float)
OTHER_SPECIALIZATIONS(double)
OTHER_SPECIALIZATIONS(int)
OTHER_SPECIALIZATIONS(uint)
OTHER_SPECIALIZATIONS(intl)
OTHER_SPECIALIZATIONS(uintl)
OTHER_SPECIALIZATIONS(short)
OTHER_SPECIALIZATIONS(ushort)
OTHER_SPECIALIZATIONS(uchar)
OTHER_SPECIALIZATIONS(char)
OTHER_SPECIALIZATIONS(common::half)

template<typename inType, typename outType, bool same_dims>
__global__ void copy(Param<outType> dst, CParam<inType> src,
                     outType default_value, double factor, const dims_t trgt,
                     uint blk_x, uint blk_y) {
    const uint lx = threadIdx.x;
    const uint ly = threadIdx.y;

    const uint gz         = blockIdx.x / blk_x;
    const uint gw         = (blockIdx.y + (blockIdx.z * gridDim.y)) / blk_y;
    const uint blockIdx_x = blockIdx.x - (blk_x)*gz;
    const uint blockIdx_y =
        (blockIdx.y + (blockIdx.z * gridDim.y)) - (blk_y)*gw;
    const uint gx = blockIdx_x * blockDim.x + lx;
    const uint gy = blockIdx_y * blockDim.y + ly;

    const inType *in = src.ptr + (gw * src.strides[3] + gz * src.strides[2] +
                                  gy * src.strides[1]);
    outType *out     = dst.ptr + (gw * dst.strides[3] + gz * dst.strides[2] +
                              gy * dst.strides[1]);

    int istride0 = src.strides[0];
    int ostride0 = dst.strides[0];

    if (gy < dst.dims[1] && gz < dst.dims[2] && gw < dst.dims[3]) {
        int loop_offset = blockDim.x * blk_x;
        bool cond = gy < trgt.dim[1] && gz < trgt.dim[2] && gw < trgt.dim[3];
        for (int rep = gx; rep < dst.dims[0]; rep += loop_offset) {
            outType temp = default_value;
            if (same_dims || (rep < trgt.dim[0] && cond)) {
                temp = convertType<inType, outType>(
                    scale<inType>(in[rep * istride0], factor));
            }
            out[rep * ostride0] = temp;
        }
    }
}

}  // namespace cuda

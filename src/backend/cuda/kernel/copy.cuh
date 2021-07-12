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

template<typename inType, typename outType, bool SAME_DIMS, bool LOOP1,
         bool LOOP2, bool FACTOR>
__global__ void reshapeCopy(Param<outType> dst, CParam<inType> src,
                            outType default_value, double factor) {
    const int g0 = blockIdx.x * blockDim.x + threadIdx.x;
    int g1       = blockIdx.y * blockDim.y + threadIdx.y;

    const bool inside_dst = (g0 < (int)dst.dims[0]) && (g1 < (int)dst.dims[1]);
    if (inside_dst) {
        int g2           = blockIdx.z * blockDim.z + threadIdx.z;
        const int idims1 = src.dims[1];
        bool inside_src =
            SAME_DIMS || ((g0 < (int)src.dims[0]) && (g1 < idims1));
        const int idims2    = src.dims[2];
        const int idims3    = src.dims[3];
        const int istrides2 = src.strides[2];
        const int istrides3 = src.strides[3];
        src.ptr += g0 * (int)src.strides[0] + g1 * (int)src.strides[1];
        const int odims3    = dst.dims[3];
        const int ostrides2 = dst.strides[2];
        const int ostrides3 = dst.strides[3];
        dst.ptr += g0 * (int)dst.strides[0] + g1 * (int)dst.strides[1];
#if LOOP1
        const int oinc1  = gridDim.y * (int)dst.strides[1];
        const int odims1 = dst.dims[1];
        const int iinc1  = gridDim.y * (int)src.strides[1];
        do {
#endif

#if LOOP2
            do {
#endif
                int ioffset           = g2 * istrides2;
                int ooffset           = g2 * ostrides2;
                const int ooffsetEnd1 = ooffset + idims3 * ostrides3;
                if (SAME_DIMS) {
                    do {
                        outType val = convertType<inType, outType>(
                            FACTOR ? scale<inType>(src.ptr[ioffset], factor)
                                   : src.ptr[ioffset]);
                        ioffset += istrides3;
                        dst.ptr[ooffset] = val;
                        ooffset += ostrides3;
                    } while (ooffset != ooffsetEnd1);
                } else {
                    bool inside = SAME_DIMS || (inside_src && (g2 < idims2));
                    const int ooffsetEnd2 = ooffset + odims3 * ostrides3;
                    if (inside) {
                        do {
                            outType val = convertType<inType, outType>(
                                FACTOR ? scale<inType>(src.ptr[ioffset], factor)
                                       : src.ptr[ioffset]);
                            ioffset += istrides3;
                            dst.ptr[ooffset] = val;
                            ooffset += ostrides3;
                        } while (ooffset != ooffsetEnd1);
                    }
                    while (ooffset != ooffsetEnd2) {
                        dst.ptr[ooffset] = default_value;
                        ooffset += ostrides3;
                    }
                }

#if LOOP2
                g2 += gridDim.z;
            } while (g2 < (int)dst.dims[2]);
            g2 = blockIdx.z * blockDim.z + threadIdx.z;
#endif

#if LOOP1
            g1 += gridDim.y;
            src.ptr += iinc1;
            dst.ptr += oinc1;
            inside_src &= (SAME_DIMS || (g1 < idims1));
        } while (g1 < odims1);
#endif
    }
}

}  // namespace cuda

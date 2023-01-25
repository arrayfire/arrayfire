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

namespace arrayfire {
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
__inline__ __device__ uchar
convertType<compute_t<common::half>, uchar>(compute_t<common::half> value) {
    return (uchar)((short)value);
}

template<>
__inline__ __device__ compute_t<common::half>
convertType<uchar, compute_t<common::half>>(uchar value) {
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

// scaledCopy without looping, so dim3 has to be 1.
// conditions:
//      global dims[0] >= dims[0]
//      global dims[1] >= dims[1]
//      global dims[2] == dims[2]
//      only dims[3] == 1 will be processed!!
template<typename inType, typename outType, bool SAME_DIMS, bool FACTOR>
__global__ void scaledCopy(Param<outType> dst, CParam<inType> src,
                           const outType default_value, const double factor) {
    const int id0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int id1 = blockIdx.y * blockDim.y + threadIdx.y;
    if ((id0 < (int)dst.dims[0]) & (id1 < (int)dst.dims[1])) {
        const int id2 = blockIdx.z * blockDim.z + threadIdx.z;

        const int idx_in =
            id0 * src.strides[0] + id1 * src.strides[1] + id2 * src.strides[2];
        const int idx_out =
            id0 * dst.strides[0] + id1 * dst.strides[1] + id2 * dst.strides[2];

        if (SAME_DIMS | ((id0 < (int)src.dims[0]) & (id1 < (int)src.dims[1]) &
                         (id2 < (int)src.dims[2]))) {
            dst.ptr[idx_out] = convertType<inType, outType>(
                FACTOR ? scale<inType>(src.ptr[idx_in], factor)
                       : src.ptr[idx_in]);
        } else {
            dst.ptr[idx_out] = default_value;
        }
    }
}

// scaledCopy with looping over dims[0] -- VECTOR ONLY
// Conditions:
//      global dims[0] has no restrictions
//      only dims[1] == 1 will be processed!!
//      only dims[2] == 1 will be processed!!
//      only dims[3] == 1 will be processed!!
template<typename inType, typename outType, bool SAME_DIMS, bool FACTOR>
__global__ void scaledCopyLoop0(Param<outType> dst, CParam<inType> src,
                                const outType default_value,
                                const double factor) {
    int id0              = blockIdx.x * blockDim.x + threadIdx.x;
    const int id0End_out = dst.dims[0];
    if (id0 < id0End_out) {
        const int id0End_in     = src.dims[0];
        const int istrides0     = src.strides[0];
        const int ostrides0     = dst.strides[0];
        const int id0Inc        = gridDim.x * blockDim.x;
        int idx_in              = id0 * istrides0;
        const int idxID0Inc_in  = id0Inc * istrides0;
        int idx_out             = id0 * ostrides0;
        const int idxID0Inc_out = id0Inc * ostrides0;

        while (id0 < id0End_in) {
            // inside input array, so convert
            dst.ptr[idx_out] = convertType<inType, outType>(
                FACTOR ? scale<inType>(src.ptr[idx_in], factor)
                       : src.ptr[idx_in]);
            id0 += id0Inc;
            idx_in += idxID0Inc_in;
            idx_out += idxID0Inc_out;
        }
        if (!SAME_DIMS) {
            while (id0 < id0End_out) {
                // outside the input array, so copy default value
                dst.ptr[idx_out] = default_value;
                id0 += id0Inc;
                idx_out += idxID0Inc_out;
            }
        }
    }
}

// scaledCopy with looping over dims[1]
// Conditions:
//      global dims[0] >= dims[0]
//      global dims[1] has no restrictions
//      global dims[2] == dims[2]
//      only dims[3] == 1 will be processed!!
template<typename inType, typename outType, bool SAME_DIMS, bool FACTOR>
__global__ void scaledCopyLoop1(Param<outType> dst, CParam<inType> src,
                                const outType default_value,
                                const double factor) {
    const int id0        = blockIdx.x * blockDim.x + threadIdx.x;
    int id1              = blockIdx.y * blockDim.y + threadIdx.y;
    const int id1End_out = dst.dims[1];
    if ((id0 < (int)dst.dims[0]) & (id1 < id1End_out)) {
        const int id2       = blockIdx.z * blockDim.z + threadIdx.z;
        const int ostrides1 = dst.strides[1];
        const int id1Inc    = gridDim.y * blockDim.y;
        int idx_out         = id0 * (int)dst.strides[0] + id1 * ostrides1 +
                      id2 * (int)dst.strides[2];
        const int idxID1Inc_out = id1Inc * ostrides1;
        const int id1End_in     = src.dims[1];
        const int istrides1     = src.strides[1];
        int idx_in              = id0 * (int)src.strides[0] + id1 * istrides1 +
                     id2 * (int)src.strides[2];
        const int idxID1Inc_in = id1Inc * istrides1;

        if (SAME_DIMS | ((id0 < (int)src.dims[0]) & (id2 < src.dims[2]))) {
            while (id1 < id1End_in) {
                // inside input array, so convert
                dst.ptr[idx_out] = convertType<inType, outType>(
                    FACTOR ? scale<inType>(src.ptr[idx_in], factor)
                           : src.ptr[idx_in]);
                id1 += id1Inc;
                idx_in += idxID1Inc_in;
                idx_out += idxID1Inc_out;
            }
        }
        if (!SAME_DIMS) {
            while (id1 < id1End_out) {
                // outside the input array, so copy default value
                dst.ptr[idx_out] = default_value;
                id1 += id1Inc;
                idx_out += idxID1Inc_out;
            }
        }
    }
}

// scaledCopy with looping over dims[1], dims[2] and dims[3]
// Conditions:
//      global dims[0] >= dims[0]
//      global dims[1] has no restrictions
//      global dims[2] <= dims[2]
template<typename inType, typename outType, bool SAME_DIMS, bool FACTOR>
__global__ void scaledCopyLoop123(Param<outType> out, CParam<inType> in,
                                  outType default_value, double factor) {
    const int id0    = blockIdx.x * blockDim.x + threadIdx.x;  // Limit 2G
    int id1          = blockIdx.y * blockDim.y + threadIdx.y;  // Limit 64K
    const int odims0 = out.dims[0];
    const int odims1 = out.dims[1];
    if ((id0 < odims0) & (id1 < odims1)) {
        int id2 = blockIdx.z * blockDim.z + threadIdx.z;  // Limit 64K
        int idxBaseBase_out = id0 * (int)out.strides[0] +
                              id1 * (int)out.strides[1] +
                              id2 * (int)out.strides[2];
        const int idxIncID3_out     = out.strides[3];
        const int odims2            = out.dims[2];
        const int idxEndIncID3_out  = out.dims[3] * idxIncID3_out;
        const int incID1            = gridDim.y * blockDim.y;
        const int idxBaseIncID1_out = incID1 * (int)out.strides[1];
        const int incID2            = gridDim.z * blockDim.z;
        const int idxBaseIncID2_out = incID2 * (int)out.strides[2];

        int idxBaseBase_in = id0 * (int)in.strides[0] +
                             id1 * (int)in.strides[1] +
                             id2 * (int)in.strides[2];
        const int idxIncID3_in     = in.strides[3];
        const int idims0           = in.dims[0];
        const int idims1           = in.dims[1];
        const int idims2           = in.dims[2];
        const int idxEndIncID3_in  = in.dims[3] * idxIncID3_in;
        const int idxBaseIncID1_in = incID1 * (int)in.strides[1];
        const int idxBaseIncID2_in = incID2 * (int)in.strides[2];

        do {
            int idxBase_in  = idxBaseBase_in;
            int idxBase_out = idxBaseBase_out;
            do {
                int idxEndID3_in  = idxEndIncID3_in + idxBase_in;
                int idxEndID3_out = idxEndIncID3_out + idxBase_out;
                int idx_in        = idxBase_in;
                int idx_out       = idxBase_out;
                if (SAME_DIMS |
                    ((id0 < idims0) & (id1 < idims1) & (id2 < idims2))) {
                    // inside input array, so convert
                    do {
                        out.ptr[idx_out] = convertType<inType, outType>(
                            FACTOR ? scale<inType>(in.ptr[idx_in], factor)
                                   : in.ptr[idx_in]);
                        idx_in += idxIncID3_in;
                        idx_out += idxIncID3_out;
                    } while (idx_in != idxEndID3_in);
                }
                if (!SAME_DIMS) {
                    while (idx_out != idxEndID3_out) {
                        // outside the input array, so copy default value
                        out.ptr[idx_out] = default_value;
                        idx_out += idxIncID3_out;
                    }
                }
                id1 += incID1;
                if (id1 >= odims1) break;
                idxBase_in += idxBaseIncID1_in;
                idxBase_out += idxBaseIncID1_out;
            } while (true);
            id2 += incID2;
            if (id2 >= odims2) break;
            idxBaseBase_in += idxBaseIncID2_in;
            idxBaseBase_out += idxBaseIncID2_out;
        } while (true);
    }
}

}  // namespace cuda
}  // namespace arrayfire

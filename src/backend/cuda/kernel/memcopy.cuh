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

namespace arrayfire {
namespace cuda {

// memCopy without looping, so dim3 has to be 1.
// conditions:
//      kernel dims[0] >= dims[0]
//      kernel dims[1] >= dims[1]
//      kernel dims[2] == dims[2]
//      only dims[3] == 1 will be processed!!
template<typename T>
__global__ void memCopy(Param<T> out, CParam<T> in) {
    const int id0 = blockIdx.x * blockDim.x + threadIdx.x;  // Limit 2G
    const int id1 = blockIdx.y * blockDim.y + threadIdx.y;  // Limit 64K
    if ((id0 < (int)in.dims[0]) & (id1 < (int)in.dims[1])) {
        const int id2 = blockIdx.z * blockDim.z + threadIdx.z;  // Limit 64K

        out.ptr[id0 * (int)out.strides[0] + id1 * (int)out.strides[1] +
                id2 * (int)out.strides[2]] =
            in.ptr[id0 * (int)in.strides[0] + id1 * (int)in.strides[1] +
                   id2 * (int)in.strides[2]];
    }
}

// memCopy with looping over dims[0] -- VECTOR ONLY
// Conditions:
//      kernel dims[0] has no restrictions
//      only dims[1] == 1 will be processed!!
//      only dims[2] == 1 will be procesed!!
//      only dims[3] == 1 will be processed!!
template<typename T>
__global__ void memCopyLoop0(Param<T> out, CParam<T> in) {
    int id0          = blockIdx.x * blockDim.x + threadIdx.x;  // Limit 2G
    const int idims0 = in.dims[0];
    if (id0 < idims0) {
        const int incID0        = gridDim.x * blockDim.x;
        const int istrides0     = in.strides[0];
        int idx_in              = id0 * istrides0;
        const int idxIncID0_in  = incID0 * istrides0;
        const int ostrides0     = out.strides[0];
        int idx_out             = id0 * ostrides0;
        const int idxIncID0_out = incID0 * ostrides0;

        do {
            out.ptr[idx_out] = in.ptr[idx_in];
            id0 += incID0;
            if (id0 >= idims0) break;
            idx_in += idxIncID0_in;
            idx_out += idxIncID0_out;
        } while (true);
    }
}

// memCopy with looping over dims[1]
// Conditions:
//      kernel dims[0] >= dims[0]
//      kernel dims[1] has no restrictions
//      kernel dims[2] == dims[2]
//      only dims[3] == 1 will be processed!!
template<typename T>
__global__ void memCopyLoop1(Param<T> out, CParam<T> in) {
    const int id0    = blockIdx.x * blockDim.x + threadIdx.x;  // Limit 2G
    int id1          = blockIdx.y * blockDim.y + threadIdx.y;  // Limit 64K
    const int idims1 = in.dims[1];
    if ((id0 < (int)in.dims[0]) & (id1 < idims1)) {
        const int id2 = blockIdx.z * blockDim.z + threadIdx.z;  // Limit 64K
        const int istrides1 = in.strides[1];
        int idx_in          = id0 * (int)in.strides[0] + id1 * istrides1 +
                     id2 * (int)in.strides[2];
        const int incID1       = gridDim.y * blockDim.y;
        const int idxIncID1_in = incID1 * istrides1;
        const int ostrides1    = out.strides[1];
        int idx_out            = id0 * (int)out.strides[0] + id1 * ostrides1 +
                      id2 * (int)out.strides[2];
        const int idxIncID1_out = incID1 * ostrides1;

        do {
            out.ptr[idx_out] = in.ptr[idx_in];
            id1 += incID1;
            if (id1 >= idims1) break;
            idx_in += idxIncID1_in;
            idx_out += idxIncID1_out;
        } while (true);
    }
}

// memCopy with looping over dims[3]
// Conditions:
//      kernel dims[0] >= dims[0]
//      kernel dims[1] >= dims[1]
//      kernel dims[2] == dims[2]
template<typename T>
__global__ void memCopyLoop3(Param<T> out, CParam<T> in) {
    const int id0 = blockIdx.x * blockDim.x + threadIdx.x;  // Limit 2G
    const int id1 = blockIdx.y * blockDim.y + threadIdx.y;  // Limit 64K
    if ((id0 < (int)in.dims[0]) & (id1 < (int)in.dims[1])) {
        const int id2 = blockIdx.z * blockDim.z + threadIdx.z;  // Limit 64K
        int idx_in    = id0 * (int)in.strides[0] + id1 * (int)in.strides[1] +
                     id2 * (int)in.strides[2];
        const int idxIncID3_in = in.strides[3];
        const int idxEnd_in    = (int)in.dims[3] * idxIncID3_in + idx_in;
        int idx_out = id0 * (int)out.strides[0] + id1 * (int)out.strides[1] +
                      id2 * (int)out.strides[2];
        const int idxIncID3_out = out.strides[3];

        do {
            out.ptr[idx_out] = in.ptr[idx_in];
            idx_in += idxIncID3_in;
            if (idx_in == idxEnd_in) break;
            idx_out += idxIncID3_out;
        } while (true);
    }
}

// memCopy with looping over dims[1] and dims[3]
// Conditions:
//      kernel dims[0] >= dims[0]
//      kernel dims[1] has no restrictions
//      kernel dims[2] == dims[2]
template<typename T>
__global__ void memCopyLoop13(Param<T> out, CParam<T> in) {
    const int id0    = blockIdx.x * blockDim.x + threadIdx.x;  // Limit 2G
    int id1          = blockIdx.y * blockDim.y + threadIdx.y;  // Limit 64K
    const int idims1 = in.dims[1];
    if ((id0 < (int)in.dims[0]) & (g1 < idims1)) {
        const int id2 = blockIdx.z * blockDim.z + threadIdx.z;  // Limit 64K
        const int istrides1 = in.strides[1];
        int idxBase_in      = id0 * (int)in.strides[0] + id1 * istrides1 +
                         id2 * (int)in.strides[2];
        const int incID1           = gridDim.y * blockDim.y;
        const int idxBaseIncID1_in = incID1 * istrides1;
        const int idxIncID3_in     = (int)in.strides[3];
        int idxEndID3_in = (int)in.dims[3] * idxIncID3_in + idxBase_in;
        int idxBase_out  = id0 * (int)out.strides[0] +
                          id1 * (int)out.strides[1] + id2 * (int)out.strides[2];
        const int idxBaseIncID1_out = incID1 * (int)out.strides[1];
        const int idxIncID3_out     = (int)out.strides[3];

        do {
            int idx_in  = idxBase_in;
            int idx_out = idxBase_out;
            while (true) {
                out.ptr[idx_out] = in.ptr[idx_in];
                idx_in += idxIncID3_in;
                if (idx_in == idxEndID3_in) break;
                idx_out += idxIncID3_out;
            }
            id1 += incID1;
            if (id1 >= idims1) break;
            idxBase_in += idxBaseIncID1_in;
            idxEndID3_in += idxBaseIncID1_in;
            idxBase_out += idxBaseIncID1_out;
        } while (true);
    }
}

// memCopy with looping over dims[1],dims[2] and dims[3]
// Conditions:
//      kernel dims[0] >= dims[0]
//      kernel dims[1] has no restrictions
//      kernel dims[2] <= dims[2]
template<typename T>
__global__ void memCopyLoop123(Param<T> out, CParam<T> in) {
    const int id0    = blockIdx.x * blockDim.x + threadIdx.x;  // Limit 2G
    int id1          = blockIdx.y * blockDim.y + threadIdx.y;  // Limit 64K
    const int idims1 = in.dims[1];
    if ((id0 < (int)in.dims[0]) & (id1 < idims1)) {
        int id2 = blockIdx.z * blockDim.z + threadIdx.z;  // Limit 64K
        const int istrides1 = in.strides[1];
        const int istrides2 = in.strides[2];
        int idxBaseBase_in =
            id0 * (int)in.strides[0] + id1 * istrides1 + id2 * istrides2;
        const int incID1           = gridDim.y * blockDim.y;
        const int idxBaseIncID1_in = incID1 * istrides1;
        const int incID2           = gridDim.z * blockDim.z;
        const int idxBaseIncID2_in = incID2 * istrides2;
        const int idxIncID3_in     = in.strides[3];
        const int idxEndIncID3_in  = (int)in.dims[3] * idxIncID3_in;

        const int ostrides1 = out.strides[1];
        const int ostrides2 = out.strides[2];
        int idxBaseBase_out =
            id0 * (int)out.strides[0] + id1 * ostrides1 + id2 * ostrides2;
        const int idxBaseIncID1_out = incID1 * ostrides1;
        const int idxBaseIncID2_out = incID2 * ostrides2;
        const int idxIncID3_out     = out.strides[3];
        const int idims2            = in.dims[2];

        do {
            int idxBase_in  = idxBaseBase_in;
            int idxBase_out = idxBaseBase_out;
            do {
                int idxEndID3_in = idxEndIncID3_in + idxBase_in;
                int idx_in       = idxBase_in;
                int idx_out      = idxBase_out;
                do {
                    out.ptr[idx_out] = in.ptr[idx_in];
                    idx_in += idxIncID3_in;
                    if (idx_in == idxEndID3_in) break;
                    idx_out += idxIncID3_out;
                } while (true);
                id1 += incID1;
                if (id1 >= idims1) break;
                idxBase_in += idxBaseIncID1_in;
                idxBase_out += idxBaseIncID1_out;
            } while (true);
            id2 += incID2;
            if (id2 >= idims2) break;
            idxBaseBase_in += idxBaseIncID2_in;
            idxBaseBase_out += idxBaseIncID2_out;
        } while (true);
    }
}
}  // namespace cuda
}  // namespace arrayfire

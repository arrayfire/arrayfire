/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

void diff_this(__global T* out, __global const T* in, const int oMem,
               const int iMem0, const int iMem1, const int iMem2)
{
    if(isDiff2 == 0) {
        out[oMem] = in[iMem1] - in[iMem0];
    } else {
        out[oMem] = in[iMem2] - in[iMem1] - in[iMem1] + in[iMem0];
    }
}

__kernel
void diff_kernel(__global T *out, __global const T *in,
                 const KParam op, const KParam ip, const int oElem,
                 const int blocksPerMatX, const int blocksPerMatY)
{
    const int idz = get_group_id(0) / blocksPerMatX;
    const int idw = get_group_id(1) / blocksPerMatY;

    const int blockIdx_x = get_group_id(0) - idz * blocksPerMatX;
    const int blockIdx_y = get_group_id(1) - idw * blocksPerMatY;

    const int idx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const int idy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if(idx >= op.dims[0] ||
       idy >= op.dims[1] ||
       idz >= op.dims[2] ||
       idw >= op.dims[3])
        return;

    int iMem0 = idw * ip.strides[3] + idz * ip.strides[2] + idy * ip.strides[1] + idx;
    int iMem1 = iMem0 + ip.strides[DIM];
    int iMem2 = iMem1 + ip.strides[DIM];

    int oMem = idw * op.strides[3] + idz * op.strides[2] + idy * op.strides[1] + idx;

    iMem2 *= isDiff2;

    diff_this(out, in + ip.offset, oMem, iMem0, iMem1, iMem2);
}

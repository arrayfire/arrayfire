/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if CPLX
#define set(a, b) a = b
#define set_scalar(a, b) do {                   \
        a.x = b;                                \
        a.y = 0;                                \
    } while(0)

#else

#define set(a, b) a = b
#define set_scalar(a, b) a = b

#endif

#define sidx(y, x) scratch[((y + 1) * (TX + 2)) + (x + 1)]

__kernel
void gradient_kernel(__global T *d_grad0, const KParam grad0,
                     __global T *d_grad1, const KParam grad1,
                     __global const T* d_in, const KParam in,
                     const int blocksPerMatX, const int blocksPerMatY)
{
    const int idz = get_group_id(0) / blocksPerMatX;
    const int idw = get_group_id(1) / blocksPerMatY;

    const int blockIdx_x = get_group_id(0) - idz * blocksPerMatX;
    const int blockIdx_y = get_group_id(1) - idw * blocksPerMatY;

    const int xB = blockIdx_x * get_local_size(0);
    const int yB = blockIdx_y * get_local_size(1);

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    const int idx = tx + xB;
    const int idy = ty + yB;

    const bool cond = (idx >= in.dims[0] || idy >= in.dims[1] ||
                       idz >= in.dims[2] || idw >= in.dims[3]);

    int xmax = (TX > (in.dims[0] - xB)) ? (in.dims[0] - xB) : TX;
    int ymax = (TY > (in.dims[1] - yB)) ? (in.dims[1] - yB) : TY;

    int iIdx = in.offset + idw * in.strides[3] + idz * in.strides[2]
                              + idy * in.strides[1] + idx;

    int g0dx = idw * grad0.strides[3] + idz * grad0.strides[2]
                  + idy * grad0.strides[1] + idx;

    int g1dx = idw * grad1.strides[3] + idz * grad1.strides[2]
                  + idy * grad1.strides[1] + idx;

    __local T scratch[(TY + 2) * (TX + 2)];

    // Multipliers - 0.5 for interior, 1 for edge cases
    float xf = 0.5 * (1 + (idx == 0 || idx >= (in.dims[0] - 1)));
    float yf = 0.5 * (1 + (idy == 0 || idy >= (in.dims[1] - 1)));

    // Copy data to scratch space
    T zero = ZERO;
    if(cond) {
        sidx(ty, tx) = zero;
    } else {
        sidx(ty, tx) = d_in[iIdx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Copy buffer zone data. Corner (0,0) etc, are not used.
    // Cols
    if(ty == 0) {
        // Y-1
        sidx(-1, tx) = (cond || idy == 0) ?
                        sidx(0, tx) : d_in[iIdx - in.strides[1]];
        sidx(ymax, tx) = (cond || (idy + ymax) >= in.dims[1]) ?
                          sidx(ymax - 1, tx) : d_in[iIdx + ymax * in.strides[1]];
    }
    // Rows
    if(tx == 0) {
        sidx(ty, -1) = (cond || idx == 0) ?
                        sidx(ty, 0) : d_in[iIdx - 1];
        sidx(ty, xmax) = (cond || (idx + xmax) >= in.dims[0]) ?
                          sidx(ty, xmax - 1) : d_in[iIdx + xmax];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (cond) return;

    //set_scalar(d_grad0[iIdx], sidx(ty, tx));
    d_grad0[g0dx] = xf * (sidx(ty, tx + 1) -  sidx(ty, tx - 1));
    d_grad1[g1dx] = yf * (sidx(ty + 1, tx) -  sidx(ty - 1, tx));
}

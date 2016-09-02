/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define NEAREST core_nearest2
#define LINEAR core_linear2
#define CUBIC core_cubic2

#if CPLX
#define set(a, b) a = b
#define set_scalar(a, b) do {                   \
        a.x = b;                                \
        a.y = 0;                                \
    } while(0)

Ty mul(Ty a, Tp b) { a.x = a.x * b; a.y = a.y * b; return a; }
Ty div(Ty a, Tp b) { a.x = a.x / b; a.y = a.y / b; return a; }


#else

#define set(a, b) a = b
#define set_scalar(a, b) a = b
#define mul(a, b) ((a) * (b))
#define div(a, b) ((a) / (b))

#endif

///////////////////////////////////////////////////////////////////////////
// nearest-neighbor resampling
///////////////////////////////////////////////////////////////////////////
void core_nearest2(const int idx, const int idy, const int idz, const int idw,
                   __global       Ty *d_out, const KParam out,
                   __global const Ty *d_in,  const KParam in,
                   __global const Tp *d_pos, const KParam pos,
                   __global const Tp *d_qos, const KParam qos,
                   const float offGrid, const bool pBatch)
{
    const int omId = idw * out.strides[3] + idz * out.strides[2]
                     + idy * out.strides[1] + idx;
    int pmId = idy * pos.strides[1] + idx;
    int qmId = idy * qos.strides[1] + idx;
    if(pBatch) {
        pmId += idw * pos.strides[3] + idz * pos.strides[2];
        qmId += idw * qos.strides[3] + idz * qos.strides[2];
    }

    const Tp x = d_pos[pmId], y = d_qos[qmId];
    if (x < 0 || y < 0 || in.dims[0] < x+1 || in.dims[1] < y+1) {
        set_scalar(d_out[omId], offGrid);
        return;
    }

    const int grid_x = round(x), grid_y = round(y); // nearest grid
    const int imId = idw * in.strides[3] + idz * in.strides[2]
                  + grid_y * in.strides[1] + grid_x;

    Ty z;
    set(z, d_in[imId]);
    set(d_out[omId], z);
}

///////////////////////////////////////////////////////////////////////////
// linear resampling
///////////////////////////////////////////////////////////////////////////
void core_linear2(const int idx, const int idy, const int idz, const int idw,
                  __global       Ty *d_out, const KParam out,
                  __global const Ty *d_in,  const KParam in,
                  __global const Tp *d_pos, const KParam pos,
                  __global const Tp *d_qos, const KParam qos,
                  const float offGrid, const bool pBatch)
{
    const int omId = idw * out.strides[3] + idz * out.strides[2]
                     + idy * out.strides[1] + idx;
    int pmId = idy * pos.strides[1] + idx;
    int qmId = idy * qos.strides[1] + idx;
    if(pBatch) {
        pmId += idw * pos.strides[3] + idz * pos.strides[2];
        qmId += idw * qos.strides[3] + idz * qos.strides[2];
    }

    const Tp x = d_pos[pmId], y = d_qos[qmId];
    if (x < 0 || y < 0 || in.dims[0] < x+1 || in.dims[1] < y+1) {
        set_scalar(d_out[omId], offGrid);
        return;
    }

    const int grid_x = floor(x),   grid_y = floor(y);   // nearest grid
    const Tp off_x  = x - grid_x, off_y  = y - grid_y; // fractional offset

    int ioff = idw * in.strides[3] + idz * in.strides[2] + grid_y * in.strides[1] + grid_x;

    // Check if pVal and pVal + 1 are both valid indices
    bool condY = (y < in.dims[1] - 1);
    bool condX = (x < in.dims[0] - 1);

    // Compute wieghts used
    Tp wt00 = ((Tp)1.0 - off_x) * ((Tp)1.0 - off_y);
    Tp wt10 = (condY) ? ((Tp)1.0 - off_x) * (off_y) : 0;
    Tp wt01 = (condX) ? (off_x) * ((Tp)1.0 - off_y) : 0;
    Tp wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

    Tp wt = wt00 + wt10 + wt01 + wt11;

    // Compute Weighted Values
    Ty zero = ZERO;
    Ty y00 =                    mul(d_in[ioff],                     wt00)       ;
    Ty y10 = (condY) ?          mul(d_in[ioff + in.strides[1]],     wt10) : zero;
    Ty y01 = (condX) ?          mul(d_in[ioff + 1],                 wt01) : zero;
    Ty y11 = (condX && condY) ? mul(d_in[ioff + in.strides[1] + 1], wt11) : zero;

    Ty yo = y00 + y10 + y01 + y11;

    // Write Final Value
    set(d_out[omId], div(yo, wt));
}

///////////////////////////////////////////////////////////////////////////
// cubic resampling
///////////////////////////////////////////////////////////////////////////

Ty cubicInterpolate(Ty p[4], Tp x) {
    return p[1] + (Ty)0.5 * x * (p[2] - p[0] + x * ((Ty)2.0 * p[0] - (Ty)5.0 * p[1] + (Ty)4.0 * p[2] - p[3] + x*((Ty)3.0*(p[1] - p[2]) + p[3] - p[0])));
}

Ty bicubicInterpolate(Ty p[4][4], Tp x, Tp y) {
    Ty arr[4];
    arr[0] = cubicInterpolate(p[0], x);
    arr[1] = cubicInterpolate(p[1], x);
    arr[2] = cubicInterpolate(p[2], x);
    arr[3] = cubicInterpolate(p[3], x);
    return cubicInterpolate(arr, y);
}

void core_cubic2(const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw,
                  __global       Ty *d_out, const KParam out,
                  __global const Ty *d_in,  const KParam in,
                  __global const Tp *d_pos, const KParam pos,
                  __global const Tp *d_qos, const KParam qos,
                  const float offGrid, const bool pBatch)
{
    const dim_t omId = idw * out.strides[3] + idz * out.strides[2]
                     + idy * out.strides[1] + idx;
    dim_t pmId = idy * pos.strides[1] + idx;
    dim_t qmId = idy * qos.strides[1] + idx;
    if(pBatch) {
        pmId += idw * pos.strides[3] + idz * pos.strides[2];
        qmId += idw * qos.strides[3] + idz * qos.strides[2];
    }

    const Tp x = d_pos[pmId], y = d_qos[qmId];
    if (x < 0 || y < 0 || in.dims[0] < x+1 || in.dims[1] < y+1) {
        set_scalar(d_out[omId], offGrid);
        return;
    }

    const dim_t grid_x = floor(x),   grid_y = floor(y);   // nearest grid
    const Tp off_x  = x - grid_x, off_y  = y - grid_y; // fractional offset

    dim_t ioff = idw * in.strides[3] + idz * in.strides[2] + grid_y * in.strides[1] + grid_x;
    // used for setting values at boundaries
    bool condXl = (grid_x < 1);
    bool condYl = (grid_y < 1);
    bool condXg = (grid_x > in.dims[0] - 3);
    bool condYg = (grid_y > in.dims[1] - 3);

    //for bicubic interpolation, work with 4x4 patch at a time
    Ty patch[4][4];

    //assumption is that inner patch consisting of 4 points is minimum requirement for bicubic interpolation
    //inner square
    patch[1][1] = d_in[ioff];
    patch[1][2] = d_in[ioff + 1];
    patch[2][1] = d_in[ioff + in.strides[1]];
    patch[2][2] = d_in[ioff + in.strides[1] + 1];
    //outer sides
    patch[0][1] = (condYl)? d_in[ioff]     : d_in[ioff - in.strides[1]];
    patch[0][2] = (condYl)? d_in[ioff + 1] : d_in[ioff - in.strides[1] + 1];
    patch[3][1] = (condYg)? d_in[ioff + in.strides[1]]     : d_in[ioff + 2 * in.strides[1]];
    patch[3][2] = (condYg)? d_in[ioff + in.strides[1] + 1] : d_in[ioff + 2 * in.strides[1] + 1];
    patch[1][0] = (condXl)? d_in[ioff] : d_in[ioff - 1];
    patch[2][0] = (condXl)? d_in[ioff + in.strides[1]] : d_in[ioff + in.strides[1] - 1];
    patch[1][3] = (condXg)? d_in[ioff + 1] : d_in[ioff + 2];
    patch[2][3] = (condXg)? d_in[ioff + in.strides[1] + 1] : d_in[ioff + in.strides[1] + 2];
    //corners
    patch[0][0] = (condXl || condYl)? d_in[ioff] : d_in[ioff - in.strides[1] - 1]    ;
    patch[0][3] = (condYl || condXg)? d_in[ioff + 1] : d_in[ioff - in.strides[1] + 1]    ;
    patch[3][0] = (condXl || condYg)? d_in[ioff + in.strides[1]] : d_in[ioff + 2 * in.strides[1] - 1];
    patch[3][3] = (condXg || condYg)? d_in[ioff + in.strides[1] + 1] : d_in[ioff + 2 * in.strides[1] + 2];

    set(d_out[omId], bicubicInterpolate(patch, off_x, off_y));
}

////////////////////////////////////////////////////////////////////////////////////
// Wrapper Kernel
////////////////////////////////////////////////////////////////////////////////////
__kernel
void approx2_kernel(__global       Ty *d_out, const KParam out,
                    __global const Ty *d_in,  const KParam in,
                    __global const Tp *d_pos, const KParam pos,
                    __global const Tp *d_qos, const KParam qos,
                    const float offGrid, const int blocksMatX, const int blocksMatY,
                    const int pBatch)
{
    const int idz = get_group_id(0) / blocksMatX;
    const int idw = get_group_id(1) / blocksMatY;

    const int blockIdx_x = get_group_id(0) - idz * blocksMatX;
    const int blockIdx_y = get_group_id(1) - idw * blocksMatY;

    const int idx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const int idy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if(idx >= out.dims[0] ||
       idy >= out.dims[1] ||
       idz >= out.dims[2] ||
       idw >= out.dims[3])
        return;

    INTERP(idx, idy, idz, idw, d_out, out, d_in + in.offset, in,
           d_pos + pos.offset, pos, d_qos + qos.offset, qos, offGrid, pBatch);
}

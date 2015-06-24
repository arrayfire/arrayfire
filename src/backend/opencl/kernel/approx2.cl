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
                   const float offGrid)
{
    const int omId = idw * out.strides[3] + idz * out.strides[2]
                        + idy * out.strides[1] + idx;
    const int pmId = idy * pos.strides[1] + idx;
    const int qmId = idy * qos.strides[1] + idx;

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
                  const float offGrid)
{
    const int omId = idw * out.strides[3] + idz * out.strides[2]
                        + idy * out.strides[1] + idx;
    const int pmId = idy * pos.strides[1] + idx;
    const int qmId = idy * qos.strides[1] + idx;

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
    Ty zero; set_scalar(zero, 0);
    Ty y00; set(y00,                    mul(d_in[ioff],                     wt00)       );
    Ty y10; set(y10, (condY) ?          mul(d_in[ioff + in.strides[1]],     wt10) : zero);
    Ty y01; set(y01, (condX) ?          mul(d_in[ioff + 1],                 wt01) : zero);
    Ty y11; set(y11, (condX && condY) ? mul(d_in[ioff + in.strides[1] + 1], wt11) : zero);

    Ty yo = y00 + y10 + y01 + y11;

    // Write Final Value
    set(d_out[omId], div(yo, wt));
}

////////////////////////////////////////////////////////////////////////////////////
// Wrapper Kernel
////////////////////////////////////////////////////////////////////////////////////
__kernel
void approx2_kernel(__global       Ty *d_out, const KParam out,
                    __global const Ty *d_in,  const KParam in,
                    __global const Tp *d_pos, const KParam pos,
                    __global const Tp *d_qos, const KParam qos,
                    const float offGrid, const int blocksMatX, const int blocksMatY)
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
           d_pos + pos.offset, pos, d_qos + qos.offset, qos, offGrid);
}

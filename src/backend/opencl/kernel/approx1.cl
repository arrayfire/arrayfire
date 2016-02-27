/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define NEAREST core_nearest1
#define LINEAR core_linear1
#define CUBIC core_cubic

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
void core_nearest1(const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw,
                   __global       Ty *d_out, const KParam out,
                   __global const Ty *d_in,  const KParam in,
                   __global const Tp *d_pos, const KParam pos,
                   const float offGrid, const bool pBatch)
{
    const dim_t omId = idw * out.strides[3] + idz * out.strides[2]
                     + idy * out.strides[1] + idx;
    dim_t pmId = idx;
    if(pBatch) pmId += idw * pos.strides[3] + idz * pos.strides[2] + idy * pos.strides[1];

    const Tp pVal = d_pos[pmId];
    if (pVal < 0 || in.dims[0] < pVal+1) {
        set_scalar(d_out[omId], offGrid);
        return;
    }

    dim_t ioff = idw * in.strides[3] + idz * in.strides[2] + idy * in.strides[1];
    const dim_t imId = round(pVal) + ioff;

    Ty y;
    set(y, d_in[imId]);
    set(d_out[omId], y);
}

///////////////////////////////////////////////////////////////////////////
// linear resampling
///////////////////////////////////////////////////////////////////////////
void core_linear1(const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw,
                   __global       Ty *d_out, const KParam out,
                   __global const Ty *d_in,  const KParam in,
                   __global const Tp *d_pos, const KParam pos,
                   const float offGrid, const bool pBatch)
{
    const dim_t omId = idw * out.strides[3] + idz * out.strides[2]
                     + idy * out.strides[1] + idx;
    dim_t pmId = idx;
    if(pBatch) pmId += idw * pos.strides[3] + idz * pos.strides[2] + idy * pos.strides[1];

    const Tp pVal = d_pos[pmId];
    if (pVal < 0 || in.dims[0] < pVal+1) {
        set_scalar(d_out[omId], offGrid);
        return;
    }

    const dim_t grid_x = floor(pVal);  // nearest grid
    const Tp off_x = pVal - grid_x; // fractional offset

    dim_t ioff = idw * in.strides[3] + idz * in.strides[2] + idy * in.strides[1] + grid_x;

    // Check if pVal and pVal + 1 are both valid indices
    bool cond = (pVal < in.dims[0] - 1);

    Ty zero = ZERO;

    // Compute Left and Right Weighted Values
    Ty yl = mul(d_in[ioff] , (1 - off_x));
    Ty yr = cond ? mul(d_in[ioff + 1], off_x) : zero;
    Ty yo = yl + yr;

    // Compute Weight used
    Tp wt = cond ? 1 : (1 - off_x);

    // Write final value
    set(d_out[omId], div(yo, wt));
}

///////////////////////////////////////////////////////////////////////////
// cubic resampling
///////////////////////////////////////////////////////////////////////////
void core_cubic(const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw,
                   __global       Ty *d_out, const KParam out,
                   __global const Ty *d_in,  const KParam in,
                   __global const Tp *d_pos, const KParam pos,
                   const float offGrid, const bool pBatch)
{
    const dim_t omId = idw * out.strides[3] + idz * out.strides[2]
                     + idy * out.strides[1] + idx;
    dim_t pmId = idx;
    if(pBatch) pmId += idw * pos.strides[3] + idz * pos.strides[2] + idy * pos.strides[1];

    const Tp pVal = d_pos[pmId];
    if (pVal < 0 || in.dims[0] < pVal+1) {
        set_scalar(d_out[omId], offGrid);
        return;
    }

    const dim_t grid_x = floor(pVal);  // nearest grid
    const Tp off_x = pVal - grid_x; // fractional offset

    dim_t ioff = idw * in.strides[3] + idz * in.strides[2] + idy * in.strides[1] + grid_x;

    // Check if x-1, x, and x+1, x+2 are both valid indices
    bool condl1 = (pVal < in.dims[0] - 1);
    bool condl2 = (pVal < in.dims[0] - 2);
    bool condr = (pVal > 0);

    //compute basis function values
    Ty h00 = (1 + 2 * off_x) * (1 - off_x)*(1 - off_x);
    Ty h10 = off_x * (1 - off_x)*(1 - off_x);
    Ty h01 = off_x * off_x * (3 - 2 * off_x);
    Ty h11 = off_x * off_x * (off_x - 1);

    Ty zero = ZERO;

    // Compute Left and Right Weighted Values
    Ty pl = condr ? d_in[ioff] : offGrid;
    Ty pr = condl1 ? d_in[ioff + 1] : offGrid;
    Ty tl = condr ? 0.5 * ((d_in[ioff + 1] -d_in[ioff]) + (d_in[ioff] -d_in[ioff - 1])) :
                    0.5 * ((d_in[ioff + 1] -d_in[ioff]) + (d_in[ioff] - offGrid));
    Ty tr = condl2 ? 0.5 * ((d_in[ioff + 2] -d_in[ioff + 1]) + (d_in[ioff + 1] - d_in[ioff])) :
            condl1 ? 0.5 * ((offGrid - d_in[ioff + 1]) + d_in[ioff + 1] - d_in[ioff]) :
                     0.5 * ((offGrid - d_in[ioff]));

    // Write final value
    set(d_out[omId], h00 * pl + h10 * tl + h01 * pr + h11 * tr);
}

////////////////////////////////////////////////////////////////////////////////////
// Wrapper Kernel
////////////////////////////////////////////////////////////////////////////////////
__kernel
void approx1_kernel(__global       Ty *d_out, const KParam out,
                    __global const Ty *d_in,  const KParam in,
                    __global const Tp *d_pos, const KParam pos,
                    const float offGrid, const dim_t blocksMatX, const int pBatch)
{
    const dim_t idw = get_group_id(1) / out.dims[2];
    const dim_t idz = get_group_id(1)  - idw * out.dims[2];

    const dim_t idy = get_group_id(0) / blocksMatX;
    const dim_t blockIdx_x = get_group_id(0) - idy * blocksMatX;
    const dim_t idx = get_local_id(0) + blockIdx_x * get_local_size(0);

    if(idx >= out.dims[0] ||
       idy >= out.dims[1] ||
       idz >= out.dims[2] ||
       idw >= out.dims[3])
        return;

    INTERP(idx, idy, idz, idw, d_out, out, d_in + in.offset, in, d_pos + pos.offset, pos, offGrid, pBatch);
}

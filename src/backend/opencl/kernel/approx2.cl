#if Tp == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

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

typedef struct {
    dim_type dim[4];
} dims_t;

///////////////////////////////////////////////////////////////////////////
// nearest-neighbor resampling
///////////////////////////////////////////////////////////////////////////
void core_nearest2(const dim_type idx, const dim_type idy, const dim_type idz, const dim_type idw,
                   __global       Ty *d_out, const dims_t odims, const dim_type oElems,
                   __global const Ty *d_in,  const dims_t idims, const dim_type iElems,
                   __global const Tp *d_pos, const dims_t pdims,
                   __global const Tp *d_qos, const dims_t qdims,
                   const dims_t ostrides, const dims_t istrides,
                   const dims_t pstrides, const dims_t qstrides, const float offGrid)
{
    const dim_type omId = idw * ostrides.dim[3] + idz * ostrides.dim[2]
                        + idy * ostrides.dim[1] + idx;
    const dim_type pmId = idy * pstrides.dim[1] + idx;
    const dim_type qmId = idy * qstrides.dim[1] + idx;

    const Tp x = d_pos[pmId], y = d_qos[qmId];
    if (x < 0 || y < 0 || idims.dim[0] < x+1 || idims.dim[1] < y+1) {
        set_scalar(d_out[omId], offGrid);
        return;
    }

    const dim_type grid_x = round(x), grid_y = round(y); // nearest grid
    const dim_type imId = idw * istrides.dim[3] + idz * istrides.dim[2]
                        + grid_y * istrides.dim[1] + grid_x;

    Ty z;
    set(z, d_in[imId]);
    set(d_out[omId], z);
}

///////////////////////////////////////////////////////////////////////////
// linear resampling
///////////////////////////////////////////////////////////////////////////
void core_linear2(const dim_type idx, const dim_type idy, const dim_type idz, const dim_type idw,
                  __global       Ty *d_out, const dims_t odims, const dim_type oElems,
                  __global const Ty *d_in,  const dims_t idims, const dim_type iElems,
                  __global const Tp *d_pos, const dims_t pdims,
                  __global const Tp *d_qos, const dims_t qdims,
                  const dims_t ostrides, const dims_t istrides,
                  const dims_t pstrides, const dims_t qstrides, const float offGrid)
{
    const dim_type omId = idw * ostrides.dim[3] + idz * ostrides.dim[2]
                        + idy * ostrides.dim[1] + idx;
    const dim_type pmId = idy * pstrides.dim[1] + idx;
    const dim_type qmId = idy * qstrides.dim[1] + idx;

    const Tp x = d_pos[pmId], y = d_qos[qmId];
    if (x < 0 || y < 0 || idims.dim[0] < x+1 || idims.dim[1] < y+1) {
        set_scalar(d_out[omId], offGrid);
        return;
    }

    const Tp grid_x = floor(x),   grid_y = floor(y);   // nearest grid
    const Tp off_x  = x - grid_x, off_y  = y - grid_y; // fractional offset

    dim_type ioff = idw * istrides.dim[3] + idz * istrides.dim[2] + grid_y * istrides.dim[1] + grid_x;

    // Check if pVal and pVal + 1 are both valid indices
    bool condY = (y < idims.dim[1] - 1);
    bool condX = (x < idims.dim[0] - 1);

    // Compute wieghts used
    Tp wt00 = ((Tp)1.0 - off_x) * ((Tp)1.0 - off_y);
    Tp wt10 = (condY) ? ((Tp)1.0 - off_x) * (off_y) : 0;
    Tp wt01 = (condX) ? (off_x) * ((Tp)1.0 - off_y) : 0;
    Tp wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

    Tp wt = wt00 + wt10 + wt01 + wt11;

    // Compute Weighted Values
    Ty zero; set_scalar(zero, 0);
    Ty y00; set(y00,                    mul(d_in[ioff],                       wt00)       );
    Ty y10; set(y10, (condY) ?          mul(d_in[ioff + istrides.dim[1]],     wt10) : zero);
    Ty y01; set(y01, (condX) ?          mul(d_in[ioff + 1],                   wt01) : zero);
    Ty y11; set(y11, (condX && condY) ? mul(d_in[ioff + istrides.dim[1] + 1], wt11) : zero);

    Ty yo = y00 + y10 + y01 + y11;

    // Write Final Value
    set(d_out[omId], div(yo, wt));
}

////////////////////////////////////////////////////////////////////////////////////
// Wrapper Kernel
////////////////////////////////////////////////////////////////////////////////////
__kernel
void approx2_kernel(__global       Ty *d_out, const dims_t odims, const dim_type oElems,
                    __global const Ty *d_in,  const dims_t idims, const dim_type iElems,
                    __global const Tp *d_pos, const dims_t pdims,
                    __global const Tp *d_qos, const dims_t qdims,
                    const dims_t ostrides, const dims_t istrides,
                    const dims_t pstrides, const dims_t qstrides,
                    const float offGrid, const dim_type blocksMatX, const dim_type blocksMatY,
                    const dim_type iOffset, const dim_type pOffset, const dim_type qOffset)
{
    const dim_type idz = get_group_id(0) / blocksMatX;
    const dim_type idw = get_group_id(1) / blocksMatY;

    const dim_type blockIdx_x = get_group_id(0) - idz * blocksMatX;
    const dim_type blockIdx_y = get_group_id(1) - idw * blocksMatY;

    const dim_type idx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const dim_type idy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if(idx >= odims.dim[0] ||
       idy >= odims.dim[1] ||
       idz >= odims.dim[2] ||
       idw >= odims.dim[3])
        return;

    INTERP(idx, idy, idz, idw, d_out, odims, oElems, d_in + iOffset, idims, iElems,
            d_pos + pOffset, pdims, d_qos + qOffset, qdims,
            ostrides, istrides, pstrides, qstrides, offGrid);
}

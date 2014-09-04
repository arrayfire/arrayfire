#if Tp == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define NEAREST core_nearest1
#define LINEAR core_linear1

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
void core_nearest1(const dim_type idx, const dim_type idy, const dim_type idz, const dim_type idw,
                   __global       Ty *d_out, const dims_t odims, const dim_type oElems,
                   __global const Ty *d_in,  const dims_t idims, const dim_type iElems,
                   __global const Tp *d_pos, const dims_t pdims,
                   const dims_t ostrides, const dims_t istrides,
                   const dims_t pstrides, const float offGrid)
{
    const dim_type omId = idw * ostrides.dim[3] + idz * ostrides.dim[2]
                        + idy * ostrides.dim[1] + idx;
    const dim_type pmId = idx;

    const Tp x = d_pos[pmId];
    if (x < 0 || idims.dim[0] < x+1) {
        set_scalar(d_out[omId], offGrid);
        return;
    }

    dim_type ioff = idw * istrides.dim[3] + idz * istrides.dim[2] + idy * istrides.dim[1];
    const dim_type imId = round(x) + ioff;

    Ty y;
    set(y, d_in[imId]);
    set(d_out[omId], y);
}

///////////////////////////////////////////////////////////////////////////
// linear resampling
///////////////////////////////////////////////////////////////////////////
void core_linear1(const dim_type idx, const dim_type idy, const dim_type idz, const dim_type idw,
                  __global       Ty *d_out, const dims_t odims, const dim_type oElems,
                  __global const Ty *d_in,  const dims_t idims, const dim_type iElems,
                  __global const Tp *d_pos, const dims_t pdims,
                  const dims_t ostrides, const dims_t istrides,
                  const dims_t pstrides, const float offGrid)
{
    const dim_type omId = idw * ostrides.dim[3] + idz * ostrides.dim[2]
                        + idy * ostrides.dim[1] + idx;
    const dim_type pmId = idx;

    const Tp pVal = d_pos[pmId];
    if (pVal < 0 || idims.dim[0] < pVal+1) {
        set_scalar(d_out[omId], offGrid);
        return;
    }

    const Tp grid_x = floor(pVal);  // nearest grid
    const Tp off_x = pVal - grid_x; // fractional offset

    Tp w = 0;
    Ty y; set_scalar(y, 0);
    dim_type ioff = idw * istrides.dim[3] + idz * istrides.dim[2] + idy * istrides.dim[1];
    for(dim_type xx = 0; xx <= (pVal < idims.dim[0] - 1); ++xx) {
        Tp fxx = (Tp)(xx);
        Tp wx = 1 - fabs(off_x - fxx);
        dim_type imId = (dim_type)(fxx + grid_x) + ioff;
        Ty yt; set(yt, d_in[imId]);
        y = y + mul(yt, wx);
        w = w + wx;
    }
    set(d_out[omId], div(y, w));
}

////////////////////////////////////////////////////////////////////////////////////
// Wrapper Kernel
////////////////////////////////////////////////////////////////////////////////////
__kernel
void approx1_kernel(__global Ty *d_out, const dims_t odims, const dim_type oElems,
                    __global const Ty *d_in, const dims_t idims, const dim_type iElems,
                    __global const Tp *d_pos, const dims_t pdims,
                    const dims_t ostrides, const dims_t istrides, const dims_t pstrides,
                    const float offGrid, const dim_type blocksMatX,
                    const dim_type iOffset, const dim_type pOffset)
{
    const dim_type idw = get_group_id(1) / odims.dim[2];
    const dim_type idz = get_group_id(1)  - idw * odims.dim[2];

    const dim_type idy = get_group_id(0) / blocksMatX;
    const dim_type blockIdx_x = get_group_id(0) - idy * blocksMatX;
    const dim_type idx = get_local_id(0) + blockIdx_x * get_local_size(0);

    if(idx >= odims.dim[0] ||
       idy >= odims.dim[1] ||
       idz >= odims.dim[2] ||
       idw >= odims.dim[3])
        return;

    INTERP(idx, idy, idz, idw, d_out, odims, oElems, d_in + iOffset, idims, iElems,
           d_pos + pOffset, pdims, ostrides, istrides, pstrides, offGrid);
}

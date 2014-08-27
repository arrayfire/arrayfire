#if T == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

typedef struct {
    dim_type dim[4];
} dims_t;

void diff_this(__global T* out, __global const T* in, unsigned oMem,
               unsigned iMem0, unsigned iMem1, unsigned iMem2)
{
    if(isDiff2 == 0) {
        out[oMem] = in[iMem1] - in[iMem0];
    } else {
        out[oMem] = in[iMem2] - in[iMem1] - in[iMem1] + in[iMem0];
    }
}

__kernel
void diff_kernel(__global T *out, __global const T *in,
                 const unsigned oElem, const dims_t odims,
                 const dims_t ostrides, const dims_t istrides,
                 dim_type offset,
                 const unsigned blocksPerMatX, const unsigned blocksPerMatY)
{
    unsigned idz = get_group_id(0) / blocksPerMatX;
    unsigned idw = get_group_id(1) / blocksPerMatY;

    unsigned blockIdx_x = get_group_id(0) - idz * blocksPerMatX;
    unsigned blockIdx_y = get_group_id(1) - idw * blocksPerMatY;

    unsigned idx = get_local_id(0) + blockIdx_x * get_local_size(0);
    unsigned idy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if(idx >= odims.dim[0] ||
       idy >= odims.dim[1] ||
       idz >= odims.dim[2] ||
       idw >= odims.dim[3])
        return;

    unsigned iMem0 = idw * istrides.dim[3] + idz * istrides.dim[2] + idy * istrides.dim[1] + idx;
    unsigned iMem1 = iMem0 + istrides.dim[DIM];
    unsigned iMem2 = iMem1 + istrides.dim[DIM];

    unsigned oMem = idw * ostrides.dim[3] + idz * ostrides.dim[2] + idy * ostrides.dim[1] + idx;

    iMem2 *= isDiff2;

    diff_this(out, in + offset, oMem, iMem0, iMem1, iMem2);
}

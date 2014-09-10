#if T == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

typedef struct {
    dim_type dims[4];
} dims_t;

__kernel
void tile_kernel(__global T *out, __global const T *in,
                 const dims_t odims, const dims_t idims,
                 const dims_t ostrides, const dims_t istrides, const dim_type offset,
                 const unsigned blocksPerMatX, const unsigned blocksPerMatY)
{
    unsigned oz = get_group_id(0) / blocksPerMatX;
    unsigned ow = get_group_id(1) / blocksPerMatY;

    unsigned blockIdx_x = get_group_id(0) - oz * blocksPerMatX;
    unsigned blockIdx_y = get_group_id(1) - ow * blocksPerMatY;

    unsigned ox = get_local_id(0) + blockIdx_x * get_local_size(0);
    unsigned oy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if(ox >= odims.dims[0] ||
       oy >= odims.dims[1] ||
       oz >= odims.dims[2] ||
       ow >= odims.dims[3])
        return;

    const dim_type ix = (idims.dims[0] == odims.dims[0]) ? ox : ox - ((ox / idims.dims[0]) * idims.dims[0]);
    const dim_type iy = (idims.dims[1] == odims.dims[1]) ? oy : oy - ((oy / idims.dims[1]) * idims.dims[1]);
    const dim_type iz = (idims.dims[2] == odims.dims[2]) ? oz : oz - ((oz / idims.dims[2]) * idims.dims[2]);
    const dim_type iw = (idims.dims[3] == odims.dims[3]) ? ow : ow - ((ow / idims.dims[3]) * idims.dims[3]);

    dim_type iMem = iw * istrides.dims[3] + iz * istrides.dims[2] +
                    iy * istrides.dims[1] + ix;
    dim_type oMem = ow * ostrides.dims[3] + oz * ostrides.dims[2] +
                    oy * ostrides.dims[1] + ox;

    out[oMem] = in[offset + iMem];
}

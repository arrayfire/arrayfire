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
    const dim_type oz = get_group_id(0) / blocksPerMatX;
    const dim_type ow = get_group_id(1) / blocksPerMatY;

    const dim_type blockIdx_x = get_group_id(0) - oz * blocksPerMatX;
    const dim_type blockIdx_y = get_group_id(1) - ow * blocksPerMatY;

    const dim_type xx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const dim_type yy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if(xx >= odims.dims[0] ||
       yy >= odims.dims[1] ||
       oz >= odims.dims[2] ||
       ow >= odims.dims[3])
        return;

    const dim_type iz = oz % idims.dims[2];
    const dim_type iw = ow % idims.dims[3];
    const dim_type izw = iw * istrides.dims[3] + iz * istrides.dims[2];
    const dim_type ozw = ow * ostrides.dims[3] + oz * ostrides.dims[2];

    const dim_type incy = blocksPerMatY * get_local_size(1);
    const dim_type incx = blocksPerMatX * get_local_size(0);

    for(dim_type oy = yy; oy < odims.dims[1]; oy += incy) {
        const dim_type iy = oy % idims.dims[1];
        for(dim_type ox = xx; ox < odims.dims[0]; ox += incx) {
            const dim_type ix = ox % idims.dims[0];

            dim_type iMem = izw + iy * istrides.dims[1] + ix;
            dim_type oMem = ozw + oy * ostrides.dims[1] + ox;

            out[oMem] = in[offset + iMem];
        }
    }
}

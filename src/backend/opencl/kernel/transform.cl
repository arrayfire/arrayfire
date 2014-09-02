#if T == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

typedef struct {
    dim_type dim[4];
} dims_t;

void calc_affine_inverse(float* txo, __global const float* txi)
{
    float det = txi[0]*txi[4] - txi[1]*txi[3];

    txo[0] = txi[4] / det;
    txo[1] = txi[3] / det;
    txo[3] = txi[1] / det;
    txo[4] = txi[0] / det;

    txo[2] = txi[2] * -txo[0] + txi[5] * -txo[1];
    txo[5] = txi[2] * -txo[3] + txi[5] * -txo[4];
}

__kernel
void transform_kernel(__global T* out, const dim_type xo, const dim_type yo,
                      __global const T* in, const dim_type xi, const dim_type yi,
                      __global const float* c_tmat,
                      const dims_t ostrides, const dims_t istrides,
                      const dim_type nimages, const dim_type ntransforms,
                      const dim_type i_offset)
{
    // Get thread indices
    int xx = get_global_id(0);
    int yy = get_global_id(1);

    if(xx >= xo * nimages || yy >= yo * ntransforms)
        return;

    // Index of channel of images and transform
    int i_idx = xx / xo;
    int t_idx = yy / yo;

    // Index in local channel -> This is output index
    int xido = xx - i_idx * xo;
    int yido = yy - t_idx * yo;


    // Global offset
    //          Offset for transform channel + Offset for image channel.
    out += t_idx * nimages * ostrides.dim[2] + i_idx * ostrides.dim[2];
    in  += i_idx * istrides.dim[2] + i_offset;

    // Transform is in global memory.
    // Needs offset to correct transform being processed.
    __global const float *tmat_ptr = c_tmat + t_idx * 6;
    float tmat[6];

    // We expect a inverse transform matrix by default
    // If it is an forward transform, then we need its inverse
    if(INVERSE == 1) {
        #pragma unroll
        for(int i = 0; i < 6; i++)
            tmat[i] = tmat_ptr[i];
    } else {
        calc_affine_inverse(tmat, tmat_ptr);
    }

    if (xido >= xo && yido >= yo) return;

    // Compute input index
    const int xidi = round(xido * tmat[0]
                         + yido * tmat[1]
                                + tmat[2]);
    const int yidi = round(xido * tmat[3]
                         + yido * tmat[4]
                                + tmat[5]);

    // Compute memory location of indices
    dim_type loci = (yidi * istrides.dim[1] + xidi);
    dim_type loco = (yido * ostrides.dim[1] + xido);

    T val = 0;
    if (xidi < xi && yidi < yi && xidi >= 0 && yidi >= 0) val = in[loci];

    out[loco] = val;
}

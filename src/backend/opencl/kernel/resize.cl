#if T == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

typedef struct {
    dim_type dim[4];
} dims_t;

////////////////////////////////////////////////////////////////////////////////////
// nearest-neighbor resampling
void resize_n_(__global T* d_out, const dim_type odim0, const dim_type odim1,
               __global const T* d_in, const dim_type idim0, const dim_type idim1,
               const dims_t ostrides, const dims_t istrides,
               const unsigned blockIdx_x, const float xf, const float yf)
{
    int const ox = get_local_id(0) + blockIdx_x * get_local_size(0);
    int const oy = get_global_id(1);

    //int ix = convert_int_rtp(ox * xf);
    //int iy = convert_int_rtp(oy * yf);
    int ix = round(ox * xf);
    int iy = round(oy * yf);

    if (ox >= odim0 || oy >= odim1) { return; }
    if (ix >= idim0) { ix = idim0 - 1; }
    if (iy >= idim1) { iy = idim1 - 1; }

    d_out[ox + oy * ostrides.dim[1]] = d_in[ix + iy * istrides.dim[1]];
}

__kernel
void resize_n(__global T *d_out, const dim_type odim0, const dim_type odim1,
              __global const T *d_in, const dim_type idim0, const dim_type idim1,
              const dims_t ostrides, const dims_t istrides, const dim_type offset,
              const unsigned b0, const float xf, const float yf)
{
    unsigned id = get_group_id(0) / b0;
    // gfor adjustment
    int i_off = id * istrides.dim[2] + offset;
    int o_off = id * ostrides.dim[2];
    unsigned blockIdx_x =  get_group_id(0) - id * b0;

    // core
    resize_n_(d_out+o_off, odim0, odim1,
              d_in +i_off, idim0, idim1,
              ostrides, istrides, blockIdx_x, xf, yf);
}

////////////////////////////////////////////////////////////////////////////////////
// bilinear resampling
void resize_b_(__global T* d_out, const dim_type odim0, const dim_type odim1,
               __global const T* d_in, const dim_type idim0, const dim_type idim1,
               const dims_t ostrides, const dims_t istrides,
               const unsigned blockIdx_x, const float xf_, const float yf_)
{
    int const ox = get_local_id(0) + blockIdx_x * get_local_size(0);
    int const oy = get_global_id(1);

    float xf = ox * xf_;
    float yf = oy * yf_;

    int ix   = floor(xf);
    int iy   = floor(yf);

    if (ox >= odim0 || oy >= odim1) { return; }
    if (ix >= idim0) { ix = idim0 - 1; }
    if (iy >= idim1) { iy = idim1 - 1; }

    float b = xf - ix;
    float a = yf - iy;

    const int ix2 = ix + 1 <  idim0 ? ix + 1 : ix;
    const int iy2 = iy + 1 <  idim1 ? iy + 1 : iy;

    const T p1 = d_in[ix  + istrides.dim[1] * iy ];
    const T p2 = d_in[ix  + istrides.dim[1] * iy2];
    const T p3 = d_in[ix2 + istrides.dim[1] * iy ];
    const T p4 = d_in[ix2 + istrides.dim[1] * iy2];

    T out = (1.0f-a) * (1.0f-b) * p1 +
            (a)      * (1.0f-b) * p2 +
            (1.0f-a) * (b)      * p3 +
            (a)      * (b)      * p4;

    d_out[ox + oy * ostrides.dim[1]] = out;
}

__kernel
void resize_b(__global T *d_out, const dim_type odim0, const dim_type odim1,
              __global const T *d_in, const dim_type idim0, const dim_type idim1,
              const dims_t ostrides, const dims_t istrides, const dim_type offset,
              const unsigned b0, const float xf, const float yf)
{
    unsigned id = get_group_id(0) / b0;
    // gfor adjustment
    int i_off = id * istrides.dim[2] + offset;
    int o_off = id * ostrides.dim[2];
    unsigned blockIdx_x =  get_group_id(0) - id * b0;

    // core
    resize_b_(d_out+o_off, odim0, odim1,
              d_in +i_off, idim0, idim1,
              ostrides, istrides, blockIdx_x, xf, yf);
}

////////////////////////////////////////////////////////////////////////////////////

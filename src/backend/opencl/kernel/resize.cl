#if T == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define NEAREST resize_n_
#define BILINEAR resize_b_

////////////////////////////////////////////////////////////////////////////////////
// nearest-neighbor resampling
void resize_n_(__global T* d_out, const KParam out,
               __global const T* d_in, const KParam in,
               const dim_type blockIdx_x, const float xf, const float yf)
{
    int const ox = get_local_id(0) + blockIdx_x * get_local_size(0);
    int const oy = get_global_id(1);

    //int ix = convert_int_rtp(ox * xf);
    //int iy = convert_int_rtp(oy * yf);
    int ix = round(ox * xf);
    int iy = round(oy * yf);

    if (ox >= out.dims[0] || oy >= out.dims[1]) { return; }
    if (ix >=  in.dims[0]) { ix = in.dims[0] - 1; }
    if (iy >=  in.dims[1]) { iy = in.dims[1] - 1; }

    d_out[ox + oy * out.strides[1]] = d_in[ix + iy * in.strides[1]];
}

////////////////////////////////////////////////////////////////////////////////////
// bilinear resampling
void resize_b_(__global T* d_out, const KParam out,
               __global const T* d_in, const KParam in,
               const dim_type blockIdx_x, const float xf_, const float yf_)
{
    int const ox = get_local_id(0) + blockIdx_x * get_local_size(0);
    int const oy = get_global_id(1);

    float xf = ox * xf_;
    float yf = oy * yf_;

    int ix   = floor(xf);
    int iy   = floor(yf);

    if (ox >= out.dims[0] || oy >= out.dims[1]) { return; }
    if (ix >=  in.dims[0]) { ix = in.dims[0] - 1; }
    if (iy >=  in.dims[1]) { iy = in.dims[1] - 1; }

    float b = xf - ix;
    float a = yf - iy;

    const int ix2 = (ix + 1) < in.dims[0] ? (ix + 1) : ix;
    const int iy2 = (iy + 1) < in.dims[1] ? (iy + 1) : iy;

    const T p1 = d_in[ix  + in.strides[1] * iy ];
    const T p2 = d_in[ix  + in.strides[1] * iy2];
    const T p3 = d_in[ix2 + in.strides[1] * iy ];
    const T p4 = d_in[ix2 + in.strides[1] * iy2];

    T val = (1.0f-a) * (1.0f-b) * p1 +
            (a)      * (1.0f-b) * p2 +
            (1.0f-a) * (b)      * p3 +
            (a)      * (b)      * p4;

    d_out[ox + oy * out.strides[1]] = val;
}

////////////////////////////////////////////////////////////////////////////////////
// Wrapper Kernel
__kernel
void resize_kernel(__global T *d_out, const KParam out,
                   __global const T *d_in, const KParam in,
                   const dim_type b0, const float xf, const float yf)
{
    dim_type id = get_group_id(0) / b0;
    // batch adjustment
    int i_off = id *  in.strides[2] + in.offset;
    int o_off = id * out.strides[2];
    dim_type blockIdx_x =  get_group_id(0) - id * b0;

    INTERP(d_out + o_off, out, d_in + i_off, in, blockIdx_x, xf, yf);
}

////////////////////////////////////////////////////////////////////////////////////

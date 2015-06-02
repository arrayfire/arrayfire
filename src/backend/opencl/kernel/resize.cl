/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if CPLX
#define set(a, b) a = b
#define set_scalar(a, b) do {                   \
        a.x = b;                                \
        a.y = 0;                                \
    } while(0)

#else

#define set(a, b) a = b
#define set_scalar(a, b) a = b

#endif

#define NEAREST resize_n_
#define BILINEAR resize_b_

////////////////////////////////////////////////////////////////////////////////////
// nearest-neighbor resampling
void resize_n_(__global T* d_out, const KParam out,
               __global const T* d_in, const KParam in,
               const int blockIdx_x, const int blockIdx_y,
               const float xf, const float yf)
{
    int const ox = get_local_id(0) + blockIdx_x * get_local_size(0);
    int const oy = get_local_id(1) + blockIdx_y * get_local_size(1);

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
               const int blockIdx_x, const int blockIdx_y,
               const float xf_, const float yf_)
{
    int const ox = get_local_id(0) + blockIdx_x * get_local_size(0);
    int const oy = get_local_id(1) + blockIdx_y * get_local_size(1);

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

    const VT p1 = d_in[ix  + in.strides[1] * iy ];
    const VT p2 = d_in[ix  + in.strides[1] * iy2];
    const VT p3 = d_in[ix2 + in.strides[1] * iy ];
    const VT p4 = d_in[ix2 + in.strides[1] * iy2];

    d_out[ox + oy * out.strides[1]] =
             (((1.0f-a) * (1.0f-b)) * p1) +
             (((a)      * (1.0f-b)) * p2) +
             (((1.0f-a) * (b)     ) * p3) +
             (((a)      * (b)     ) * p4);

}

////////////////////////////////////////////////////////////////////////////////////
// Wrapper Kernel
__kernel
void resize_kernel(__global T *d_out, const KParam out,
                   __global const T *d_in, const KParam in,
                   const int b0, const int b1, const float xf, const float yf)
{
    int bIdx = get_group_id(0) / b0;
    int bIdy = get_group_id(1) / b1;
    // batch adjustment
    int i_off = bIdy *  in.strides[3] + bIdx *  in.strides[2] + in.offset;
    int o_off = bIdy * out.strides[3] + bIdx * out.strides[2];
    int blockIdx_x =  get_group_id(0) - bIdx * b0;
    int blockIdx_y =  get_group_id(1) - bIdy * b1;

    INTERP(d_out + o_off, out, d_in + i_off, in, blockIdx_x, blockIdx_y, xf, yf);
}

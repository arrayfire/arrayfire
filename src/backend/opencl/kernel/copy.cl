/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

typedef struct {
    dim_t dim[4];
} dims_t;

inType scale(inType value, float factor)
{
#ifdef inType_float2
    return (inType)(value.s0*factor, value.s1*factor);
#else
    return (inType)(value*factor);
#endif
}

#if defined(outType_double2)

// complex double output begin
#if defined(inType_float2) || defined(inType_double2)
#define CONVERT(value) convert_double2(value)
#else
#define CONVERT(value) (double2)((value), 0.0)
#endif
// complex double output macro ends

#elif defined(outType_float2)

// complex float output begins
#if defined(inType_float2) || defined(inType_double2)
#define CONVERT(value) convert_float2(value)
#else
#define CONVERT(value) (float2)((value), 0.0f)
#endif
// complex float output macro ends

#else

// scalar output, hence no complex input
// just enforce regular casting
#define CONVERT(value) ((outType)(value))

#endif

__kernel
void copy(__global outType * dst,
          KParam oInfo,
          __global const inType * src,
          KParam iInfo,
          outType default_value,
          float factor, dims_t trgt,
          int blk_x, int blk_y)
{
    uint lx = get_local_id(0);
    uint ly = get_local_id(1);

    uint gz = get_group_id(0) / blk_x;
    uint gw = get_group_id(1) / blk_y;
    uint blockIdx_x = get_group_id(0) - (blk_x) * gz;
    uint blockIdx_y = get_group_id(1) - (blk_y) * gw;
    uint gx = blockIdx_x * get_local_size(0) + lx;
    uint gy = blockIdx_y * get_local_size(1) + ly;

    __global const inType *in = src + (gw * iInfo.strides[3] + gz * iInfo.strides[2] + gy * iInfo.strides[1] + iInfo.offset);
    __global outType *out     = dst + (gw * oInfo.strides[3] + gz * oInfo.strides[2] + gy * oInfo.strides[1] + oInfo.offset);

    uint istride0 = iInfo.strides[0];
    uint ostride0 = oInfo.strides[0];

    if (gy < oInfo.dims[1] && gz < oInfo.dims[2] && gw < oInfo.dims[3]) {
        int loop_offset = get_local_size(0) * blk_x;
        bool cond = gy < trgt.dim[1] && gz < trgt.dim[2] && gw < trgt.dim[3];
        for(int rep=gx; rep<oInfo.dims[0]; rep+=loop_offset) {
            outType temp  = default_value;
#if SAME_DIMS
            temp = CONVERT(scale(in[rep*istride0], factor));
#else
            if (rep < trgt.dim[0] && cond) {
                temp = CONVERT(scale(in[rep*istride0], factor));
            }
#endif
            out[rep*ostride0] = temp;
        }
    }
}

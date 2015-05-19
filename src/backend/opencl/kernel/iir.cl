/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifdef USE_DOUBLE
#define TR double
#else
#define TR float
#endif

#if CPLX
T __mul(T lhs, T rhs)
{
    T out;
    out.x = lhs.x * rhs.x - lhs.y * rhs.y;
    out.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return out;
}

T __cconjf(T in)
{
    T out = {in.x, -in.y};
    return out;
}

// FIXME: overflow / underflow issues
T __div(T lhs, T rhs)
{
    T out;
    TR den = (rhs.x * rhs.x + rhs.y * rhs.y);
    T num = __mul(lhs, __cconjf(rhs));

    out.x = num.x / den;
    out.y = num.y / den;

    return out;
}
#else
#define __mul(lhs, rhs) ((lhs)*(rhs))
#define __div(lhs, rhs) ((lhs)/(rhs))
#endif

__kernel
void iir_kernel(      __global T *yptr, const KParam yinfo,
                const __global T *cptr, const KParam cinfo,
                const __global T *aptr, const KParam ainfo,
                const int groups_y)
{
    __local T s_z[MAX_A_SIZE];
    __local T s_a[MAX_A_SIZE];
    __local T s_y;

    const int idz = get_group_id(0);
    const int idw = get_group_id(1) / groups_y;
    const int idy = get_group_id(1) - idw * groups_y;

    const int tx = get_local_id(0);
    const int num_a = ainfo.dims[0];

    int y_off = idw * yinfo.strides[3] + idz * yinfo.strides[2] + idy * yinfo.strides[1];
    int c_off = idw * cinfo.strides[3] + idz * cinfo.strides[2] + idy * cinfo.strides[1];

#if BATCH_A
    int a_off = idw * ainfo.strides[3] + idz * ainfo.strides[2] + idy * ainfo.strides[1];
#else
    int a_off = 0;
#endif

    __global T *d_y = yptr + y_off;
    const __global T *d_c = cptr + c_off;
    const __global T *d_a = aptr + a_off;
    const int repeat = (num_a + get_local_size(0) - 1) / get_local_size(0);

    for (int ii = 0; ii < MAX_A_SIZE / get_local_size(0); ii++) {
        int id = ii * get_local_size(0) + tx;
        s_z[id] = ZERO;
        s_a[id] = (id < num_a) ? d_a[id] : ZERO;
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    for (int i = 0; i < yinfo.dims[0]; i++) {
        if (tx == 0) {
            s_y = __div((d_c[i] + s_z[0]), s_a[0]);
            d_y[i] = s_y;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int ii = 0; ii < repeat; ii++) {
            int id = ii * get_local_size(0) + tx + 1;

            T z = s_z[id] - __mul(s_a[id],  s_y);
            barrier(CLK_LOCAL_MEM_FENCE);

            s_z[id - 1] = z;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}

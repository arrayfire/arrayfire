/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if IS_CPLX
#define IS_ZERO(val) ((val.x == 0) && (val.y == 0))
#else
#define IS_ZERO(val) (val == 0)
#endif

kernel void dense2Csr(global T *svalptr, global int *scolptr,
                      global const T *dvalptr, const KParam valinfo,
                      global const int *dcolptr, const KParam colinfo,
                      global const int *rowptr) {
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    if (gidx >= valinfo.dims[0]) return;
    if (gidy >= valinfo.dims[1]) return;

    int rowoff = rowptr[gidx];
    svalptr += rowoff;
    scolptr += rowoff;

    dvalptr += valinfo.offset;
    dcolptr += colinfo.offset;

    int idx = gidx + gidy * valinfo.strides[1];
    T val   = dvalptr[gidx + gidy * valinfo.strides[1]];
    if (IS_ZERO(val)) return;

    int oloc          = dcolptr[gidx + gidy * colinfo.strides[1]];
    svalptr[oloc - 1] = val;
    scolptr[oloc - 1] = gidy;
}

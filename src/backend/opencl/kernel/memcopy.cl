/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

typedef struct {
    int dims[4];
} dims_t;

// memcopy without looping, so dim3 has to be 1.
// conditions:
//      global dims[0] >= dims[0]
//      global dims[1] >= dims[1]
//      global dims[2] == dims[2]
//      only dims[3] == 1 will be processed!!
kernel void memCopy(global T *d_out, const dims_t ostrides, const int ooffset,
                    global const T *d_in, const dims_t idims,
                    const dims_t istrides, const int ioffset) {
    const int id0 = get_global_id(0);  // dim[0]
    const int id1 = get_global_id(1);  // dim[1]
    if ((id0 < idims.dims[0]) & (id1 < idims.dims[1])) {
        const int id2 = get_global_id(2);  // dim[2] never overflows
                                           // dim[3] is no processed
        d_out[id0 * ostrides.dims[0] + id1 * ostrides.dims[1] +
              id2 * ostrides.dims[2] + ooffset] =
            d_in[id0 * istrides.dims[0] + id1 * istrides.dims[1] +
                 id2 * istrides.dims[2] + ioffset];
    }
}

// memcopy with looping over dims[0] -- VECTOR ONLY
// Conditions:
//      global dims[0] has no restrictions
//      only dims[1] == 1 will be processed!!
//      only dims[2] == 1 will be processed!!
//      only dims[3] == 1 will be processed!!
kernel void memCopyLoop0(global T *d_out, const dims_t ostrides,
                         const int ooffset, global const T *d_in,
                         const dims_t idims, const dims_t istrides,
                         const int ioffset) {
    int id0          = get_global_id(0);  // dim[0]
    const int idims0 = idims.dims[0];
    if (id0 < idims0) {
        const int incID0        = get_global_size(0);
        const int istrides0     = istrides.dims[0];
        int idx_in              = id0 * istrides0 + ioffset;
        const int idxIncID0_in  = incID0 * istrides0;
        const int ostrides0     = ostrides.dims[0];
        int idx_out             = id0 * ostrides0 + ooffset;
        const int idxIncID0_out = incID0 * ostrides0;

        do {
            d_out[idx_out] = d_in[idx_in];
            id0 += incID0;
            if (id0 >= idims0) break;
            idx_in += idxIncID0_in;
            idx_out += idxIncID0_out;
        } while (true);
    }
}

// memcopy with looping over dims[1]
// Conditions:
//      global dims[0] >= dims[0]
//      global dims[1] has no restrictions
//      global dims[2] == dims[2]
//      only dims[3] == 1 will be processed!!
kernel void memCopyLoop1(global T *d_out, const dims_t ostrides,
                         const int ooffset, global const T *d_in,
                         const dims_t idims, const dims_t istrides,
                         const int ioffset) {
    const int id0    = get_global_id(0);  // dim[0]
    int id1          = get_global_id(1);  // dim[1]
    const int idims1 = idims.dims[1];
    if ((id0 < idims.dims[0]) & (id1 < idims1)) {
        const int id2 = get_global_id(2);  // dim[2] never overflows
                                           // dim[3] is no processed
        const int istrides1 = istrides.dims[1];
        int idx_in          = id0 * istrides.dims[0] + id1 * istrides1 +
                     id2 * istrides.dims[2] + ioffset;
        const int incID1       = get_global_size(1);
        const int idxIncID1_in = incID1 * istrides1;
        const int ostrides1    = ostrides.dims[1];
        int idx_out            = id0 * ostrides.dims[0] + id1 * ostrides1 +
                      id2 * ostrides.dims[2] + ooffset;
        const int idxIncID1_out = incID1 * ostrides1;

        do {
            d_out[idx_out] = d_in[idx_in];
            id1 += incID1;
            if (id1 >= idims1) break;
            idx_in += idxIncID1_in;
            idx_out += idxIncID1_out;
        } while (true);
    }
}

// memcopy with looping over dims[3]
// Conditions:
//      global dims[0] >= dims[0]
//      global dims[1] >= dims[1]
//      global dims[2] == dims[2]
kernel void memCopyLoop3(global T *d_out, const dims_t ostrides,
                         const int ooffset, global const T *d_in,
                         const dims_t idims, const dims_t istrides,
                         const int ioffset) {
    const int id0 = get_global_id(0);  // dim[0]
    const int id1 = get_global_id(1);  // dim[1]
    if ((id0 < idims.dims[0]) & (id1 < idims.dims[1])) {
        const int id2 = get_global_id(2);  // dim[2] never overflows
                                           // dim[3] is no processed
        int idx_in = id0 * istrides.dims[0] + id1 * istrides.dims[1] +
                     id2 * istrides.dims[2] + ioffset;
        const int idxIncID3_in = istrides.dims[3];
        const int idxEnd_in    = idims.dims[3] * idxIncID3_in + idx_in;
        int idx_out = id0 * ostrides.dims[0] + id1 * ostrides.dims[1] +
                      id2 * ostrides.dims[2] + ooffset;
        const int idxIncID3_out = ostrides.dims[3];

        do {
            d_out[idx_out] = d_in[idx_in];
            idx_in += idxIncID3_in;
            if (idx_in == idxEnd_in) break;
            idx_out += idxIncID3_out;
        } while (true);
    }
}

// memcopy with looping over dims[1] and dims[3]
// Conditions:
//      global dims[0] >= dims[0]
//      global dims[1] has no restrictions
//      global dims[2] == dims[2]
kernel void memCopyLoop13(global T *d_out, const dims_t ostrides,
                          const int ooffset, global const T *d_in,
                          const dims_t idims, const dims_t istrides,
                          const int ioffset) {
    const int id0    = get_global_id(0);  // dim[0]
    int id1          = get_global_id(1);  // dim[1]
    const int idims1 = idims.dims[1];
    if ((id0 < idims.dims[0]) & (id1 < idims1)) {
        const int id2       = get_global_id(2);  // dim[2] never overflows
        const int istrides1 = istrides.dims[1];
        int idxBase_in      = id0 * istrides.dims[0] + id1 * istrides1 +
                         id2 * istrides.dims[2] + ioffset;
        const int incID1           = get_global_size(1);
        const int idxBaseIncID1_in = incID1 * istrides1;
        const int idxIncID3_in     = istrides.dims[3];
        int idxEndID3_in           = idims.dims[3] * idxIncID3_in + idxBase_in;
        int idxBase_out = id0 * ostrides.dims[0] + id1 * ostrides.dims[1] +
                          id2 * ostrides.dims[2] + ooffset;
        const int idxBaseIncID1_out = incID1 * ostrides.dims[1];
        const int idxIncID3_out     = ostrides.dims[3];

        do {
            int idx_in  = idxBase_in;
            int idx_out = idxBase_out;
            while (true) {
                d_out[idx_out] = d_in[idx_in];
                idx_in += idxIncID3_in;
                if (idx_in == idxEndID3_in) break;
                idx_out += idxIncID3_out;
            }
            id1 += incID1;
            if (id1 >= idims1) break;
            idxBase_in += idxBaseIncID1_in;
            idxEndID3_in += idxBaseIncID1_in;
            idxBase_out += idxBaseIncID1_out;
        } while (true);
    }
}

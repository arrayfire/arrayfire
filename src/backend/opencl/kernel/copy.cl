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
} dims_type;

#ifdef FACTOR
#define SCALE(value, factor) (value * factor)
#else
#define SCALE(value, factor) (value)
#endif

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

// scaledCopy without looping, so dim3 has to be 1.
// conditions:
//      global dims[0] >= dims[0]
//      global dims[1] >= dims[1]
//      global dims[2] == dims[2]
//      only dims[3] == 1 will be processed!!
kernel void scaledCopy(global outType *out, const dims_type odims,
                       const dims_type ostrides, const int ooffset,
                       global const inType *in, const dims_type idims,
                       const dims_type istrides, const int ioffset,
                       const outType default_value, const factorType factor) {
    const int g0 = get_global_id(0);
    const int g1 = get_global_id(1);
    if ((g0 < (int)odims.dims[0]) & (g1 < (int)odims.dims[1])) {
        const int g2 = get_global_id(2);

        int idx_in = g0 * (int)istrides.dims[0] + g1 * (int)istrides.dims[1] +
                     g2 * (int)istrides.dims[2] + ioffset;
        int idx_out = g0 * (int)ostrides.dims[0] + g1 * (int)ostrides.dims[1] +
                      g2 * (int)ostrides.dims[2] + ooffset;

        if (SAME_DIMS | ((g0 < (int)idims.dims[0]) & (g1 < (int)idims.dims[1]) &
                         (g2 < (int)idims.dims[2]))) {
            out[idx_out] = CONVERT(SCALE(in[idx_in], factor));
        } else {
            out[idx_out] = default_value;
        }
    }
}

// scaledCopy with looping over dims[0] -- VECTOR ONLY
// Conditions:
//      global dims[0] has no restrictions
//      only dims[1] == 1 will be processed!!
//      only dims[2] == 1 will be processed!!
//      only dims[3] == 1 will be processed!!
kernel void scaledCopyLoop0(global outType *out, const dims_type odims,
                            const dims_type ostrides, const int ooffset,
                            global const inType *in, const dims_type idims,
                            const dims_type istrides, const int ioffset,
                            const outType default_value,
                            const factorType factor) {
    int id0              = get_global_id(0);
    const int id0End_out = odims.dims[0];
    if (id0 < id0End_out) {
        const int ostrides0     = ostrides.dims[0];
        const int id0Inc        = get_global_size(0);
        int idx_out             = id0 * ostrides0 + ooffset;
        const int idxID0Inc_out = id0Inc * ostrides0;
        const int id0End_in     = idims.dims[0];
        const int istrides0     = istrides.dims[0];
        int idx_in              = id0 * istrides0 + ioffset;
        const int idxID0Inc_in  = id0Inc * istrides0;

        while (id0 < id0End_in) {
            // inside input array, so convert
            out[idx_out] = CONVERT(SCALE(in[idx_in], factor));
            id0 += id0Inc;
            idx_in += idxID0Inc_in;
            idx_out += idxID0Inc_out;
        }
        if (!SAME_DIMS) {
            while (id0 < id0End_out) {
                // outside the input array, so copy default value
                out[idx_out] = default_value;
                id0 += id0Inc;
                idx_out += idxID0Inc_out;
            }
        }
    }
}

// scaledCopy with looping over dims[1]
// Conditions:
//      global dims[0] >= dims[0]
//      global dims[1] has no restrictions
//      global dims[2] == dims[2]
//      only dims[3] == 1 will be processed!!
kernel void scaledCopyLoop1(global outType *out, const dims_type odims,
                            const dims_type ostrides, const int ooffset,
                            global const inType *in, const dims_type idims,
                            const dims_type istrides, const int ioffset,
                            const outType default_value,
                            const factorType factor) {
    const int id0        = get_global_id(0);
    int id1              = get_global_id(1);
    const int id1End_out = odims.dims[1];
    if ((id0 < (int)odims.dims[0]) & (id1 < id1End_out)) {
        const int id2       = get_global_id(2);
        const int ostrides1 = ostrides.dims[1];
        const int id1Inc    = get_global_size(1);
        int idx_out         = id0 * (int)ostrides.dims[0] + id1 * ostrides1 +
                      id2 * (int)ostrides.dims[2] + ooffset;
        const int idxID1Inc_out = id1Inc * ostrides1;
        const int id1End_in     = idims.dims[1];
        const int istrides1     = istrides.dims[1];
        int idx_in = id0 * (int)istrides.dims[0] + id1 * istrides1 +
                     id2 * (int)istrides.dims[2] + ioffset;
        const int idxID1Inc_in = id1Inc * istrides1;

        if (SAME_DIMS | ((id0 < idims.dims[0]) & (id2 < idims.dims[2]))) {
            while (id1 < id1End_in) {
                // inside input array, so convert
                out[idx_out] = CONVERT(SCALE(in[idx_in], factor));
                id1 += id1Inc;
                idx_in += idxID1Inc_in;
                idx_out += idxID1Inc_out;
            }
        }
        if (!SAME_DIMS) {
            while (id1 < id1End_out) {
                // outside the input array, so copy default value
                out[idx_out] = default_value;
                id1 += id1Inc;
                idx_out += idxID1Inc_out;
            }
        }
    }
}

// scaledCopy with looping over dims[1] and dims[3]
// Conditions:
//      global dims[0] >= dims[0]
//      global dims[1] has no restrictions
//      global dims[2] == dims[2]
kernel void scaledCopyLoop13(global outType *out, const dims_type odims,
                             const dims_type ostrides, const int ooffset,
                             global const inType *in, const dims_type idims,
                             const dims_type istrides, const int ioffset,
                             const outType default_value,
                             const factorType factor) {
    const int id0        = get_global_id(0);
    int id1              = get_global_id(1);
    const int id1End_out = odims.dims[1];
    if ((id0 < (int)odims.dims[0]) & (id1 < id1End_out)) {
        const int id2               = get_global_id(2);
        const int id1Inc            = get_global_size(1);
        const int ostrides1         = ostrides.dims[1];
        const int idxIncID3_out     = ostrides.dims[3];
        const int idxBaseIncID1_out = id1Inc * ostrides1;
        int idxBase_out             = id0 * ostrides.dims[0] + id1 * ostrides1 +
                          id2 * ostrides.dims[2] + ooffset;
        int idxEndID3_out = odims.dims[3] * idxIncID3_out + idxBase_out;

        const int id0End_in        = idims.dims[0];
        const int id1End_in        = idims.dims[1];
        const int id2End_in        = idims.dims[2];
        const int istrides1        = istrides.dims[1];
        const int idxIncID3_in     = istrides.dims[3];
        const int idxBaseIncID1_in = id1Inc * istrides1;
        int idxBase_in             = id0 * istrides.dims[0] + id1 * istrides1 +
                         id2 * istrides.dims[2] + ioffset;
        int idxEndID3_in = idims.dims[3] * idxIncID3_in + idxBase_in;

        do {
            int idx_in  = idxBase_in;
            int idx_out = idxBase_out;
            if (SAME_DIMS |
                ((id0 < id0End_in) & (id1 < id1End_in) & (id2 < id2End_in))) {
                // inside input array, so convert
                do {
                    out[idx_out] = CONVERT(SCALE(in[idx_in], factor));
                    idx_in += idxIncID3_in;
                    idx_out += idxIncID3_out;
                } while (idx_in != idxEndID3_in);
            }
            if (!SAME_DIMS) {
                while (idx_out != idxEndID3_out) {
                    // outside the input array, so copy default value
                    out[idx_out] = default_value;
                    idx_out += idxIncID3_out;
                }
            }
            id1 += id1Inc;
            if (id1 >= id1End_out) break;
            idxBase_in += idxBaseIncID1_in;
            idxEndID3_in += idxBaseIncID1_in;
            idxBase_out += idxBaseIncID1_out;
            idxEndID3_out += idxBaseIncID1_out;
        } while (true);
    }
}
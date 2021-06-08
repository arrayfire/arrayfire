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

kernel void reshapeCopy(global outType *out, const dims_t odims,
                        const dims_t ostrides, const int ooffset,
                        global const inType *in, const dims_t idims,
                        const dims_t istrides, const int ioffset,
                        const outType default_value, const factorType factor) {
    const int g0 = get_global_id(0);  // dim0 of out buffer, not in buffer!!
    const int g1 = get_global_id(1);  // dim1 of out buffer, not in buffer!!
    const bool inside_out =
        (g0 < (int)odims.dims[0]) && (g1 < (int)odims.dims[1]);

    if (inside_out) {
        const int g2 = get_global_id(2);  // dim2 of out buffer, not in buffer!!

        int idx_in = ioffset + g0 * (int)istrides.dims[0] +
                     g1 * (int)istrides.dims[1] + g2 * (int)istrides.dims[2];
        const int istrides3 = istrides.dims[3];
        int idx_out         = ooffset + g0 * (int)ostrides.dims[0] +
                      g1 * (int)ostrides.dims[1] + g2 * (int)ostrides.dims[2];
        const int ostrides3   = ostrides.dims[3];
        const int idx_outEnd1 = idx_out + (int)idims.dims[3] * ostrides3;
#if SAME_DIMS
        do {
            outType val = CONVERT(SCALE(in[idx_in], factor));
            idx_in += istrides3;
            out[idx_out] = val;
            idx_out += ostrides3;
        } while (idx_out != idx_outEnd1);

#else
        const bool inside_in = (g0 < (int)idims.dims[0]) &&
                               (g1 < (int)idims.dims[1]) &&
                               (g2 < (int)idims.dims[2]);
        const int idx_outEnd2 = idx_out + (int)odims.dims[3] * ostrides3;
        if (inside_in) {
            do {
                outType val = CONVERT(SCALE(in[idx_in], factor));
                idx_in += istrides3;
                out[idx_out] = val;
                idx_out += ostrides3;
            } while (idx_out < idx_outEnd1);
        }
        while (idx_out < idx_outEnd2) {
            out[idx_out] = default_value;
            idx_out += ostrides3;
        }
#endif
    }
}
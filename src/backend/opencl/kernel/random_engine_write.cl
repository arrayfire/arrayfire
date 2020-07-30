/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define PI_VAL \
    3.1415926535897932384626433832795028841971693993751058209749445923078164

// Conversion to floats adapted from Random123
#define UINTMAX 0xffffffff
#define FLT_FACTOR ((1.0f) / (UINTMAX + (1.0f)))
#define HALF_FLT_FACTOR ((0.5f) * FLT_FACTOR)

// Generates rationals in (0, 1]
float getFloat(const uint *const num) {
    return ((*num) * FLT_FACTOR + HALF_FLT_FACTOR);
}

// Writes without boundary checking

void writeOut128Bytes_uchar(global uchar *out, const uint *const index,
                            const uint *const r1, const uint *const r2,
                            const uint *const r3, const uint *const r4) {
    out[*index]                = *r1;
    out[*index + THREADS]      = *r1 >> 8;
    out[*index + 2 * THREADS]  = *r1 >> 16;
    out[*index + 3 * THREADS]  = *r1 >> 24;
    out[*index + 4 * THREADS]  = *r2;
    out[*index + 5 * THREADS]  = *r2 >> 8;
    out[*index + 6 * THREADS]  = *r2 >> 16;
    out[*index + 7 * THREADS]  = *r2 >> 24;
    out[*index + 8 * THREADS]  = *r3;
    out[*index + 9 * THREADS]  = *r3 >> 8;
    out[*index + 10 * THREADS] = *r3 >> 16;
    out[*index + 11 * THREADS] = *r3 >> 24;
    out[*index + 12 * THREADS] = *r4;
    out[*index + 13 * THREADS] = *r4 >> 8;
    out[*index + 14 * THREADS] = *r4 >> 16;
    out[*index + 15 * THREADS] = *r4 >> 24;
}

void writeOut128Bytes_char(global char *out, const uint *const index,
                           const uint *const r1, const uint *const r2,
                           const uint *const r3, const uint *const r4) {
    out[*index]                = (*r1) & 0x1;
    out[*index + THREADS]      = (*r1 >> 1) & 0x1;
    out[*index + 2 * THREADS]  = (*r1 >> 2) & 0x1;
    out[*index + 3 * THREADS]  = (*r1 >> 3) & 0x1;
    out[*index + 4 * THREADS]  = (*r2) & 0x1;
    out[*index + 5 * THREADS]  = (*r2 >> 1) & 0x1;
    out[*index + 6 * THREADS]  = (*r2 >> 2) & 0x1;
    out[*index + 7 * THREADS]  = (*r2 >> 3) & 0x1;
    out[*index + 8 * THREADS]  = (*r3) & 0x1;
    out[*index + 9 * THREADS]  = (*r3 >> 1) & 0x1;
    out[*index + 10 * THREADS] = (*r3 >> 2) & 0x1;
    out[*index + 11 * THREADS] = (*r3 >> 3) & 0x1;
    out[*index + 12 * THREADS] = (*r4) & 0x1;
    out[*index + 13 * THREADS] = (*r4 >> 1) & 0x1;
    out[*index + 14 * THREADS] = (*r4 >> 2) & 0x1;
    out[*index + 15 * THREADS] = (*r4 >> 3) & 0x1;
}

void writeOut128Bytes_short(global short *out, const uint *const index,
                            const uint *const r1, const uint *const r2,
                            const uint *const r3, const uint *const r4) {
    out[*index]               = *r1;
    out[*index + THREADS]     = *r1 >> 16;
    out[*index + 2 * THREADS] = *r2;
    out[*index + 3 * THREADS] = *r2 >> 16;
    out[*index + 4 * THREADS] = *r3;
    out[*index + 5 * THREADS] = *r3 >> 16;
    out[*index + 6 * THREADS] = *r4;
    out[*index + 7 * THREADS] = *r4 >> 16;
}

void writeOut128Bytes_ushort(global ushort *out, const uint *const index,
                             const uint *const r1, const uint *const r2,
                             const uint *const r3, const uint *const r4) {
    out[*index]               = *r1;
    out[*index + THREADS]     = *r1 >> 16;
    out[*index + 2 * THREADS] = *r2;
    out[*index + 3 * THREADS] = *r2 >> 16;
    out[*index + 4 * THREADS] = *r3;
    out[*index + 5 * THREADS] = *r3 >> 16;
    out[*index + 6 * THREADS] = *r4;
    out[*index + 7 * THREADS] = *r4 >> 16;
}

void writeOut128Bytes_int(global int *out, const uint *const index,
                          const uint *const r1, const uint *const r2,
                          const uint *const r3, const uint *const r4) {
    out[*index]               = *r1;
    out[*index + THREADS]     = *r2;
    out[*index + 2 * THREADS] = *r3;
    out[*index + 3 * THREADS] = *r4;
}

void writeOut128Bytes_uint(global uint *out, const uint *const index,
                           const uint *const r1, const uint *const r2,
                           const uint *const r3, const uint *const r4) {
    out[*index]               = *r1;
    out[*index + THREADS]     = *r2;
    out[*index + 2 * THREADS] = *r3;
    out[*index + 3 * THREADS] = *r4;
}

void writeOut128Bytes_long(global long *out, const uint *const index,
                           const uint *const r1, const uint *const r2,
                           const uint *const r3, const uint *const r4) {
    long c1               = *r2;
    c1                    = (c1 << 32) | *r1;
    long c2               = *r4;
    c2                    = (c2 << 32) | *r3;
    out[*index]           = c1;
    out[*index + THREADS] = c2;
}

void writeOut128Bytes_ulong(global ulong *out, const uint *const index,
                            const uint *const r1, const uint *const r2,
                            const uint *const r3, const uint *const r4) {
    long c1               = *r2;
    c1                    = (c1 << 32) | *r1;
    long c2               = *r4;
    c2                    = (c2 << 32) | *r3;
    out[*index]           = c1;
    out[*index + THREADS] = c2;
}

void writeOut128Bytes_float(global float *out, const uint *const index,
                            const uint *const r1, const uint *const r2,
                            const uint *const r3, const uint *const r4) {
    out[*index]               = 1.f - getFloat(r1);
    out[*index + THREADS]     = 1.f - getFloat(r2);
    out[*index + 2 * THREADS] = 1.f - getFloat(r3);
    out[*index + 3 * THREADS] = 1.f - getFloat(r4);
}

#if RAND_DIST == 1
#endif

// Writes with boundary checking

void partialWriteOut128Bytes_uchar(global uchar *out, const uint *const index,
                                   const uint *const r1, const uint *const r2,
                                   const uint *const r3, const uint *const r4,
                                   const uint *const elements) {
    if (*index < *elements) { out[*index] = *r1; }
    if (*index + THREADS < *elements) { out[*index + THREADS] = *r1 >> 8; }
    if (*index + 2 * THREADS < *elements) {
        out[*index + 2 * THREADS] = *r1 >> 16;
    }
    if (*index + 3 * THREADS < *elements) {
        out[*index + 3 * THREADS] = *r1 >> 24;
    }
    if (*index + 4 * THREADS < *elements) { out[*index + 4 * THREADS] = *r2; }
    if (*index + 5 * THREADS < *elements) {
        out[*index + 5 * THREADS] = *r2 >> 8;
    }
    if (*index + 6 * THREADS < *elements) {
        out[*index + 6 * THREADS] = *r2 >> 16;
    }
    if (*index + 7 * THREADS < *elements) {
        out[*index + 7 * THREADS] = *r2 >> 24;
    }
    if (*index + 8 * THREADS < *elements) { out[*index + 8 * THREADS] = *r3; }
    if (*index + 9 * THREADS < *elements) {
        out[*index + 9 * THREADS] = *r3 >> 8;
    }
    if (*index + 10 * THREADS < *elements) {
        out[*index + 10 * THREADS] = *r3 >> 16;
    }
    if (*index + 11 * THREADS < *elements) {
        out[*index + 11 * THREADS] = *r3 >> 24;
    }
    if (*index + 12 * THREADS < *elements) { out[*index + 12 * THREADS] = *r4; }
    if (*index + 13 * THREADS < *elements) {
        out[*index + 13 * THREADS] = *r4 >> 8;
    }
    if (*index + 14 * THREADS < *elements) {
        out[*index + 14 * THREADS] = *r4 >> 16;
    }
    if (*index + 15 * THREADS < *elements) {
        out[*index + 15 * THREADS] = *r4 >> 24;
    }
}

void partialWriteOut128Bytes_char(global char *out, const uint *const index,
                                  const uint *const r1, const uint *const r2,
                                  const uint *const r3, const uint *const r4,
                                  const uint *const elements) {
    if (*index < *elements) { out[*index] = (*r1) & 0x1; }
    if (*index + THREADS < *elements) {
        out[*index + THREADS] = (*r1 >> 1) & 0x1;
    }
    if (*index + 2 * THREADS < *elements) {
        out[*index + 2 * THREADS] = (*r1 >> 2) & 0x1;
    }
    if (*index + 3 * THREADS < *elements) {
        out[*index + 3 * THREADS] = (*r1 >> 3) & 0x1;
    }
    if (*index + 4 * THREADS < *elements) {
        out[*index + 4 * THREADS] = (*r2) & 0x1;
    }
    if (*index + 5 * THREADS < *elements) {
        out[*index + 5 * THREADS] = (*r2 >> 1) & 0x1;
    }
    if (*index + 6 * THREADS < *elements) {
        out[*index + 6 * THREADS] = (*r2 >> 2) & 0x1;
    }
    if (*index + 7 * THREADS < *elements) {
        out[*index + 7 * THREADS] = (*r2 >> 3) & 0x1;
    }
    if (*index + 8 * THREADS < *elements) {
        out[*index + 8 * THREADS] = (*r3) & 0x1;
    }
    if (*index + 9 * THREADS < *elements) {
        out[*index + 9 * THREADS] = (*r3 >> 1) & 0x1;
    }
    if (*index + 10 * THREADS < *elements) {
        out[*index + 10 * THREADS] = (*r3 >> 2) & 0x1;
    }
    if (*index + 11 * THREADS < *elements) {
        out[*index + 11 * THREADS] = (*r3 >> 3) & 0x1;
    }
    if (*index + 12 * THREADS < *elements) {
        out[*index + 12 * THREADS] = (*r4) & 0x1;
    }
    if (*index + 13 * THREADS < *elements) {
        out[*index + 13 * THREADS] = (*r4 >> 1) & 0x1;
    }
    if (*index + 14 * THREADS < *elements) {
        out[*index + 14 * THREADS] = (*r4 >> 2) & 0x1;
    }
    if (*index + 15 * THREADS < *elements) {
        out[*index + 15 * THREADS] = (*r4 >> 3) & 0x1;
    }
}

void partialWriteOut128Bytes_short(global short *out, const uint *const index,
                                   const uint *const r1, const uint *const r2,
                                   const uint *const r3, const uint *const r4,
                                   const uint *const elements) {
    if (*index < *elements) { out[*index] = *r1; }
    if (*index + THREADS < *elements) { out[*index + THREADS] = *r1 >> 16; }
    if (*index + 2 * THREADS < *elements) { out[*index + 2 * THREADS] = *r2; }
    if (*index + 3 * THREADS < *elements) {
        out[*index + 3 * THREADS] = *r2 >> 16;
    }
    if (*index + 4 * THREADS < *elements) { out[*index + 4 * THREADS] = *r3; }
    if (*index + 5 * THREADS < *elements) {
        out[*index + 5 * THREADS] = *r3 >> 16;
    }
    if (*index + 6 * THREADS < *elements) { out[*index + 6 * THREADS] = *r4; }
    if (*index + 7 * THREADS < *elements) {
        out[*index + 7 * THREADS] = *r4 >> 16;
    }
}

void partialWriteOut128Bytes_ushort(global ushort *out, const uint *const index,
                                    const uint *const r1, const uint *const r2,
                                    const uint *const r3, const uint *const r4,
                                    const uint *const elements) {
    if (*index < *elements) { out[*index] = *r1; }
    if (*index + THREADS < *elements) { out[*index + THREADS] = *r1 >> 16; }
    if (*index + 2 * THREADS < *elements) { out[*index + 2 * THREADS] = *r2; }
    if (*index + 3 * THREADS < *elements) {
        out[*index + 3 * THREADS] = *r2 >> 16;
    }
    if (*index + 4 * THREADS < *elements) { out[*index + 4 * THREADS] = *r3; }
    if (*index + 5 * THREADS < *elements) {
        out[*index + 5 * THREADS] = *r3 >> 16;
    }
    if (*index + 6 * THREADS < *elements) { out[*index + 6 * THREADS] = *r4; }
    if (*index + 7 * THREADS < *elements) {
        out[*index + 7 * THREADS] = *r4 >> 16;
    }
}

void partialWriteOut128Bytes_int(global int *out, const uint *const index,
                                 const uint *const r1, const uint *const r2,
                                 const uint *const r3, const uint *const r4,
                                 const uint *const elements) {
    if (*index < *elements) { out[*index] = *r1; }
    if (*index + THREADS < *elements) { out[*index + THREADS] = *r2; }
    if (*index + 2 * THREADS < *elements) { out[*index + 2 * THREADS] = *r3; }
    if (*index + 3 * THREADS < *elements) { out[*index + 3 * THREADS] = *r4; }
}

void partialWriteOut128Bytes_uint(global uint *out, const uint *const index,
                                  const uint *const r1, const uint *const r2,
                                  const uint *const r3, const uint *const r4,
                                  const uint *const elements) {
    if (*index < *elements) { out[*index] = *r1; }
    if (*index + THREADS < *elements) { out[*index + THREADS] = *r2; }
    if (*index + 2 * THREADS < *elements) { out[*index + 2 * THREADS] = *r3; }
    if (*index + 3 * THREADS < *elements) { out[*index + 3 * THREADS] = *r4; }
}

void partialWriteOut128Bytes_long(global long *out, const uint *const index,
                                  const uint *const r1, const uint *const r2,
                                  const uint *const r3, const uint *const r4,
                                  const uint *const elements) {
    long c1 = *r2;
    c1      = (c1 << 32) | *r1;
    long c2 = *r4;
    c2      = (c2 << 32) | *r3;
    if (*index < *elements) { out[*index] = c1; }
    if (*index + THREADS < *elements) { out[*index + THREADS] = c2; }
}

void partialWriteOut128Bytes_ulong(global ulong *out, const uint *const index,
                                   const uint *const r1, const uint *const r2,
                                   const uint *const r3, const uint *const r4,
                                   const uint *const elements) {
    long c1 = *r2;
    c1      = (c1 << 32) | *r1;
    long c2 = *r4;
    c2      = (c2 << 32) | *r3;
    if (*index < *elements) { out[*index] = c1; }
    if (*index + THREADS < *elements) { out[*index + THREADS] = c2; }
}

void partialWriteOut128Bytes_float(global float *out, const uint *const index,
                                   const uint *const r1, const uint *const r2,
                                   const uint *const r3, const uint *const r4,
                                   const uint *const elements) {
    if (*index < *elements) { out[*index] = 1.f - getFloat(r1); }
    if (*index + THREADS < *elements) {
        out[*index + THREADS] = 1.f - getFloat(r2);
    }
    if (*index + 2 * THREADS < *elements) {
        out[*index + 2 * THREADS] = 1.f - getFloat(r3);
    }
    if (*index + 3 * THREADS < *elements) {
        out[*index + 3 * THREADS] = 1.f - getFloat(r4);
    }
}

#if RAND_DIST == 1
void boxMullerTransform(T *const out1, T *const out2, const T r1, const T r2) {
    /*
     * The log of a real value x where 0 < x < 1 is negative.
     */
#if defined(IS_APPLE)  // Because Apple is.. "special"
    T r = sqrt((T)(-2.0) * log10(r1) * (T)log10_val);
#else
    T r = sqrt((T)(-2.0) * log(r1));
#endif
    T theta = 2 * (T)PI_VAL * (r2);
    *out1   = r * sin(theta);
    *out2   = r * cos(theta);
}

// BoxMuller writes without boundary checking
void boxMullerWriteOut128Bytes_float(global float *out, const uint *const index,
                                     const uint *const r1, const uint *const r2,
                                     const uint *const r3,
                                     const uint *const r4) {
    float n1, n2, n3, n4;
    boxMullerTransform((T *)&n1, (T *)&n2, getFloat(r1), getFloat(r2));
    boxMullerTransform((T *)&n3, (T *)&n4, getFloat(r3), getFloat(r4));
    out[*index]               = n1;
    out[*index + THREADS]     = n2;
    out[*index + 2 * THREADS] = n3;
    out[*index + 3 * THREADS] = n4;
}

// BoxMuller writes with boundary checking
void partialBoxMullerWriteOut128Bytes_float(
    global float *out, const uint *const index, const uint *const r1,
    const uint *const r2, const uint *const r3, const uint *const r4,
    const uint *const elements) {
    float n1, n2, n3, n4;
    boxMullerTransform((T *)&n1, (T *)&n2, getFloat(r1), getFloat(r2));
    boxMullerTransform((T *)&n3, (T *)&n4, getFloat(r3), getFloat(r4));
    if (*index < *elements) { out[*index] = n1; }
    if (*index + THREADS < *elements) { out[*index + THREADS] = n2; }
    if (*index + 2 * THREADS < *elements) { out[*index + 2 * THREADS] = n3; }
    if (*index + 3 * THREADS < *elements) { out[*index + 3 * THREADS] = n4; }
}
#endif

#ifdef USE_DOUBLE

// Conversion to floats adapted from Random123
#define UINTLMAX 0xffffffffffffffff
#define DBL_FACTOR ((1.0) / (UINTLMAX + (1.0)))
#define HALF_DBL_FACTOR ((0.5) * DBL_FACTOR)

// Generates rationals in (0, 1]
double getDouble(const uint *const num1, const uint *const num2) {
    ulong num = (((ulong)*num1) << 32) | ((ulong)*num2);
    return (num * DBL_FACTOR + HALF_DBL_FACTOR);
}

void writeOut128Bytes_double(global double *out, const uint *const index,
                             const uint *const r1, const uint *const r2,
                             const uint *const r3, const uint *const r4) {
    out[*index]           = 1.0 - getDouble(r1, r2);
    out[*index + THREADS] = 1.0 - getDouble(r3, r4);
}

void partialWriteOut128Bytes_double(global double *out, const uint *const index,
                                    const uint *const r1, const uint *const r2,
                                    const uint *const r3, const uint *const r4,
                                    const uint *const elements) {
    if (*index < *elements) { out[*index] = 1.0 - getDouble(r1, r2); }
    if (*index + THREADS < *elements) {
        out[*index + THREADS] = 1.0 - getDouble(r3, r4);
    }
}

#if RAND_DIST == 1
void boxMullerWriteOut128Bytes_double(
    global double *out, const uint *const index, const uint *const r1,
    const uint *const r2, const uint *const r3, const uint *const r4) {
    double n1, n2;
    boxMullerTransform(&n1, &n2, getDouble(r1, r2), getDouble(r3, r4));
    out[*index]           = n1;
    out[*index + THREADS] = n2;
}

void partialBoxMullerWriteOut128Bytes_double(
    global double *out, const uint *const index, const uint *const r1,
    const uint *const r2, const uint *const r3, const uint *const r4,
    const uint *const elements) {
    double n1, n2;
    boxMullerTransform(&n1, &n2, getDouble(r1, r2), getDouble(r3, r4));
    if (*index < *elements) { out[*index] = n1; }
    if (*index + THREADS < *elements) { out[*index + THREADS] = n2; }
}
#endif
#endif

#ifdef USE_HALF

// Conversion to floats adapted from Random123
#define USHORTMAX 0xffff
#define HALF_FACTOR ((1.0f) / (USHORTMAX + (1.0f)))
#define HALF_HALF_FACTOR ((0.5f) * HALF_FACTOR)

// Generates rationals in (0, 1]
half getHalf(const uint *const num, int index) {
    float v = num[index >> 1U] >> (16U * (index & 1U)) & 0x0000ffff;
    return 1.0f - (v * HALF_FACTOR + HALF_HALF_FACTOR);
}

void writeOut128Bytes_half(global half *out, const uint *const index,
                           const uint *const r1, const uint *const r2,
                           const uint *const r3, const uint *const r4) {
    out[*index]               = getHalf(r1, 0);
    out[*index + THREADS]     = getHalf(r1, 1);
    out[*index + 2 * THREADS] = getHalf(r2, 0);
    out[*index + 3 * THREADS] = getHalf(r2, 1);
    out[*index + 4 * THREADS] = getHalf(r3, 0);
    out[*index + 5 * THREADS] = getHalf(r3, 1);
    out[*index + 6 * THREADS] = getHalf(r4, 0);
    out[*index + 7 * THREADS] = getHalf(r4, 1);
}

void partialWriteOut128Bytes_half(global half *out, const uint *const index,
                                  const uint *const r1, const uint *const r2,
                                  const uint *const r3, const uint *const r4,
                                  const uint *const elements) {
    if (*index < *elements) { out[*index] = getHalf(r1, 0); }
    if (*index + THREADS < *elements) {
        out[*index + THREADS] = getHalf(r1, 1);
    }
    if (*index + 2 * THREADS < *elements) {
        out[*index + 2 * THREADS] = getHalf(r2, 0);
    }
    if (*index + 3 * THREADS < *elements) {
        out[*index + 3 * THREADS] = getHalf(r2, 1);
    }
    if (*index + 4 * THREADS < *elements) {
        out[*index + 4 * THREADS] = getHalf(r3, 0);
    }
    if (*index + 5 * THREADS < *elements) {
        out[*index + 5 * THREADS] = getHalf(r3, 1);
    }
    if (*index + 6 * THREADS < *elements) {
        out[*index + 6 * THREADS] = getHalf(r4, 0);
    }
    if (*index + 7 * THREADS < *elements) {
        out[*index + 7 * THREADS] = getHalf(r4, 1);
    }
}

#if RAND_DIST == 1
void boxMullerWriteOut128Bytes_half(global half *out, const uint *const index,
                                    const uint *const r1, const uint *const r2,
                                    const uint *const r3,
                                    const uint *const r4) {
    boxMullerTransform(&out[*index], &out[*index + THREADS], getHalf(r1, 0),
                       getHalf(r1, 1));
    boxMullerTransform(&out[*index + 2 * THREADS], &out[*index + 3 * THREADS],
                       getHalf(r2, 0), getHalf(r2, 1));
    boxMullerTransform(&out[*index + 4 * THREADS], &out[*index + 5 * THREADS],
                       getHalf(r3, 0), getHalf(r3, 1));
    boxMullerTransform(&out[*index + 6 * THREADS], &out[*index + 7 * THREADS],
                       getHalf(r4, 0), getHalf(r4, 1));
}

void partialBoxMullerWriteOut128Bytes_half(
    global half *out, const uint *const index, const uint *const r1,
    const uint *const r2, const uint *const r3, const uint *const r4,
    const uint *const elements) {
    half n1, n2;
    boxMullerTransform(&n1, &n2, getHalf(r1, 0), getHalf(r1, 1));
    if (*index < *elements) { out[*index] = n1; }
    if (*index + THREADS < *elements) { out[*index + THREADS] = n2; }

    boxMullerTransform(&n1, &n2, getHalf(r2, 0), getHalf(r2, 1));
    if (*index + 2 * THREADS < *elements) { out[*index + 2 * THREADS] = n1; }
    if (*index + 3 * THREADS < *elements) { out[*index + 3 * THREADS] = n2; }

    boxMullerTransform(&n1, &n2, getHalf(r3, 0), getHalf(r3, 1));
    if (*index + 4 * THREADS < *elements) { out[*index + 4 * THREADS] = n1; }
    if (*index + 5 * THREADS < *elements) { out[*index + 5 * THREADS] = n2; }

    boxMullerTransform(&n1, &n2, getHalf(r4, 0), getHalf(r4, 1));
    if (*index + 6 * THREADS < *elements) { out[*index + 6 * THREADS] = n1; }
    if (*index + 7 * THREADS < *elements) { out[*index + 7 * THREADS] = n2; }
}
#endif
#endif

#define PASTER(x, y) x##_##y
#define EVALUATOR(x, y) PASTER(x, y)
#define EVALUATE_T(function) EVALUATOR(function, T)
#define UNIFORM_WRITE EVALUATE_T(writeOut128Bytes)
#define UNIFORM_PARTIAL_WRITE EVALUATE_T(partialWriteOut128Bytes)
#define NORMAL_WRITE EVALUATE_T(boxMullerWriteOut128Bytes)
#define NORMAL_PARTIAL_WRITE EVALUATE_T(partialBoxMullerWriteOut128Bytes)

#if RAND_DIST == 0
#define WRITE UNIFORM_WRITE
#define PARTIAL_WRITE UNIFORM_PARTIAL_WRITE
#elif RAND_DIST == 1
#define WRITE NORMAL_WRITE
#define PARTIAL_WRITE NORMAL_PARTIAL_WRITE
#endif

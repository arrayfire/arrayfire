/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Conversion to floats adapted from Random123
#define FLT_FACTOR ((1.0f) / ((float)UINT_MAX + 1.0f))
#define HALF_FLT_FACTOR ((0.5f) * FLT_FACTOR)

// Conversion to floats adapted from Random123
#define SIGNED_FLT_FACTOR ((1.0f) / ((float)INT_MAX + 1.0f))
#define SIGNED_HALF_FLT_FACTOR (0.5f * SIGNED_FLT_FACTOR)

// Generates rationals in (0, 1]
float getFloat01(uint num) {
    return fma((float)num, FLT_FACTOR, HALF_FLT_FACTOR);
}

// Generates rationals in (-1, 1]
float getFloatNegative11(uint num) {
    return fma((float)num, SIGNED_FLT_FACTOR, SIGNED_HALF_FLT_FACTOR);
}

// Writes without boundary checking

void writeOut128Bytes_uchar(global uchar *out, uint index, uint r1, uint r2,
                            uint r3, uint r4) {
    out[index]                = r1;
    out[index + THREADS]      = r1 >> 8;
    out[index + 2 * THREADS]  = r1 >> 16;
    out[index + 3 * THREADS]  = r1 >> 24;
    out[index + 4 * THREADS]  = r2;
    out[index + 5 * THREADS]  = r2 >> 8;
    out[index + 6 * THREADS]  = r2 >> 16;
    out[index + 7 * THREADS]  = r2 >> 24;
    out[index + 8 * THREADS]  = r3;
    out[index + 9 * THREADS]  = r3 >> 8;
    out[index + 10 * THREADS] = r3 >> 16;
    out[index + 11 * THREADS] = r3 >> 24;
    out[index + 12 * THREADS] = r4;
    out[index + 13 * THREADS] = r4 >> 8;
    out[index + 14 * THREADS] = r4 >> 16;
    out[index + 15 * THREADS] = r4 >> 24;
}

void writeOut128Bytes_char(global char *out, uint index, uint r1, uint r2,
                           uint r3, uint r4) {
    out[index]                = (r1)&0x1;
    out[index + THREADS]      = (r1 >> 8) & 0x1;
    out[index + 2 * THREADS]  = (r1 >> 16) & 0x1;
    out[index + 3 * THREADS]  = (r1 >> 24) & 0x1;
    out[index + 4 * THREADS]  = (r2)&0x1;
    out[index + 5 * THREADS]  = (r2 >> 8) & 0x1;
    out[index + 6 * THREADS]  = (r2 >> 16) & 0x1;
    out[index + 7 * THREADS]  = (r2 >> 24) & 0x1;
    out[index + 8 * THREADS]  = (r3)&0x1;
    out[index + 9 * THREADS]  = (r3 >> 8) & 0x1;
    out[index + 10 * THREADS] = (r3 >> 16) & 0x1;
    out[index + 11 * THREADS] = (r3 >> 24) & 0x1;
    out[index + 12 * THREADS] = (r4)&0x1;
    out[index + 13 * THREADS] = (r4 >> 8) & 0x1;
    out[index + 14 * THREADS] = (r4 >> 16) & 0x1;
    out[index + 15 * THREADS] = (r4 >> 24) & 0x1;
}

void writeOut128Bytes_short(global short *out, uint index, uint r1, uint r2,
                            uint r3, uint r4) {
    out[index]               = r1;
    out[index + THREADS]     = r1 >> 16;
    out[index + 2 * THREADS] = r2;
    out[index + 3 * THREADS] = r2 >> 16;
    out[index + 4 * THREADS] = r3;
    out[index + 5 * THREADS] = r3 >> 16;
    out[index + 6 * THREADS] = r4;
    out[index + 7 * THREADS] = r4 >> 16;
}

void writeOut128Bytes_ushort(global ushort *out, uint index, uint r1, uint r2,
                             uint r3, uint r4) {
    out[index]               = r1;
    out[index + THREADS]     = r1 >> 16;
    out[index + 2 * THREADS] = r2;
    out[index + 3 * THREADS] = r2 >> 16;
    out[index + 4 * THREADS] = r3;
    out[index + 5 * THREADS] = r3 >> 16;
    out[index + 6 * THREADS] = r4;
    out[index + 7 * THREADS] = r4 >> 16;
}

void writeOut128Bytes_int(global int *out, uint index, uint r1, uint r2,
                          uint r3, uint r4) {
    out[index]               = r1;
    out[index + THREADS]     = r2;
    out[index + 2 * THREADS] = r3;
    out[index + 3 * THREADS] = r4;
}

void writeOut128Bytes_uint(global uint *out, uint index, uint r1, uint r2,
                           uint r3, uint r4) {
    out[index]               = r1;
    out[index + THREADS]     = r2;
    out[index + 2 * THREADS] = r3;
    out[index + 3 * THREADS] = r4;
}

void writeOut128Bytes_long(global long *out, uint index, uint r1, uint r2,
                           uint r3, uint r4) {
    long c1              = r2;
    c1                   = (c1 << 32) | r1;
    long c2              = r4;
    c2                   = (c2 << 32) | r3;
    out[index]           = c1;
    out[index + THREADS] = c2;
}

void writeOut128Bytes_ulong(global ulong *out, uint index, uint r1, uint r2,
                            uint r3, uint r4) {
    long c1              = r2;
    c1                   = (c1 << 32) | r1;
    long c2              = r4;
    c2                   = (c2 << 32) | r3;
    out[index]           = c1;
    out[index + THREADS] = c2;
}

void writeOut128Bytes_float(global float *out, uint index, uint r1, uint r2,
                            uint r3, uint r4) {
    out[index]               = 1.f - getFloat01(r1);
    out[index + THREADS]     = 1.f - getFloat01(r2);
    out[index + 2 * THREADS] = 1.f - getFloat01(r3);
    out[index + 3 * THREADS] = 1.f - getFloat01(r4);
}

#if RAND_DIST == 1
void boxMullerTransform(T *const out1, T *const out2, T r1, T r2) {
    /*
     * The log of a real value x where 0 < x < 1 is negative.
     */
#if defined(IS_APPLE)  // Because Apple is.. "special"
    T r = sqrt((T)(-2.0) * log10(r2) * (T)log10_val);
#else
    T r = sqrt((T)(-2.0) * log(r2));
#endif
    T c   = cospi(r1);
    T s   = sinpi(r1);
    *out1 = r * s;
    *out2 = r * c;
}
#endif

// Writes with boundary checking

void partialWriteOut128Bytes_uchar(global uchar *out, uint index, uint r1,
                                   uint r2, uint r3, uint r4, uint elements) {
    if (index < elements) { out[index] = r1; }
    if (index + THREADS < elements) { out[index + THREADS] = r1 >> 8; }
    if (index + 2 * THREADS < elements) { out[index + 2 * THREADS] = r1 >> 16; }
    if (index + 3 * THREADS < elements) { out[index + 3 * THREADS] = r1 >> 24; }
    if (index + 4 * THREADS < elements) { out[index + 4 * THREADS] = r2; }
    if (index + 5 * THREADS < elements) { out[index + 5 * THREADS] = r2 >> 8; }
    if (index + 6 * THREADS < elements) { out[index + 6 * THREADS] = r2 >> 16; }
    if (index + 7 * THREADS < elements) { out[index + 7 * THREADS] = r2 >> 24; }
    if (index + 8 * THREADS < elements) { out[index + 8 * THREADS] = r3; }
    if (index + 9 * THREADS < elements) { out[index + 9 * THREADS] = r3 >> 8; }
    if (index + 10 * THREADS < elements) {
        out[index + 10 * THREADS] = r3 >> 16;
    }
    if (index + 11 * THREADS < elements) {
        out[index + 11 * THREADS] = r3 >> 24;
    }
    if (index + 12 * THREADS < elements) { out[index + 12 * THREADS] = r4; }
    if (index + 13 * THREADS < elements) {
        out[index + 13 * THREADS] = r4 >> 8;
    }
    if (index + 14 * THREADS < elements) {
        out[index + 14 * THREADS] = r4 >> 16;
    }
    if (index + 15 * THREADS < elements) {
        out[index + 15 * THREADS] = r4 >> 24;
    }
}

void partialWriteOut128Bytes_char(global char *out, uint index, uint r1,
                                  uint r2, uint r3, uint r4, uint elements) {
    if (index < elements) { out[index] = (r1)&0x1; }
    if (index + THREADS < elements) { out[index + THREADS] = (r1 >> 8) & 0x1; }
    if (index + 2 * THREADS < elements) {
        out[index + 2 * THREADS] = (r1 >> 16) & 0x1;
    }
    if (index + 3 * THREADS < elements) {
        out[index + 3 * THREADS] = (r1 >> 24) & 0x1;
    }
    if (index + 4 * THREADS < elements) { out[index + 4 * THREADS] = (r2)&0x1; }
    if (index + 5 * THREADS < elements) {
        out[index + 5 * THREADS] = (r2 >> 8) & 0x1;
    }
    if (index + 6 * THREADS < elements) {
        out[index + 6 * THREADS] = (r2 >> 16) & 0x1;
    }
    if (index + 7 * THREADS < elements) {
        out[index + 7 * THREADS] = (r2 >> 24) & 0x1;
    }
    if (index + 8 * THREADS < elements) { out[index + 8 * THREADS] = (r3)&0x1; }
    if (index + 9 * THREADS < elements) {
        out[index + 9 * THREADS] = (r3 >> 8) & 0x1;
    }
    if (index + 10 * THREADS < elements) {
        out[index + 10 * THREADS] = (r3 >> 16) & 0x1;
    }
    if (index + 11 * THREADS < elements) {
        out[index + 11 * THREADS] = (r3 >> 24) & 0x1;
    }
    if (index + 12 * THREADS < elements) {
        out[index + 12 * THREADS] = (r4)&0x1;
    }
    if (index + 13 * THREADS < elements) {
        out[index + 13 * THREADS] = (r4 >> 8) & 0x1;
    }
    if (index + 14 * THREADS < elements) {
        out[index + 14 * THREADS] = (r4 >> 16) & 0x1;
    }
    if (index + 15 * THREADS < elements) {
        out[index + 15 * THREADS] = (r4 >> 24) & 0x1;
    }
}

void partialWriteOut128Bytes_short(global short *out, uint index, uint r1,
                                   uint r2, uint r3, uint r4, uint elements) {
    if (index < elements) { out[index] = r1; }
    if (index + THREADS < elements) { out[index + THREADS] = r1 >> 16; }
    if (index + 2 * THREADS < elements) { out[index + 2 * THREADS] = r2; }
    if (index + 3 * THREADS < elements) { out[index + 3 * THREADS] = r2 >> 16; }
    if (index + 4 * THREADS < elements) { out[index + 4 * THREADS] = r3; }
    if (index + 5 * THREADS < elements) { out[index + 5 * THREADS] = r3 >> 16; }
    if (index + 6 * THREADS < elements) { out[index + 6 * THREADS] = r4; }
    if (index + 7 * THREADS < elements) { out[index + 7 * THREADS] = r4 >> 16; }
}

void partialWriteOut128Bytes_ushort(global ushort *out, uint index, uint r1,
                                    uint r2, uint r3, uint r4, uint elements) {
    if (index < elements) { out[index] = r1; }
    if (index + THREADS < elements) { out[index + THREADS] = r1 >> 16; }
    if (index + 2 * THREADS < elements) { out[index + 2 * THREADS] = r2; }
    if (index + 3 * THREADS < elements) { out[index + 3 * THREADS] = r2 >> 16; }
    if (index + 4 * THREADS < elements) { out[index + 4 * THREADS] = r3; }
    if (index + 5 * THREADS < elements) { out[index + 5 * THREADS] = r3 >> 16; }
    if (index + 6 * THREADS < elements) { out[index + 6 * THREADS] = r4; }
    if (index + 7 * THREADS < elements) { out[index + 7 * THREADS] = r4 >> 16; }
}

void partialWriteOut128Bytes_int(global int *out, uint index, uint r1, uint r2,
                                 uint r3, uint r4, uint elements) {
    if (index < elements) { out[index] = r1; }
    if (index + THREADS < elements) { out[index + THREADS] = r2; }
    if (index + 2 * THREADS < elements) { out[index + 2 * THREADS] = r3; }
    if (index + 3 * THREADS < elements) { out[index + 3 * THREADS] = r4; }
}

void partialWriteOut128Bytes_uint(global uint *out, uint index, uint r1,
                                  uint r2, uint r3, uint r4, uint elements) {
    if (index < elements) { out[index] = r1; }
    if (index + THREADS < elements) { out[index + THREADS] = r2; }
    if (index + 2 * THREADS < elements) { out[index + 2 * THREADS] = r3; }
    if (index + 3 * THREADS < elements) { out[index + 3 * THREADS] = r4; }
}

void partialWriteOut128Bytes_long(global long *out, uint index, uint r1,
                                  uint r2, uint r3, uint r4, uint elements) {
    long c1 = r2;
    c1      = (c1 << 32) | r1;
    long c2 = r4;
    c2      = (c2 << 32) | r3;
    if (index < elements) { out[index] = c1; }
    if (index + THREADS < elements) { out[index + THREADS] = c2; }
}

void partialWriteOut128Bytes_ulong(global ulong *out, uint index, uint r1,
                                   uint r2, uint r3, uint r4, uint elements) {
    long c1 = r2;
    c1      = (c1 << 32) | r1;
    long c2 = r4;
    c2      = (c2 << 32) | r3;
    if (index < elements) { out[index] = c1; }
    if (index + THREADS < elements) { out[index + THREADS] = c2; }
}

void partialWriteOut128Bytes_float(global float *out, uint index, uint r1,
                                   uint r2, uint r3, uint r4, uint elements) {
    if (index < elements) { out[index] = 1.f - getFloat01(r1); }
    if (index + THREADS < elements) {
        out[index + THREADS] = 1.f - getFloat01(r2);
    }
    if (index + 2 * THREADS < elements) {
        out[index + 2 * THREADS] = 1.f - getFloat01(r3);
    }
    if (index + 3 * THREADS < elements) {
        out[index + 3 * THREADS] = 1.f - getFloat01(r4);
    }
}

#if RAND_DIST == 1
// BoxMuller writes without boundary checking
void boxMullerWriteOut128Bytes_float(global float *out, uint index, uint r1,
                                     uint r2, uint r3, uint r4) {
    float n1, n2, n3, n4;
    boxMullerTransform(&n1, &n2, getFloatNegative11(r1), getFloat01(r2));
    boxMullerTransform(&n3, &n4, getFloatNegative11(r3), getFloat01(r4));
    out[index]               = n1;
    out[index + THREADS]     = n2;
    out[index + 2 * THREADS] = n3;
    out[index + 3 * THREADS] = n4;
}

// BoxMuller writes with boundary checking
void partialBoxMullerWriteOut128Bytes_float(global float *out, uint index,
                                            uint r1, uint r2, uint r3, uint r4,
                                            uint elements) {
    float n1, n2, n3, n4;
    boxMullerTransform(&n1, &n2, getFloatNegative11(r1), getFloat01(r2));
    boxMullerTransform(&n3, &n4, getFloatNegative11(r3), getFloat01(r4));
    if (index < elements) { out[index] = n1; }
    if (index + THREADS < elements) { out[index + THREADS] = n2; }
    if (index + 2 * THREADS < elements) { out[index + 2 * THREADS] = n3; }
    if (index + 3 * THREADS < elements) { out[index + 3 * THREADS] = n4; }
}
#endif

#ifdef USE_DOUBLE

// Conversion to floats adapted from Random123
#define DBL_FACTOR ((1.0) / (ULONG_MAX + (1.0)))
#define HALF_DBL_FACTOR ((0.5) * DBL_FACTOR)

#define SIGNED_DBL_FACTOR ((1.0) / (LONG_MAX + (1.0)))
#define SIGNED_HALF_DBL_FACTOR ((0.5) * SIGNED_DBL_FACTOR)

// Generates rationals in (0, 1]
double getDouble01(uint num1, uint num2) {
    ulong num = (((ulong)num1) << 32) | ((ulong)num2);
    return fma(num, DBL_FACTOR, HALF_DBL_FACTOR);
}

// Generates rationals in (-1, 1]
float getDoubleNegative11(uint num1, uint num2) {
    ulong num = (((ulong)num1) << 32) | ((ulong)num2);
    return fma(num, SIGNED_DBL_FACTOR, SIGNED_HALF_DBL_FACTOR);
}

void writeOut128Bytes_double(global double *out, uint index, uint r1, uint r2,
                             uint r3, uint r4) {
    out[index]           = 1.0 - getDouble01(r1, r2);
    out[index + THREADS] = 1.0 - getDouble01(r3, r4);
}

void partialWriteOut128Bytes_double(global double *out, uint index, uint r1,
                                    uint r2, uint r3, uint r4, uint elements) {
    if (index < elements) { out[index] = 1.0 - getDouble01(r1, r2); }
    if (index + THREADS < elements) {
        out[index + THREADS] = 1.0 - getDouble01(r3, r4);
    }
}

#if RAND_DIST == 1
void boxMullerWriteOut128Bytes_double(global double *out, uint index, uint r1,
                                      uint r2, uint r3, uint r4) {
    double n1, n2;
    boxMullerTransform(&n1, &n2, getDoubleNegative11(r1, r2),
                       getDouble01(r3, r4));
    out[index]           = n1;
    out[index + THREADS] = n2;
}

void partialBoxMullerWriteOut128Bytes_double(global double *out, uint index,
                                             uint r1, uint r2, uint r3, uint r4,
                                             uint elements) {
    double n1, n2;
    boxMullerTransform(&n1, &n2, getDoubleNegative11(r1, r2),
                       getDouble01(r3, r4));
    if (index < elements) { out[index] = n1; }
    if (index + THREADS < elements) { out[index + THREADS] = n2; }
}
#endif
#endif

#ifdef USE_HALF

// Conversion to floats adapted from Random123

// NOTE HALF_FACTOR is calculated in float to avoid conversion of 65535 to +inf
// because of the limited range of half.
#define HALF_FACTOR ((half)((1.f) / ((USHRT_MAX) + (1.f))))
#define HALF_HALF_FACTOR ((0.5h) * (HALF_FACTOR))

#define SIGNED_HALF_FACTOR ((1.h) / (SHRT_MAX + (1.h)))
#define SIGNED_HALF_HALF_FACTOR ((0.5h) * SIGNED_HALF_FACTOR)

/// This is the largest integer representable by fp16. We need to
/// make sure that the value converted from ushort is smaller than this
/// value to avoid generating infinity
#define MAX_INT_BEFORE_INFINITY (ushort)65504u

// Generates rationals in (0, 1]
half getHalf01(uint num, uint index) {
    half v = (half)min(MAX_INT_BEFORE_INFINITY,
                       (ushort)(num >> (16U * (index & 1U)) & 0x0000ffff));
    return fma(v, HALF_FACTOR, HALF_HALF_FACTOR);
}

// Generates rationals in (-1, 1]
half getHalfNegative11(uint num, uint index) {
    half v = (half)min(MAX_INT_BEFORE_INFINITY,
                       (ushort)(num >> (16U * (index & 1U)) & 0x0000ffff));
    return fma(v, SIGNED_HALF_FACTOR, SIGNED_HALF_HALF_FACTOR);
}

void writeOut128Bytes_half(global half *out, uint index, uint r1, uint r2,
                           uint r3, uint r4) {
    out[index]               = 1.h - getHalf01(r1, 0);
    out[index + THREADS]     = 1.h - getHalf01(r1, 1);
    out[index + 2 * THREADS] = 1.h - getHalf01(r2, 0);
    out[index + 3 * THREADS] = 1.h - getHalf01(r2, 1);
    out[index + 4 * THREADS] = 1.h - getHalf01(r3, 0);
    out[index + 5 * THREADS] = 1.h - getHalf01(r3, 1);
    out[index + 6 * THREADS] = 1.h - getHalf01(r4, 0);
    out[index + 7 * THREADS] = 1.h - getHalf01(r4, 1);
}

void partialWriteOut128Bytes_half(global half *out, uint index, uint r1,
                                  uint r2, uint r3, uint r4, uint elements) {
    if (index < elements) { out[index] = 1.h - getHalf01(r1, 0); }
    if (index + THREADS < elements) {
        out[index + THREADS] = 1.h - getHalf01(r1, 1);
    }
    if (index + 2 * THREADS < elements) {
        out[index + 2 * THREADS] = 1.h - getHalf01(r2, 0);
    }
    if (index + 3 * THREADS < elements) {
        out[index + 3 * THREADS] = 1.h - getHalf01(r2, 1);
    }
    if (index + 4 * THREADS < elements) {
        out[index + 4 * THREADS] = 1.h - getHalf01(r3, 0);
    }
    if (index + 5 * THREADS < elements) {
        out[index + 5 * THREADS] = 1.h - getHalf01(r3, 1);
    }
    if (index + 6 * THREADS < elements) {
        out[index + 6 * THREADS] = 1.h - getHalf01(r4, 0);
    }
    if (index + 7 * THREADS < elements) {
        out[index + 7 * THREADS] = 1.h - getHalf01(r4, 1);
    }
}

#if RAND_DIST == 1
void boxMullerWriteOut128Bytes_half(global half *out, uint index, uint r1,
                                    uint r2, uint r3, uint r4) {
    boxMullerTransform(&out[index], &out[index + THREADS],
                       getHalfNegative11(r1, 0), getHalf01(r1, 1));
    boxMullerTransform(&out[index + 2 * THREADS], &out[index + 3 * THREADS],
                       getHalfNegative11(r2, 0), getHalf01(r2, 1));
    boxMullerTransform(&out[index + 4 * THREADS], &out[index + 5 * THREADS],
                       getHalfNegative11(r3, 0), getHalf01(r3, 1));
    boxMullerTransform(&out[index + 6 * THREADS], &out[index + 7 * THREADS],
                       getHalfNegative11(r4, 0), getHalf01(r4, 1));
}

void partialBoxMullerWriteOut128Bytes_half(global half *out, uint index,
                                           uint r1, uint r2, uint r3, uint r4,
                                           uint elements) {
    half n1, n2;
    boxMullerTransform(&n1, &n2, getHalfNegative11(r1, 0), getHalf01(r1, 1));
    if (index < elements) { out[index] = n1; }
    if (index + THREADS < elements) { out[index + THREADS] = n2; }

    boxMullerTransform(&n1, &n2, getHalfNegative11(r2, 0), getHalf01(r2, 1));
    if (index + 2 * THREADS < elements) { out[index + 2 * THREADS] = n1; }
    if (index + 3 * THREADS < elements) { out[index + 3 * THREADS] = n2; }

    boxMullerTransform(&n1, &n2, getHalfNegative11(r3, 0), getHalf01(r3, 1));
    if (index + 4 * THREADS < elements) { out[index + 4 * THREADS] = n1; }
    if (index + 5 * THREADS < elements) { out[index + 5 * THREADS] = n2; }

    boxMullerTransform(&n1, &n2, getHalfNegative11(r4, 0), getHalf01(r4, 1));
    if (index + 6 * THREADS < elements) { out[index + 6 * THREADS] = n1; }
    if (index + 7 * THREADS < elements) { out[index + 7 * THREADS] = n2; }
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

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 *
 ********************************************************/

#define UINTMAXFLOAT 4294967296.0f
#define UINTLMAXDOUBLE (4294967296.0*4294967296.0)
#define PI_VAL 3.1415926535897932384626433832795028841971693993751058209749445923078164

float getFloat(const uint * const num)
{
    return ((float)(*num))/UINTMAXFLOAT;
}

//Writes without boundary checking

void writeOut256Bytes_uchar(__global uchar *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4)
{
    out[*index]              =     *r1;
    out[*index +    THREADS] =  *r1>>8;
    out[*index +  2*THREADS] = *r1>>16;
    out[*index +  3*THREADS] = *r1>>24;
    out[*index +  4*THREADS] =     *r2;
    out[*index +  5*THREADS] =  *r2>>8;
    out[*index +  6*THREADS] = *r2>>16;
    out[*index +  7*THREADS] = *r2>>24;
    out[*index +  8*THREADS] =     *r3;
    out[*index +  9*THREADS] =  *r3>>8;
    out[*index + 10*THREADS] = *r3>>16;
    out[*index + 11*THREADS] = *r3>>24;
    out[*index + 12*THREADS] =     *r4;
    out[*index + 13*THREADS] =  *r4>>8;
    out[*index + 14*THREADS] = *r4>>16;
    out[*index + 15*THREADS] = *r4>>24;
}

void writeOut256Bytes_char(__global char *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4)
{
    out[*index]              = (*r1   )&0x1;
    out[*index +    THREADS] = (*r1>>1)&0x1;
    out[*index +  2*THREADS] = (*r1>>2)&0x1;
    out[*index +  3*THREADS] = (*r1>>3)&0x1;
    out[*index +  4*THREADS] = (*r2   )&0x1;
    out[*index +  5*THREADS] = (*r2>>1)&0x1;
    out[*index +  6*THREADS] = (*r2>>2)&0x1;
    out[*index +  7*THREADS] = (*r2>>3)&0x1;
    out[*index +  8*THREADS] = (*r3   )&0x1;
    out[*index +  9*THREADS] = (*r3>>1)&0x1;
    out[*index + 10*THREADS] = (*r3>>2)&0x1;
    out[*index + 11*THREADS] = (*r3>>3)&0x1;
    out[*index + 12*THREADS] = (*r4   )&0x1;
    out[*index + 13*THREADS] = (*r4>>1)&0x1;
    out[*index + 14*THREADS] = (*r4>>2)&0x1;
    out[*index + 15*THREADS] = (*r4>>3)&0x1;
}

void writeOut256Bytes_short(__global short *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4)
{
    out[*index]             =     *r1;
    out[*index +   THREADS] = *r1>>16;
    out[*index + 2*THREADS] =     *r2;
    out[*index + 3*THREADS] = *r2>>16;
    out[*index + 4*THREADS] =     *r3;
    out[*index + 5*THREADS] = *r3>>16;
    out[*index + 6*THREADS] =     *r4;
    out[*index + 7*THREADS] = *r4>>16;
}

void writeOut256Bytes_ushort(__global ushort *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4)
{
    out[*index]             =     *r1;
    out[*index +   THREADS] = *r1>>16;
    out[*index + 2*THREADS] =     *r2;
    out[*index + 3*THREADS] = *r2>>16;
    out[*index + 4*THREADS] =     *r3;
    out[*index + 5*THREADS] = *r3>>16;
    out[*index + 6*THREADS] =     *r4;
    out[*index + 7*THREADS] = *r4>>16;
}

void writeOut256Bytes_int(__global int *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4)
{
    out[*index]             = *r1;
    out[*index +   THREADS] = *r2;
    out[*index + 2*THREADS] = *r3;
    out[*index + 3*THREADS] = *r4;
}

void writeOut256Bytes_uint(__global uint *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4)
{
    out[*index]             = *r1;
    out[*index +   THREADS] = *r2;
    out[*index + 2*THREADS] = *r3;
    out[*index + 3*THREADS] = *r4;
}

void writeOut256Bytes_long(__global long *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4)
{
    long c1 = *r2;
    c1 = (c1<<32) | *r1;
    long c2 = *r4;
    c2 = (c2<<32) | *r3;
    out[*index]           = c1;
    out[*index + THREADS] = c2;
}

void writeOut256Bytes_ulong(__global ulong *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4)
{
    long c1 = *r2;
    c1 = (c1<<32) | *r1;
    long c2 = *r4;
    c2 = (c2<<32) | *r3;
    out[*index]           = c1;
    out[*index + THREADS] = c2;
}

void writeOut256Bytes_float(__global float *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4)
{
    out[*index]             = getFloat(r1);
    out[*index +   THREADS] = getFloat(r2);
    out[*index + 2*THREADS] = getFloat(r3);
    out[*index + 3*THREADS] = getFloat(r4);
}


#if RAND_DIST == 1
#endif

//Writes with boundary checking

void partialWriteOut256Bytes_uchar(__global uchar *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4, const uint * const elements)
{
    if (*index              < *elements) {out[*index]              =     *r1;}
    if (*index +    THREADS < *elements) {out[*index +    THREADS] =  *r1>>8;}
    if (*index +  2*THREADS < *elements) {out[*index +  2*THREADS] = *r1>>16;}
    if (*index +  3*THREADS < *elements) {out[*index +  3*THREADS] = *r1>>24;}
    if (*index +  4*THREADS < *elements) {out[*index +  4*THREADS] =     *r2;}
    if (*index +  5*THREADS < *elements) {out[*index +  5*THREADS] =  *r2>>8;}
    if (*index +  6*THREADS < *elements) {out[*index +  6*THREADS] = *r2>>16;}
    if (*index +  7*THREADS < *elements) {out[*index +  7*THREADS] = *r2>>24;}
    if (*index +  8*THREADS < *elements) {out[*index +  8*THREADS] =     *r3;}
    if (*index +  9*THREADS < *elements) {out[*index +  9*THREADS] =  *r3>>8;}
    if (*index + 10*THREADS < *elements) {out[*index + 10*THREADS] = *r3>>16;}
    if (*index + 11*THREADS < *elements) {out[*index + 11*THREADS] = *r3>>24;}
    if (*index + 12*THREADS < *elements) {out[*index + 12*THREADS] =     *r4;}
    if (*index + 13*THREADS < *elements) {out[*index + 13*THREADS] =  *r4>>8;}
    if (*index + 14*THREADS < *elements) {out[*index + 14*THREADS] = *r4>>16;}
    if (*index + 15*THREADS < *elements) {out[*index + 15*THREADS] = *r4>>24;}
}

void partialWriteOut256Bytes_char(__global char *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4, const uint * const elements)
{
    if (*index              < *elements) {out[*index]              = (*r1   )&0x1;}
    if (*index +    THREADS < *elements) {out[*index +    THREADS] = (*r1>>1)&0x1;}
    if (*index +  2*THREADS < *elements) {out[*index +  2*THREADS] = (*r1>>2)&0x1;}
    if (*index +  3*THREADS < *elements) {out[*index +  3*THREADS] = (*r1>>3)&0x1;}
    if (*index +  4*THREADS < *elements) {out[*index +  4*THREADS] = (*r2   )&0x1;}
    if (*index +  5*THREADS < *elements) {out[*index +  5*THREADS] = (*r2>>1)&0x1;}
    if (*index +  6*THREADS < *elements) {out[*index +  6*THREADS] = (*r2>>2)&0x1;}
    if (*index +  7*THREADS < *elements) {out[*index +  7*THREADS] = (*r2>>3)&0x1;}
    if (*index +  8*THREADS < *elements) {out[*index +  8*THREADS] = (*r3   )&0x1;}
    if (*index +  9*THREADS < *elements) {out[*index +  9*THREADS] = (*r3>>1)&0x1;}
    if (*index + 10*THREADS < *elements) {out[*index + 10*THREADS] = (*r3>>2)&0x1;}
    if (*index + 11*THREADS < *elements) {out[*index + 11*THREADS] = (*r3>>3)&0x1;}
    if (*index + 12*THREADS < *elements) {out[*index + 12*THREADS] = (*r4   )&0x1;}
    if (*index + 13*THREADS < *elements) {out[*index + 13*THREADS] = (*r4>>1)&0x1;}
    if (*index + 14*THREADS < *elements) {out[*index + 14*THREADS] = (*r4>>2)&0x1;}
    if (*index + 15*THREADS < *elements) {out[*index + 15*THREADS] = (*r4>>3)&0x1;}
}

void partialWriteOut256Bytes_short(__global short *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4, const uint * const elements)
{
    if (*index             < *elements) {out[*index]             =     *r1;}
    if (*index +   THREADS < *elements) {out[*index +   THREADS] = *r1>>16;}
    if (*index + 2*THREADS < *elements) {out[*index + 2*THREADS] =     *r2;}
    if (*index + 3*THREADS < *elements) {out[*index + 3*THREADS] = *r2>>16;}
    if (*index + 4*THREADS < *elements) {out[*index + 4*THREADS] =     *r3;}
    if (*index + 5*THREADS < *elements) {out[*index + 5*THREADS] = *r3>>16;}
    if (*index + 6*THREADS < *elements) {out[*index + 6*THREADS] =     *r4;}
    if (*index + 7*THREADS < *elements) {out[*index + 7*THREADS] = *r4>>16;}
}

void partialWriteOut256Bytes_ushort(__global ushort *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4, const uint * const elements)
{
    if (*index             < *elements) {out[*index]             =     *r1;}
    if (*index +   THREADS < *elements) {out[*index +   THREADS] = *r1>>16;}
    if (*index + 2*THREADS < *elements) {out[*index + 2*THREADS] =     *r2;}
    if (*index + 3*THREADS < *elements) {out[*index + 3*THREADS] = *r2>>16;}
    if (*index + 4*THREADS < *elements) {out[*index + 4*THREADS] =     *r3;}
    if (*index + 5*THREADS < *elements) {out[*index + 5*THREADS] = *r3>>16;}
    if (*index + 6*THREADS < *elements) {out[*index + 6*THREADS] =     *r4;}
    if (*index + 7*THREADS < *elements) {out[*index + 7*THREADS] = *r4>>16;}
}

void partialWriteOut256Bytes_int(__global int *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4, const uint * const elements)
{
    if (*index             < *elements) {out[*index]             = *r1;}
    if (*index +   THREADS < *elements) {out[*index +   THREADS] = *r2;}
    if (*index + 2*THREADS < *elements) {out[*index + 2*THREADS] = *r3;}
    if (*index + 3*THREADS < *elements) {out[*index + 3*THREADS] = *r4;}
}

void partialWriteOut256Bytes_uint(__global uint *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4, const uint * const elements)
{
    if (*index             < *elements) {out[*index]             = *r1;}
    if (*index +   THREADS < *elements) {out[*index +   THREADS] = *r2;}
    if (*index + 2*THREADS < *elements) {out[*index + 2*THREADS] = *r3;}
    if (*index + 3*THREADS < *elements) {out[*index + 3*THREADS] = *r4;}
}

void partialWriteOut256Bytes_long(__global long *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4, const uint * const elements)
{
    long c1 = *r2;
    c1 = (c1<<32) | *r1;
    long c2 = *r4;
    c2 = (c2<<32) | *r3;
    if (*index           < *elements) {out[*index]           = c1;}
    if (*index + THREADS < *elements) {out[*index + THREADS] = c2;}
}

void partialWriteOut256Bytes_ulong(__global ulong *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4, const uint * const elements)
{
    long c1 = *r2;
    c1 = (c1<<32) | *r1;
    long c2 = *r4;
    c2 = (c2<<32) | *r3;
    if (*index           < *elements) {out[*index]           = c1;}
    if (*index + THREADS < *elements) {out[*index + THREADS] = c2;}
}

void partialWriteOut256Bytes_float(__global float *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4, const uint * const elements)
{
    if (*index             < *elements) {out[*index]             = getFloat(r1);}
    if (*index +   THREADS < *elements) {out[*index +   THREADS] = getFloat(r2);}
    if (*index + 2*THREADS < *elements) {out[*index + 2*THREADS] = getFloat(r3);}
    if (*index + 3*THREADS < *elements) {out[*index + 3*THREADS] = getFloat(r4);}
}

#if RAND_DIST == 1
void boxMullerTransform(T * const out1, T * const out2, const T r1, const T r2)
{
#if defined(IS_APPLE) // Because Apple is.. "special"
    T r = sqrt((T)(-2.0) * log10(r1) * (T)log10_val);
#else
    T r = sqrt((T)(-2.0) * log(r1));
#endif
    T theta = 2 * (T)PI_VAL * (r2);
    *out1 = r*sin(theta);
    *out2 = r*cos(theta);
}

//Normalized writes without boundary checking
void normalizedWriteOut256Bytes_float(__global float *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4)
{
    float n1, n2, n3, n4;
    boxMullerTransform(&n1, &n2, getFloat(r1), getFloat(r2));
    boxMullerTransform(&n3, &n4, getFloat(r1), getFloat(r2));
    out[*index]             = n1;
    out[*index +   THREADS] = n2;
    out[*index + 2*THREADS] = n3;
    out[*index + 3*THREADS] = n4;
}

//Normalized writes with boundary checking
void partialNormalizedWriteOut256Bytes_float(__global float *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4, const uint * const elements)
{
    float n1, n2, n3, n4;
    boxMullerTransform(&n1, &n2, getFloat(r1), getFloat(r2));
    boxMullerTransform(&n3, &n4, getFloat(r3), getFloat(r4));
    if (*index             < *elements) {out[*index]             = n1;}
    if (*index +   THREADS < *elements) {out[*index +   THREADS] = n2;}
    if (*index + 2*THREADS < *elements) {out[*index + 2*THREADS] = n3;}
    if (*index + 3*THREADS < *elements) {out[*index + 3*THREADS] = n4;}
}
#endif

#ifdef USE_DOUBLE
double getDouble(const uint * const num1, const uint * const num2)
{
    ulong num = (((ulong)*num1)<<32) | ((ulong)*num2);
    return ((double)num)/UINTLMAXDOUBLE;
}

void writeOut256Bytes_double(__global double *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4)
{
    out[*index]           = getDouble(r1, r2);
    out[*index + THREADS] = getDouble(r3, r4);
}

void partialWriteOut256Bytes_double(__global double *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4, const uint * const elements)
{
    if (*index           < *elements) {out[*index]           = getDouble(r1, r2);}
    if (*index + THREADS < *elements) {out[*index + THREADS] = getDouble(r3, r4);}
}

#if RAND_DIST == 1
void normalizedWriteOut256Bytes_double(__global double *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4)
{
    double n1, n2;
    boxMullerTransform(&n1, &n2, getDouble(r1, r2), getDouble(r3, r4));
    out[*index]           = n1;
    out[*index + THREADS] = n2;
}

void partialNormalizedWriteOut256Bytes_double(__global double *out, const uint * const index,
        const uint * const r1, const uint * const r2, const uint * const r3, const uint * const r4, const uint * const elements)
{
    double n1, n2;
    boxMullerTransform(&n1, &n2, getDouble(r1, r2), getDouble(r3, r4));
    if (*index           < *elements) {out[*index]           = n1;}
    if (*index + THREADS < *elements) {out[*index + THREADS] = n2;}
}
#endif
#endif

#define PASTER(x,y) x ## _ ## y
#define EVALUATOR(x,y) PASTER(x,y)
#define EVALUATE_T(function) EVALUATOR(function, T)
#define UNIFORM_WRITE EVALUATE_T(writeOut256Bytes)
#define UNIFORM_PARTIAL_WRITE EVALUATE_T(partialWriteOut256Bytes)
#define NORMAL_WRITE EVALUATE_T(normalizedWriteOut256Bytes)
#define NORMAL_PARTIAL_WRITE EVALUATE_T(partialNormalizedWriteOut256Bytes)

#if RAND_DIST == 0
#define WRITE UNIFORM_WRITE
#define PARTIAL_WRITE UNIFORM_PARTIAL_WRITE
#elif RAND_DIST == 1
#define WRITE NORMAL_WRITE
#define PARTIAL_WRITE NORMAL_PARTIAL_WRITE
#endif

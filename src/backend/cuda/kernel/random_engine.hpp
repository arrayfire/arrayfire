/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <kernel/random_engine_mersenne.hpp>
#include <kernel/random_engine_philox.hpp>
#include <kernel/random_engine_threefry.hpp>
#include <random_engine.hpp>
#include <af/defines.h>

#include <limits>

namespace cuda {
namespace kernel {
// Utils

static const int THREADS = 256;
#define PI_VAL \
    3.1415926535897932384626433832795028841971693993751058209749445923078164

// Conversion to half adapted from Random123
// #define HALF_FACTOR (1.0f) / (std::numeric_limits<ushort>::max() + (1.0f))
// #define HALF_HALF_FACTOR ((0.5f) * HALF_FACTOR)
//
// NOTE: The following constants for half were calculated using the formulas
// above. This is done so that we can avoid unnecessary computations because the
// __half datatype is not a constexprable type. This prevents the compiler from
// peforming these operations at compile time.
#define HALF_FACTOR __ushort_as_half(256)
#define HALF_HALF_FACTOR __ushort_as_half(128)

// Conversion to half adapted from Random123
//#define SIGNED_HALF_FACTOR                                \
    //((1.0f) / (std::numeric_limits<short>::max() + (1.0f)))
//#define SIGNED_HALF_HALF_FACTOR ((0.5f) * SIGNED_HALF_FACTOR)
//
// NOTE: The following constants for half were calculated using the formulas
// above. This is done so that we can avoid unnecessary computations because the
// __half datatype is not a constexprable type. This prevents the compiler from
// peforming these operations at compile time
#define SIGNED_HALF_FACTOR __ushort_as_half(512)
#define SIGNED_HALF_HALF_FACTOR __ushort_as_half(256)

// Conversion to floats adapted from Random123
constexpr float FLT_FACTOR =
    ((1.0f) /
     (static_cast<float>(std::numeric_limits<unsigned int>::max()) + (1.0f)));

constexpr float HALF_FLT_FACTOR = ((0.5f) * FLT_FACTOR);

// Conversion to floats adapted from Random123
constexpr float SIGNED_FLT_FACTOR =
    ((1.0) / (std::numeric_limits<int>::max() + (1.0)));
constexpr float SIGNED_HALF_FLT_FACTOR = ((0.5f) * SIGNED_FLT_FACTOR);

constexpr double DBL_FACTOR =
    ((1.0) / (std::numeric_limits<unsigned long long>::max() +
              static_cast<long double>(1.0l)));
constexpr double HALF_DBL_FACTOR((0.5) * DBL_FACTOR);

// Conversion to floats adapted from Random123
constexpr double SIGNED_DBL_FACTOR =
    ((1.0l) / (std::numeric_limits<long long>::max() + (1.0l)));
constexpr double SIGNED_HALF_DBL_FACTOR = ((0.5) * SIGNED_DBL_FACTOR);

// Generates rationals in (0, 1]
__device__ static __half oneMinusGetHalf01(uint num) {
    ushort v = num;
    return __ushort_as_half(0x3c00) -
           __hfma(static_cast<__half>(v), HALF_FACTOR, HALF_HALF_FACTOR);
}

// Generates rationals in (0, 1]
__device__ static __half getHalf01(uint num) {
    ushort v = num;
    return __hfma(static_cast<__half>(v), HALF_FACTOR, HALF_HALF_FACTOR);
}

// Generates rationals in (-1, 1]
__device__ static __half getHalfNegative11(uint num) {
    ushort v = num;
    return __hfma(static_cast<__half>(v), SIGNED_HALF_FACTOR,
                  SIGNED_HALF_HALF_FACTOR);
}

// Generates rationals in (0, 1]
__device__ static float getFloat01(uint num) {
    return fmaf(static_cast<float>(num), FLT_FACTOR, HALF_FLT_FACTOR);
}

// Generates rationals in (-1, 1]
__device__ static float getFloatNegative11(uint num) {
    return fmaf(static_cast<float>(num), SIGNED_FLT_FACTOR,
                SIGNED_HALF_FLT_FACTOR);
}

// Generates rationals in (0, 1]
__device__ static float getDouble01(uint num1, uint num2) {
    uint64_t n1 = num1;
    uint64_t n2 = num2;
    n1 <<= 32;
    uint64_t num = n1 | n2;
    return fma(static_cast<double>(num), DBL_FACTOR, HALF_DBL_FACTOR);
}

// Generates rationals in (-1, 1]
__device__ static float getDoubleNegative11(uint num1, uint num2) {
    uint32_t arr[2] = {num2, num1};
    uint64_t num;
    memcpy(&num, arr, sizeof(uint64_t));
    return fma(static_cast<double>(num), SIGNED_DBL_FACTOR,
               SIGNED_HALF_DBL_FACTOR);
}

namespace {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
__device__ __half hlog(const __half a) { return 0; }
__device__ __half hsqrt(const __half a) { return 0; }
__device__ __half hsin(const __half a) { return 0; }
__device__ __half hcos(const __half a) { return 0; }
#endif

#define MATH_FUNC(OP, DOUBLE_OP, FLOAT_OP, HALF_OP) \
    template<typename T>                            \
    __device__ T OP(T val);                         \
    template<>                                      \
    __device__ double OP(double val) {              \
        return ::DOUBLE_OP(val);                    \
    }                                               \
    template<>                                      \
    __device__ float OP(float val) {                \
        return FLOAT_OP(val);                       \
    }                                               \
    template<>                                      \
    __device__ __half OP(__half val) {              \
        return HALF_OP(val);                        \
    }

MATH_FUNC(log, log, logf, hlog)
MATH_FUNC(sqrt, sqrt, sqrtf, hsqrt)
MATH_FUNC(sin, sin, sinf, hsin)
MATH_FUNC(cos, cos, cosf, hcos)

template<typename T>
__device__ void sincos(T val, T *sptr, T *cptr);

template<>
__device__ void sincos(double val, double *sptr, double *cptr) {
    ::sincos(val, sptr, cptr);
}
template<>
__device__ void sincos(float val, float *sptr, float *cptr) {
    sincosf(val, sptr, cptr);
}
template<>
__device__ void sincos(__half val, __half *sptr, __half *cptr) {
    *sptr = hsin(val);
    *cptr = hcos(val);
}

template<typename T>
__device__ void sincospi(T val, T *sptr, T *cptr);

template<>
__device__ void sincospi(double val, double *sptr, double *cptr) {
    ::sincospi(val, sptr, cptr);
}
template<>
__device__ void sincospi(float val, float *sptr, float *cptr) {
    sincospif(val, sptr, cptr);
}
template<>
__device__ void sincospi(__half val, __half *sptr, __half *cptr) {
    // CUDA cannot make __half into a constexpr as of CUDA 11 so we are
    // converting this offline
    const __half pi_val = __ushort_as_half(0x4248);  // 0x4248 == 3.14062h
    *sptr               = hsin(val) * pi_val;
    *cptr               = hcos(val) * pi_val;
}

}  // namespace

template<typename T>
constexpr __device__ T neg_two() {
    return -2.0;
}

template<>
__device__ __half neg_two() {
    return __ushort_as_half(0xc000);  // 0xc000 == -2.h
}

template<typename T>
constexpr __device__ T two_pi() {
    return 2.0 * PI_VAL;
};

template<typename Td, typename Tc>
__device__ static void boxMullerTransform(Td *const out1, Td *const out2,
                                          const Tc &r1, const Tc &r2) {
    /*
     * The log of a real value x where 0 < x < 1 is negative.
     */
    Tc r = sqrt(neg_two<Tc>() * log(r2));
    Tc s, c;

    // Multiplying by PI instead of 2*PI seems to yeild a better distribution
    // even though the original boxMuller algorithm calls for 2 * PI
    // sincos(two_pi<Tc>() * r1, &s, &c);
    sincospi(r1, &s, &c);
    *out1 = static_cast<Td>(r * s);
    *out2 = static_cast<Td>(r * c);
}

// Writes without boundary checking
__device__ static void writeOut128Bytes(uchar *out, const uint &index,
                                        const uint &r1, const uint &r2,
                                        const uint &r3, const uint &r4) {
    out[index]                   = r1;
    out[index + blockDim.x]      = r1 >> 8;
    out[index + 2 * blockDim.x]  = r1 >> 16;
    out[index + 3 * blockDim.x]  = r1 >> 24;
    out[index + 4 * blockDim.x]  = r2;
    out[index + 5 * blockDim.x]  = r2 >> 8;
    out[index + 6 * blockDim.x]  = r2 >> 16;
    out[index + 7 * blockDim.x]  = r2 >> 24;
    out[index + 8 * blockDim.x]  = r3;
    out[index + 9 * blockDim.x]  = r3 >> 8;
    out[index + 10 * blockDim.x] = r3 >> 16;
    out[index + 11 * blockDim.x] = r3 >> 24;
    out[index + 12 * blockDim.x] = r4;
    out[index + 13 * blockDim.x] = r4 >> 8;
    out[index + 14 * blockDim.x] = r4 >> 16;
    out[index + 15 * blockDim.x] = r4 >> 24;
}

__device__ static void writeOut128Bytes(char *out, const uint &index,
                                        const uint &r1, const uint &r2,
                                        const uint &r3, const uint &r4) {
    out[index]                   = (r1)&0x1;
    out[index + blockDim.x]      = (r1 >> 1) & 0x1;
    out[index + 2 * blockDim.x]  = (r1 >> 2) & 0x1;
    out[index + 3 * blockDim.x]  = (r1 >> 3) & 0x1;
    out[index + 4 * blockDim.x]  = (r2)&0x1;
    out[index + 5 * blockDim.x]  = (r2 >> 1) & 0x1;
    out[index + 6 * blockDim.x]  = (r2 >> 2) & 0x1;
    out[index + 7 * blockDim.x]  = (r2 >> 3) & 0x1;
    out[index + 8 * blockDim.x]  = (r3)&0x1;
    out[index + 9 * blockDim.x]  = (r3 >> 1) & 0x1;
    out[index + 10 * blockDim.x] = (r3 >> 2) & 0x1;
    out[index + 11 * blockDim.x] = (r3 >> 3) & 0x1;
    out[index + 12 * blockDim.x] = (r4)&0x1;
    out[index + 13 * blockDim.x] = (r4 >> 1) & 0x1;
    out[index + 14 * blockDim.x] = (r4 >> 2) & 0x1;
    out[index + 15 * blockDim.x] = (r4 >> 3) & 0x1;
}

__device__ static void writeOut128Bytes(short *out, const uint &index,
                                        const uint &r1, const uint &r2,
                                        const uint &r3, const uint &r4) {
    out[index]                  = r1;
    out[index + blockDim.x]     = r1 >> 16;
    out[index + 2 * blockDim.x] = r2;
    out[index + 3 * blockDim.x] = r2 >> 16;
    out[index + 4 * blockDim.x] = r3;
    out[index + 5 * blockDim.x] = r3 >> 16;
    out[index + 6 * blockDim.x] = r4;
    out[index + 7 * blockDim.x] = r4 >> 16;
}

__device__ static void writeOut128Bytes(ushort *out, const uint &index,
                                        const uint &r1, const uint &r2,
                                        const uint &r3, const uint &r4) {
    writeOut128Bytes((short *)(out), index, r1, r2, r3, r4);
}

__device__ static void writeOut128Bytes(int *out, const uint &index,
                                        const uint &r1, const uint &r2,
                                        const uint &r3, const uint &r4) {
    out[index]                  = r1;
    out[index + blockDim.x]     = r2;
    out[index + 2 * blockDim.x] = r3;
    out[index + 3 * blockDim.x] = r4;
}

__device__ static void writeOut128Bytes(uint *out, const uint &index,
                                        const uint &r1, const uint &r2,
                                        const uint &r3, const uint &r4) {
    writeOut128Bytes((int *)(out), index, r1, r2, r3, r4);
}

__device__ static void writeOut128Bytes(intl *out, const uint &index,
                                        const uint &r1, const uint &r2,
                                        const uint &r3, const uint &r4) {
    intl c1                 = r2;
    c1                      = (c1 << 32) | r1;
    intl c2                 = r4;
    c2                      = (c2 << 32) | r3;
    out[index]              = c1;
    out[index + blockDim.x] = c2;
}

__device__ static void writeOut128Bytes(uintl *out, const uint &index,
                                        const uint &r1, const uint &r2,
                                        const uint &r3, const uint &r4) {
    writeOut128Bytes((intl *)(out), index, r1, r2, r3, r4);
}

__device__ static void writeOut128Bytes(float *out, const uint &index,
                                        const uint &r1, const uint &r2,
                                        const uint &r3, const uint &r4) {
    out[index]                  = 1.f - getFloat01(r1);
    out[index + blockDim.x]     = 1.f - getFloat01(r2);
    out[index + 2 * blockDim.x] = 1.f - getFloat01(r3);
    out[index + 3 * blockDim.x] = 1.f - getFloat01(r4);
}

__device__ static void writeOut128Bytes(cfloat *out, const uint &index,
                                        const uint &r1, const uint &r2,
                                        const uint &r3, const uint &r4) {
    out[index].x              = 1.f - getFloat01(r1);
    out[index].y              = 1.f - getFloat01(r2);
    out[index + blockDim.x].x = 1.f - getFloat01(r3);
    out[index + blockDim.x].y = 1.f - getFloat01(r4);
}

__device__ static void writeOut128Bytes(double *out, const uint &index,
                                        const uint &r1, const uint &r2,
                                        const uint &r3, const uint &r4) {
    out[index]              = 1.0 - getDouble01(r1, r2);
    out[index + blockDim.x] = 1.0 - getDouble01(r3, r4);
}

__device__ static void writeOut128Bytes(cdouble *out, const uint &index,
                                        const uint &r1, const uint &r2,
                                        const uint &r3, const uint &r4) {
    out[index].x = 1.0 - getDouble01(r1, r2);
    out[index].y = 1.0 - getDouble01(r3, r4);
}

__device__ static void writeOut128Bytes(common::half *out, const uint &index,
                                        const uint &r1, const uint &r2,
                                        const uint &r3, const uint &r4) {
    out[index]                  = oneMinusGetHalf01(r1);
    out[index + blockDim.x]     = oneMinusGetHalf01(r1 >> 16);
    out[index + 2 * blockDim.x] = oneMinusGetHalf01(r2);
    out[index + 3 * blockDim.x] = oneMinusGetHalf01(r2 >> 16);
    out[index + 4 * blockDim.x] = oneMinusGetHalf01(r3);
    out[index + 5 * blockDim.x] = oneMinusGetHalf01(r3 >> 16);
    out[index + 6 * blockDim.x] = oneMinusGetHalf01(r4);
    out[index + 7 * blockDim.x] = oneMinusGetHalf01(r4 >> 16);
}

// Normalized writes without boundary checking

__device__ static void boxMullerWriteOut128Bytes(float *out, const uint &index,
                                                 const uint &r1, const uint &r2,
                                                 const uint &r3,
                                                 const uint &r4) {
    boxMullerTransform(&out[index], &out[index + blockDim.x],
                       getFloatNegative11(r1), getFloat01(r2));
    boxMullerTransform(&out[index + 2 * blockDim.x],
                       &out[index + 3 * blockDim.x], getFloatNegative11(r3),
                       getFloat01(r4));
}

__device__ static void boxMullerWriteOut128Bytes(cfloat *out, const uint &index,
                                                 const uint &r1, const uint &r2,
                                                 const uint &r3,
                                                 const uint &r4) {
    boxMullerTransform(&out[index].x, &out[index].y, getFloatNegative11(r1),
                       getFloat01(r2));
    boxMullerTransform(&out[index + blockDim.x].x, &out[index + blockDim.x].y,
                       getFloatNegative11(r3), getFloat01(r4));
}

__device__ static void boxMullerWriteOut128Bytes(double *out, const uint &index,
                                                 const uint &r1, const uint &r2,
                                                 const uint &r3,
                                                 const uint &r4) {
    boxMullerTransform(&out[index], &out[index + blockDim.x],
                       getDoubleNegative11(r1, r2), getDouble01(r3, r4));
}

__device__ static void boxMullerWriteOut128Bytes(cdouble *out,
                                                 const uint &index,
                                                 const uint &r1, const uint &r2,
                                                 const uint &r3,
                                                 const uint &r4) {
    boxMullerTransform(&out[index].x, &out[index].y,
                       getDoubleNegative11(r1, r2), getDouble01(r3, r4));
}

__device__ static void boxMullerWriteOut128Bytes(common::half *out,
                                                 const uint &index,
                                                 const uint &r1, const uint &r2,
                                                 const uint &r3,
                                                 const uint &r4) {
    boxMullerTransform(&out[index], &out[index + blockDim.x],
                       getHalfNegative11(r1), getHalf01(r1 >> 16));
    boxMullerTransform(&out[index + 2 * blockDim.x],
                       &out[index + 3 * blockDim.x], getHalfNegative11(r2),
                       getHalf01(r2 >> 16));
    boxMullerTransform(&out[index + 4 * blockDim.x],
                       &out[index + 5 * blockDim.x], getHalfNegative11(r3),
                       getHalf01(r3 >> 16));
    boxMullerTransform(&out[index + 6 * blockDim.x],
                       &out[index + 7 * blockDim.x], getHalfNegative11(r4),
                       getHalf01(r4 >> 16));
}

// Writes with boundary checking

__device__ static void partialWriteOut128Bytes(uchar *out, const uint &index,
                                               const uint &r1, const uint &r2,
                                               const uint &r3, const uint &r4,
                                               const uint &elements) {
    if (index < elements) { out[index] = r1; }
    if (index + blockDim.x < elements) { out[index + blockDim.x] = r1 >> 8; }
    if (index + 2 * blockDim.x < elements) {
        out[index + 2 * blockDim.x] = r1 >> 16;
    }
    if (index + 3 * blockDim.x < elements) {
        out[index + 3 * blockDim.x] = r1 >> 24;
    }
    if (index + 4 * blockDim.x < elements) { out[index + 4 * blockDim.x] = r2; }
    if (index + 5 * blockDim.x < elements) {
        out[index + 5 * blockDim.x] = r2 >> 8;
    }
    if (index + 6 * blockDim.x < elements) {
        out[index + 6 * blockDim.x] = r2 >> 16;
    }
    if (index + 7 * blockDim.x < elements) {
        out[index + 7 * blockDim.x] = r2 >> 24;
    }
    if (index + 8 * blockDim.x < elements) { out[index + 8 * blockDim.x] = r3; }
    if (index + 9 * blockDim.x < elements) {
        out[index + 9 * blockDim.x] = r3 >> 8;
    }
    if (index + 10 * blockDim.x < elements) {
        out[index + 10 * blockDim.x] = r3 >> 16;
    }
    if (index + 11 * blockDim.x < elements) {
        out[index + 11 * blockDim.x] = r3 >> 24;
    }
    if (index + 12 * blockDim.x < elements) {
        out[index + 12 * blockDim.x] = r4;
    }
    if (index + 13 * blockDim.x < elements) {
        out[index + 13 * blockDim.x] = r4 >> 8;
    }
    if (index + 14 * blockDim.x < elements) {
        out[index + 14 * blockDim.x] = r4 >> 16;
    }
    if (index + 15 * blockDim.x < elements) {
        out[index + 15 * blockDim.x] = r4 >> 24;
    }
}

__device__ static void partialWriteOut128Bytes(char *out, const uint &index,
                                               const uint &r1, const uint &r2,
                                               const uint &r3, const uint &r4,
                                               const uint &elements) {
    if (index < elements) { out[index] = (r1)&0x1; }
    if (index + blockDim.x < elements) {
        out[index + blockDim.x] = (r1 >> 1) & 0x1;
    }
    if (index + 2 * blockDim.x < elements) {
        out[index + 2 * blockDim.x] = (r1 >> 2) & 0x1;
    }
    if (index + 3 * blockDim.x < elements) {
        out[index + 3 * blockDim.x] = (r1 >> 3) & 0x1;
    }
    if (index + 4 * blockDim.x < elements) {
        out[index + 4 * blockDim.x] = (r2)&0x1;
    }
    if (index + 5 * blockDim.x < elements) {
        out[index + 5 * blockDim.x] = (r2 >> 1) & 0x1;
    }
    if (index + 6 * blockDim.x < elements) {
        out[index + 6 * blockDim.x] = (r2 >> 2) & 0x1;
    }
    if (index + 7 * blockDim.x < elements) {
        out[index + 7 * blockDim.x] = (r2 >> 3) & 0x1;
    }
    if (index + 8 * blockDim.x < elements) {
        out[index + 8 * blockDim.x] = (r3)&0x1;
    }
    if (index + 9 * blockDim.x < elements) {
        out[index + 9 * blockDim.x] = (r3 >> 1) & 0x1;
    }
    if (index + 10 * blockDim.x < elements) {
        out[index + 10 * blockDim.x] = (r3 >> 2) & 0x1;
    }
    if (index + 11 * blockDim.x < elements) {
        out[index + 11 * blockDim.x] = (r3 >> 3) & 0x1;
    }
    if (index + 12 * blockDim.x < elements) {
        out[index + 12 * blockDim.x] = (r4)&0x1;
    }
    if (index + 13 * blockDim.x < elements) {
        out[index + 13 * blockDim.x] = (r4 >> 1) & 0x1;
    }
    if (index + 14 * blockDim.x < elements) {
        out[index + 14 * blockDim.x] = (r4 >> 2) & 0x1;
    }
    if (index + 15 * blockDim.x < elements) {
        out[index + 15 * blockDim.x] = (r4 >> 3) & 0x1;
    }
}

__device__ static void partialWriteOut128Bytes(short *out, const uint &index,
                                               const uint &r1, const uint &r2,
                                               const uint &r3, const uint &r4,
                                               const uint &elements) {
    if (index < elements) { out[index] = r1; }
    if (index + blockDim.x < elements) { out[index + blockDim.x] = r1 >> 16; }
    if (index + 2 * blockDim.x < elements) { out[index + 2 * blockDim.x] = r2; }
    if (index + 3 * blockDim.x < elements) {
        out[index + 3 * blockDim.x] = r2 >> 16;
    }
    if (index + 4 * blockDim.x < elements) { out[index + 4 * blockDim.x] = r3; }
    if (index + 5 * blockDim.x < elements) {
        out[index + 5 * blockDim.x] = r3 >> 16;
    }
    if (index + 6 * blockDim.x < elements) { out[index + 6 * blockDim.x] = r4; }
    if (index + 7 * blockDim.x < elements) {
        out[index + 7 * blockDim.x] = r4 >> 16;
    }
}

__device__ static void partialWriteOut128Bytes(ushort *out, const uint &index,
                                               const uint &r1, const uint &r2,
                                               const uint &r3, const uint &r4,
                                               const uint &elements) {
    partialWriteOut128Bytes((short *)(out), index, r1, r2, r3, r4, elements);
}

__device__ static void partialWriteOut128Bytes(int *out, const uint &index,
                                               const uint &r1, const uint &r2,
                                               const uint &r3, const uint &r4,
                                               const uint &elements) {
    if (index < elements) { out[index] = r1; }
    if (index + blockDim.x < elements) { out[index + blockDim.x] = r2; }
    if (index + 2 * blockDim.x < elements) { out[index + 2 * blockDim.x] = r3; }
    if (index + 3 * blockDim.x < elements) { out[index + 3 * blockDim.x] = r4; }
}

__device__ static void partialWriteOut128Bytes(uint *out, const uint &index,
                                               const uint &r1, const uint &r2,
                                               const uint &r3, const uint &r4,
                                               const uint &elements) {
    partialWriteOut128Bytes((int *)(out), index, r1, r2, r3, r4, elements);
}

__device__ static void partialWriteOut128Bytes(intl *out, const uint &index,
                                               const uint &r1, const uint &r2,
                                               const uint &r3, const uint &r4,
                                               const uint &elements) {
    intl c1 = r2;
    c1      = (c1 << 32) | r1;
    intl c2 = r4;
    c2      = (c2 << 32) | r3;
    if (index < elements) { out[index] = c1; }
    if (index + blockDim.x < elements) { out[index + blockDim.x] = c2; }
}

__device__ static void partialWriteOut128Bytes(uintl *out, const uint &index,
                                               const uint &r1, const uint &r2,
                                               const uint &r3, const uint &r4,
                                               const uint &elements) {
    partialWriteOut128Bytes((intl *)(out), index, r1, r2, r3, r4, elements);
}

__device__ static void partialWriteOut128Bytes(float *out, const uint &index,
                                               const uint &r1, const uint &r2,
                                               const uint &r3, const uint &r4,
                                               const uint &elements) {
    if (index < elements) { out[index] = 1.f - getFloat01(r1); }
    if (index + blockDim.x < elements) {
        out[index + blockDim.x] = 1.f - getFloat01(r2);
    }
    if (index + 2 * blockDim.x < elements) {
        out[index + 2 * blockDim.x] = 1.f - getFloat01(r3);
    }
    if (index + 3 * blockDim.x < elements) {
        out[index + 3 * blockDim.x] = 1.f - getFloat01(r4);
    }
}

__device__ static void partialWriteOut128Bytes(cfloat *out, const uint &index,
                                               const uint &r1, const uint &r2,
                                               const uint &r3, const uint &r4,
                                               const uint &elements) {
    if (index < elements) {
        out[index].x = 1.f - getFloat01(r1);
        out[index].y = 1.f - getFloat01(r2);
    }
    if (index + blockDim.x < elements) {
        out[index + blockDim.x].x = 1.f - getFloat01(r3);
        out[index + blockDim.x].y = 1.f - getFloat01(r4);
    }
}

__device__ static void partialWriteOut128Bytes(double *out, const uint &index,
                                               const uint &r1, const uint &r2,
                                               const uint &r3, const uint &r4,
                                               const uint &elements) {
    if (index < elements) { out[index] = 1.0 - getDouble01(r1, r2); }
    if (index + blockDim.x < elements) {
        out[index + blockDim.x] = 1.0 - getDouble01(r3, r4);
    }
}

__device__ static void partialWriteOut128Bytes(cdouble *out, const uint &index,
                                               const uint &r1, const uint &r2,
                                               const uint &r3, const uint &r4,
                                               const uint &elements) {
    if (index < elements) {
        out[index].x = 1.0 - getDouble01(r1, r2);
        out[index].y = 1.0 - getDouble01(r3, r4);
    }
}

// Normalized writes with boundary checking

__device__ static void partialBoxMullerWriteOut128Bytes(
    float *out, const uint &index, const uint &r1, const uint &r2,
    const uint &r3, const uint &r4, const uint &elements) {
    float n1, n2, n3, n4;
    boxMullerTransform(&n1, &n2, getFloatNegative11(r1), getFloat01(r2));
    boxMullerTransform(&n3, &n4, getFloatNegative11(r3), getFloat01(r4));
    if (index < elements) { out[index] = n1; }
    if (index + blockDim.x < elements) { out[index + blockDim.x] = n2; }
    if (index + 2 * blockDim.x < elements) { out[index + 2 * blockDim.x] = n3; }
    if (index + 3 * blockDim.x < elements) { out[index + 3 * blockDim.x] = n4; }
}

__device__ static void partialBoxMullerWriteOut128Bytes(
    cfloat *out, const uint &index, const uint &r1, const uint &r2,
    const uint &r3, const uint &r4, const uint &elements) {
    float n1, n2, n3, n4;
    boxMullerTransform(&n1, &n2, getFloatNegative11(r1), getFloat01(r2));
    boxMullerTransform(&n3, &n4, getFloatNegative11(r3), getFloat01(r4));
    if (index < elements) {
        out[index].x = n1;
        out[index].y = n2;
    }
    if (index + blockDim.x < elements) {
        out[index + blockDim.x].x = n3;
        out[index + blockDim.x].y = n4;
    }
}

__device__ static void partialBoxMullerWriteOut128Bytes(
    double *out, const uint &index, const uint &r1, const uint &r2,
    const uint &r3, const uint &r4, const uint &elements) {
    double n1, n2;
    boxMullerTransform(&n1, &n2, getDoubleNegative11(r1, r2),
                       getDouble01(r3, r4));
    if (index < elements) { out[index] = n1; }
    if (index + blockDim.x < elements) { out[index + blockDim.x] = n2; }
}

__device__ static void partialBoxMullerWriteOut128Bytes(
    cdouble *out, const uint &index, const uint &r1, const uint &r2,
    const uint &r3, const uint &r4, const uint &elements) {
    double n1, n2;
    boxMullerTransform(&n1, &n2, getDoubleNegative11(r1, r2),
                       getDouble01(r3, r4));
    if (index < elements) {
        out[index].x = n1;
        out[index].y = n2;
    }
}

__device__ static void partialWriteOut128Bytes(common::half *out,
                                               const uint &index,
                                               const uint &r1, const uint &r2,
                                               const uint &r3, const uint &r4,
                                               const uint &elements) {
    if (index < elements) { out[index] = getHalf01(r1); }
    if (index + blockDim.x < elements) {
        out[index + blockDim.x] = getHalf01(r1 >> 16);
    }
    if (index + 2 * blockDim.x < elements) {
        out[index + 2 * blockDim.x] = getHalf01(r2);
    }
    if (index + 3 * blockDim.x < elements) {
        out[index + 3 * blockDim.x] = getHalf01(r2 >> 16);
    }
    if (index + 4 * blockDim.x < elements) {
        out[index + 4 * blockDim.x] = getHalf01(r3);
    }
    if (index + 5 * blockDim.x < elements) {
        out[index + 5 * blockDim.x] = getHalf01(r3 >> 16);
    }
    if (index + 6 * blockDim.x < elements) {
        out[index + 6 * blockDim.x] = getHalf01(r4);
    }
    if (index + 7 * blockDim.x < elements) {
        out[index + 7 * blockDim.x] = getHalf01(r4 >> 16);
    }
}

// Normalized writes with boundary checking
__device__ static void partialBoxMullerWriteOut128Bytes(
    common::half *out, const uint &index, const uint &r1, const uint &r2,
    const uint &r3, const uint &r4, const uint &elements) {
    __half n[8];
    boxMullerTransform(n + 0, n + 1, getHalfNegative11(r1),
                       getHalf01(r1 >> 16));
    boxMullerTransform(n + 2, n + 3, getHalfNegative11(r2),
                       getHalf01(r2 >> 16));
    boxMullerTransform(n + 4, n + 5, getHalfNegative11(r3),
                       getHalf01(r3 >> 16));
    boxMullerTransform(n + 6, n + 7, getHalfNegative11(r4),
                       getHalf01(r4 >> 16));
    if (index < elements) { out[index] = n[0]; }
    if (index + blockDim.x < elements) { out[index + blockDim.x] = n[1]; }
    if (index + 2 * blockDim.x < elements) {
        out[index + 2 * blockDim.x] = n[2];
    }
    if (index + 3 * blockDim.x < elements) {
        out[index + 3 * blockDim.x] = n[3];
    }
    if (index + 4 * blockDim.x < elements) {
        out[index + 4 * blockDim.x] = n[4];
    }
    if (index + 5 * blockDim.x < elements) {
        out[index + 5 * blockDim.x] = n[5];
    }
    if (index + 6 * blockDim.x < elements) {
        out[index + 6 * blockDim.x] = n[6];
    }
    if (index + 7 * blockDim.x < elements) {
        out[index + 7 * blockDim.x] = n[7];
    }
}

template<typename T>
__global__ void uniformPhilox(T *out, uint hi, uint lo, uint hic, uint loc,
                              uint elementsPerBlock, uint elements) {
    uint index  = blockIdx.x * elementsPerBlock + threadIdx.x;
    uint key[2] = {lo, hi};
    uint ctr[4] = {loc, hic, 0, 0};
    ctr[0] += index;
    ctr[1] += (ctr[0] < loc);
    ctr[2] += (ctr[1] < hic);
    if (blockIdx.x != (gridDim.x - 1)) {
        philox(key, ctr);
        writeOut128Bytes(out, index, ctr[0], ctr[1], ctr[2], ctr[3]);
    } else {
        philox(key, ctr);
        partialWriteOut128Bytes(out, index, ctr[0], ctr[1], ctr[2], ctr[3],
                                elements);
    }
}

template<typename T>
__global__ void uniformThreefry(T *out, uint hi, uint lo, uint hic, uint loc,
                                uint elementsPerBlock, uint elements) {
    uint index  = blockIdx.x * elementsPerBlock + threadIdx.x;
    uint key[2] = {lo, hi};
    uint ctr[2] = {loc, hic};
    ctr[0] += index;
    ctr[1] += (ctr[0] < loc);
    uint o[4];

    threefry(key, ctr, o);
    uint step = elementsPerBlock / 2;
    ctr[0] += step;
    ctr[1] += (ctr[0] < step);
    threefry(key, ctr, o + 2);

    if (blockIdx.x != (gridDim.x - 1)) {
        writeOut128Bytes(out, index, o[0], o[1], o[2], o[3]);
    } else {
        partialWriteOut128Bytes(out, index, o[0], o[1], o[2], o[3], elements);
    }
}

template<typename T>
__global__ void uniformMersenne(T *const out, uint *const gState,
                                const uint *const pos_tbl,
                                const uint *const sh1_tbl,
                                const uint *const sh2_tbl, uint mask,
                                const uint *const g_recursion_table,
                                const uint *const g_temper_table,
                                uint elementsPerBlock, size_t elements) {
    __shared__ uint state[STATE_SIZE];
    __shared__ uint recursion_table[TABLE_SIZE];
    __shared__ uint temper_table[TABLE_SIZE];
    uint start                    = blockIdx.x * elementsPerBlock;
    uint end                      = start + elementsPerBlock;
    end                           = (end > elements) ? elements : end;
    int elementsPerBlockIteration = (blockDim.x * 4 * sizeof(uint)) / sizeof(T);
    int iter = divup((end - start), elementsPerBlockIteration);

    uint pos = pos_tbl[blockIdx.x];
    uint sh1 = sh1_tbl[blockIdx.x];
    uint sh2 = sh2_tbl[blockIdx.x];
    state_read(state, gState);
    read_table(recursion_table, g_recursion_table);
    read_table(temper_table, g_temper_table);
    __syncthreads();

    uint index = start;
    uint o[4];
    int offsetX1 = (STATE_SIZE - N + threadIdx.x) % STATE_SIZE;
    int offsetX2 = (STATE_SIZE - N + threadIdx.x + 1) % STATE_SIZE;
    int offsetY  = (STATE_SIZE - N + threadIdx.x + pos) % STATE_SIZE;
    int offsetT  = (STATE_SIZE - N + threadIdx.x + pos - 1) % STATE_SIZE;
    int offsetO  = threadIdx.x;

    for (int i = 0; i < iter; ++i) {
        for (int ii = 0; ii < 4; ++ii) {
            uint r = recursion(recursion_table, mask, sh1, sh2, state[offsetX1],
                               state[offsetX2], state[offsetY]);
            state[offsetO] = r;
            o[ii]          = temper(temper_table, r, state[offsetT]);
            offsetX1       = (offsetX1 + blockDim.x) % STATE_SIZE;
            offsetX2       = (offsetX2 + blockDim.x) % STATE_SIZE;
            offsetY        = (offsetY + blockDim.x) % STATE_SIZE;
            offsetT        = (offsetT + blockDim.x) % STATE_SIZE;
            offsetO        = (offsetO + blockDim.x) % STATE_SIZE;
            __syncthreads();
        }
        if (i == iter - 1) {
            partialWriteOut128Bytes(out, index + threadIdx.x, o[0], o[1], o[2],
                                    o[3], elements);
        } else {
            writeOut128Bytes(out, index + threadIdx.x, o[0], o[1], o[2], o[3]);
        }
        index += elementsPerBlockIteration;
    }
    state_write(gState, state);
}

template<typename T>
__global__ void normalPhilox(T *out, uint hi, uint lo, uint hic, uint loc,
                             uint elementsPerBlock, uint elements) {
    uint index  = blockIdx.x * elementsPerBlock + threadIdx.x;
    uint key[2] = {lo, hi};
    uint ctr[4] = {loc, hic, 0, 0};
    ctr[0] += index;
    ctr[1] += (ctr[0] < loc);
    ctr[2] += (ctr[1] < hic);

    philox(key, ctr);

    if (blockIdx.x != (gridDim.x - 1)) {
        boxMullerWriteOut128Bytes(out, index, ctr[0], ctr[1], ctr[2], ctr[3]);
    } else {
        partialBoxMullerWriteOut128Bytes(out, index, ctr[0], ctr[1], ctr[2],
                                         ctr[3], elements);
    }
}

template<typename T>
__global__ void normalThreefry(T *out, uint hi, uint lo, uint hic, uint loc,
                               uint elementsPerBlock, uint elements) {
    uint index  = blockIdx.x * elementsPerBlock + threadIdx.x;
    uint key[2] = {lo, hi};
    uint ctr[2] = {loc, hic};
    ctr[0] += index;
    ctr[1] += (ctr[0] < loc);
    uint o[4];

    threefry(key, ctr, o);
    uint step = elementsPerBlock / 2;
    ctr[0] += step;
    ctr[1] += (ctr[0] < step);
    threefry(key, ctr, o + 2);

    if (blockIdx.x != (gridDim.x - 1)) {
        boxMullerWriteOut128Bytes(out, index, o[0], o[1], o[2], o[3]);
    } else {
        partialBoxMullerWriteOut128Bytes(out, index, o[0], o[1], o[2], o[3],
                                         elements);
    }
}

template<typename T>
__global__ void normalMersenne(T *const out, uint *const gState,
                               const uint *const pos_tbl,
                               const uint *const sh1_tbl,
                               const uint *const sh2_tbl, uint mask,
                               const uint *const g_recursion_table,
                               const uint *const g_temper_table,
                               uint elementsPerBlock, uint elements) {
    __shared__ uint state[STATE_SIZE];
    __shared__ uint recursion_table[TABLE_SIZE];
    __shared__ uint temper_table[TABLE_SIZE];
    uint start = blockIdx.x * elementsPerBlock;
    uint end   = start + elementsPerBlock;
    end        = (end > elements) ? elements : end;
    int iter = divup((end - start) * sizeof(T), blockDim.x * 4 * sizeof(uint));

    uint pos = pos_tbl[blockIdx.x];
    uint sh1 = sh1_tbl[blockIdx.x];
    uint sh2 = sh2_tbl[blockIdx.x];
    state_read(state, gState);
    read_table(recursion_table, g_recursion_table);
    read_table(temper_table, g_temper_table);
    __syncthreads();

    uint index                    = start;
    int elementsPerBlockIteration = blockDim.x * 4 * sizeof(uint) / sizeof(T);
    uint o[4];
    int offsetX1 = (STATE_SIZE - N + threadIdx.x) % STATE_SIZE;
    int offsetX2 = (STATE_SIZE - N + threadIdx.x + 1) % STATE_SIZE;
    int offsetY  = (STATE_SIZE - N + threadIdx.x + pos) % STATE_SIZE;
    int offsetT  = (STATE_SIZE - N + threadIdx.x + pos - 1) % STATE_SIZE;
    int offsetO  = threadIdx.x;

    for (int i = 0; i < iter; ++i) {
        for (int ii = 0; ii < 4; ++ii) {
            uint r = recursion(recursion_table, mask, sh1, sh2, state[offsetX1],
                               state[offsetX2], state[offsetY]);
            state[offsetO] = r;
            o[ii]          = temper(temper_table, r, state[offsetT]);
            offsetX1       = (offsetX1 + blockDim.x) % STATE_SIZE;
            offsetX2       = (offsetX2 + blockDim.x) % STATE_SIZE;
            offsetY        = (offsetY + blockDim.x) % STATE_SIZE;
            offsetT        = (offsetT + blockDim.x) % STATE_SIZE;
            offsetO        = (offsetO + blockDim.x) % STATE_SIZE;
            __syncthreads();
        }
        if (i == iter - 1) {
            partialBoxMullerWriteOut128Bytes(out, index + threadIdx.x, o[0],
                                             o[1], o[2], o[3], elements);
        } else {
            boxMullerWriteOut128Bytes(out, index + threadIdx.x, o[0], o[1],
                                      o[2], o[3]);
        }
        index += elementsPerBlockIteration;
    }
    state_write(gState, state);
}

template<typename T>
void uniformDistributionMT(T *out, size_t elements, uint *const state,
                           const uint *const pos, const uint *const sh1,
                           const uint *const sh2, uint mask,
                           const uint *const recursion_table,
                           const uint *const temper_table) {
    int threads                = THREADS;
    int min_elements_per_block = 32 * threads * 4 * sizeof(uint) / sizeof(T);
    int blocks                 = divup(elements, min_elements_per_block);
    blocks                     = (blocks > BLOCKS) ? BLOCKS : blocks;
    uint elementsPerBlock      = divup(elements, blocks);
    CUDA_LAUNCH(uniformMersenne, blocks, threads, out, state, pos, sh1, sh2,
                mask, recursion_table, temper_table, elementsPerBlock,
                elements);
}

template<typename T>
void normalDistributionMT(T *out, size_t elements, uint *const state,
                          const uint *const pos, const uint *const sh1,
                          const uint *const sh2, uint mask,
                          const uint *const recursion_table,
                          const uint *const temper_table) {
    int threads                = THREADS;
    int min_elements_per_block = 32 * threads * 4 * sizeof(uint) / sizeof(T);
    int blocks                 = divup(elements, min_elements_per_block);
    blocks                     = (blocks > BLOCKS) ? BLOCKS : blocks;
    uint elementsPerBlock      = divup(elements, blocks);
    CUDA_LAUNCH(normalMersenne, blocks, threads, out, state, pos, sh1, sh2,
                mask, recursion_table, temper_table, elementsPerBlock,
                elements);
}

template<typename T>
void uniformDistributionCBRNG(T *out, size_t elements,
                              const af_random_engine_type type,
                              const uintl &seed, uintl &counter) {
    int threads          = THREADS;
    int elementsPerBlock = threads * 4 * sizeof(uint) / sizeof(T);
    int blocks           = divup(elements, elementsPerBlock);
    uint hi              = seed >> 32;
    uint lo              = seed;
    uint hic             = counter >> 32;
    uint loc             = counter;
    switch (type) {
        case AF_RANDOM_ENGINE_PHILOX_4X32_10:
            CUDA_LAUNCH(uniformPhilox, blocks, threads, out, hi, lo, hic, loc,
                        elementsPerBlock, elements);
            break;
        case AF_RANDOM_ENGINE_THREEFRY_2X32_16:
            CUDA_LAUNCH(uniformThreefry, blocks, threads, out, hi, lo, hic, loc,
                        elementsPerBlock, elements);
            break;
        default:
            AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
    }
    counter += elements;
}

template<typename T>
void normalDistributionCBRNG(T *out, size_t elements,
                             const af_random_engine_type type,
                             const uintl &seed, uintl &counter) {
    int threads          = THREADS;
    int elementsPerBlock = threads * 4 * sizeof(uint) / sizeof(T);
    int blocks           = divup(elements, elementsPerBlock);
    uint hi              = seed >> 32;
    uint lo              = seed;
    uint hic             = counter >> 32;
    uint loc             = counter;
    switch (type) {
        case AF_RANDOM_ENGINE_PHILOX_4X32_10:
            CUDA_LAUNCH(normalPhilox, blocks, threads, out, hi, lo, hic, loc,
                        elementsPerBlock, elements);
            break;
        case AF_RANDOM_ENGINE_THREEFRY_2X32_16:
            CUDA_LAUNCH(normalThreefry, blocks, threads, out, hi, lo, hic, loc,
                        elementsPerBlock, elements);
            break;
        default:
            AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
    }
    counter += elements;
}
}  // namespace kernel
}  // namespace cuda

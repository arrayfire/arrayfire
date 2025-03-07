/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once
#include <sycl/sycl.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {

// TODO: !!!! half functions still need to be ported !!!!

//// Conversion to half adapted from Random123
//// #define HALF_FACTOR (1.0f) / (std::numeric_limits<ushort>::max() + (1.0f))
//// #define HALF_HALF_FACTOR ((0.5f) * HALF_FACTOR)
////
//// NOTE: The following constants for half were calculated using the formulas
//// above. This is done so that we can avoid unnecessary computations because
/// the / __half datatype is not a constexprable type. This prevents the
/// compiler from / peforming these operations at compile time.
// #define HALF_FACTOR __ushort_as_half(0x100u)
// #define HALF_HALF_FACTOR __ushort_as_half(0x80)
//
//// Conversion to half adapted from Random123
////#define SIGNED_HALF_FACTOR                                \
//    //((1.0f) / (std::numeric_limits<short>::max() + (1.0f)))
////#define SIGNED_HALF_HALF_FACTOR ((0.5f) * SIGNED_HALF_FACTOR)
////
//// NOTE: The following constants for half were calculated using the formulas
//// above. This is done so that we can avoid unnecessary computations because
/// the / __half datatype is not a constexprable type. This prevents the
/// compiler from / peforming these operations at compile time
// #define SIGNED_HALF_FACTOR __ushort_as_half(0x200u)
// #define SIGNED_HALF_HALF_FACTOR __ushort_as_half(0x100u)
//
///// This is the largest integer representable by fp16. We need to
///// make sure that the value converted from ushort is smaller than this
///// value to avoid generating infinity
// constexpr ushort max_int_before_infinity = 65504;
//
//// Generates rationals in (0, 1]
//__device__ static __half oneMinusGetHalf01(uint num) {
//    // convert to ushort before the min operation
//    ushort v = min(max_int_before_infinity, ushort(num));
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
//    return (1.0f - __half2float(__hfma(__ushort2half_rn(v), HALF_FACTOR,
//                                       HALF_HALF_FACTOR)));
// #else
//    __half out = __ushort_as_half(0x3c00u) /*1.0h*/ -
//                 __hfma(__ushort2half_rn(v), HALF_FACTOR, HALF_HALF_FACTOR);
//    if (__hisinf(out)) printf("val: %d ushort: %d\n", num, v);
//    return out;
// #endif
//}
//
//// Generates rationals in (0, 1]
//__device__ static __half getHalf01(uint num) {
//    // convert to ushort before the min operation
//    ushort v = min(max_int_before_infinity, ushort(num));
//    return __hfma(__ushort2half_rn(v), HALF_FACTOR, HALF_HALF_FACTOR);
//}
//
//// Generates rationals in (-1, 1]
//__device__ static __half getHalfNegative11(uint num) {
//    // convert to ushort before the min operation
//    ushort v = min(max_int_before_infinity, ushort(num));
//    return __hfma(__ushort2half_rn(v), SIGNED_HALF_FACTOR,
//                  SIGNED_HALF_HALF_FACTOR);
//}
//
// Generates rationals in (0, 1]
static float getFloat01(uint num) {
    // Conversion to floats adapted from Random123
    constexpr float factor =
        ((1.0f) /
         (static_cast<float>(std::numeric_limits<unsigned int>::max()) +
          (1.0f)));
    constexpr float half_factor = ((0.5f) * factor);

    return sycl::fma(static_cast<float>(num), factor, half_factor);
}

// Generates rationals in (-1, 1]
static float getFloatNegative11(uint num) {
    // Conversion to floats adapted from Random123
    constexpr float factor =
        ((1.0) /
         (static_cast<double>(std::numeric_limits<int>::max()) + (1.0)));
    constexpr float half_factor = ((0.5f) * factor);

    return sycl::fma(static_cast<float>(num), factor, half_factor);
}

// Generates rationals in (0, 1]
static double getDouble01(uint num1, uint num2) {
    uint64_t n1 = num1;
    uint64_t n2 = num2;
    n1 <<= 32;
    uint64_t num = n1 | n2;
    constexpr double factor =
        ((1.0) /
         (static_cast<double>(std::numeric_limits<unsigned long long>::max()) +
          static_cast<double>(1.0)));
    constexpr double half_factor((0.5) * factor);

    return sycl::fma(static_cast<double>(num), factor, half_factor);
}

// Conversion to doubles adapted from Random123
constexpr double signed_factor =
    ((1.0l) / (static_cast<long double>(std::numeric_limits<long long>::max()) +
               (1.0l)));
constexpr double half_factor = ((0.5) * signed_factor);

// Generates rationals in (-1, 1]
static double getDoubleNegative11(uint num1, uint num2) {
    uint32_t arr[2] = {num2, num1};
    uint64_t num;

    memcpy(&num, arr, sizeof(uint64_t));
    return sycl::fma(static_cast<double>(num), signed_factor, half_factor);
}

namespace {
//
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
// #define HALF_MATH_FUNC(OP, HALF_OP)    \
//    template<>                         \
//    __device__ __half OP(__half val) { \
//        return ::HALF_OP(val);         \
//    }
// #else
// #define HALF_MATH_FUNC(OP, HALF_OP)     \
//    template<>                          \
//    __device__ __half OP(__half val) {  \
//        float fval = __half2float(val); \
//        return __float2half(OP(fval));  \
//    }
// #endif
//
// #define MATH_FUNC(OP, DOUBLE_OP, FLOAT_OP, HALF_OP) \
//    template<typename T>                            \
//    __device__ T OP(T val);                         \
//    template<>                                      \
//    __device__ double OP(double val) {              \
//        return ::DOUBLE_OP(val);                    \
//    }                                               \
//    template<>                                      \
//    __device__ float OP(float val) {                \
//        return ::FLOAT_OP(val);                     \
//    }                                               \
//    HALF_MATH_FUNC(OP, HALF_OP)
//
// MATH_FUNC(log, log, logf, hlog)
// MATH_FUNC(sqrt, sqrt, sqrtf, hsqrt)
// MATH_FUNC(sin, sin, sinf, hsin)
// MATH_FUNC(cos, cos, cosf, hcos)
//
// template<typename T>
//__device__ void sincos(T val, T *sptr, T *cptr);
//
// template<>
//__device__ void sincos(double val, double *sptr, double *cptr) {
//    ::sincos(val, sptr, cptr);
//}
//
// template<>
//__device__ void sincos(float val, float *sptr, float *cptr) {
//    sincosf(val, sptr, cptr);
//}
//
// template<>
//__device__ void sincos(__half val, __half *sptr, __half *cptr) {
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
//    *sptr = sin(val);
//    *cptr = cos(val);
// #else
//    float s, c;
//    float fval = __half2float(val);
//    sincos(fval, &s, &c);
//    *sptr = __float2half(s);
//    *cptr = __float2half(c);
// #endif
//}
//
template<typename T>
void sincospi(T val, T *sptr, T *cptr) {
    *sptr = sycl::sinpi(val);
    *cptr = sycl::cospi(val);
}

// template<>
//__device__ void sincospi(__half val, __half *sptr, __half *cptr) {
//    // CUDA cannot make __half into a constexpr as of CUDA 11 so we are
//    // converting this offline
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
//    const __half pi_val = __ushort_as_half(0x4248);  // 0x4248 == 3.14062h
//    val *= pi_val;
//    *sptr = sin(val);
//    *cptr = cos(val);
// #else
//    float fval = __half2float(val);
//    float s, c;
//    sincospi(fval, &s, &c);
//    *sptr = __float2half(s);
//    *cptr = __float2half(c);
// #endif
//}
//
}  // namespace
//
template<typename T>
constexpr T neg_two() {
    return -2.0;
}
//
// template<typename T>
// constexpr __device__ T two_pi() {
//    return 2.0 * PI_VAL;
//};
//
template<typename Tc>
static void boxMullerTransform(cfloat *const cOut, const Tc &r1, const Tc &r2) {
    /*
     * The log of a real value x where 0 < x < 1 is negative.
     */
    Tc r = sycl::sqrt(neg_two<Tc>() * sycl::log(r2));
    Tc s, c;

    // Multiplying by PI instead of 2*PI seems to yeild a better distribution
    // even though the original boxMuller algorithm calls for 2 * PI
    // sincos(two_pi<Tc>() * r1, &s, &c);
    sincospi(r1, &s, &c);
    cOut->real(static_cast<float>(r * s));
    cOut->imag(static_cast<float>(r * c));
}

template<typename Tc>
static void boxMullerTransform(cdouble *const cOut, const Tc &r1,
                               const Tc &r2) {
    /*
     * The log of a real value x where 0 < x < 1 is negative.
     */
    Tc r = sycl::sqrt(neg_two<Tc>() * sycl::log(r2));
    Tc s, c;

    // Multiplying by PI instead of 2*PI seems to yeild a better distribution
    // even though the original boxMuller algorithm calls for 2 * PI
    // sincos(two_pi<Tc>() * r1, &s, &c);
    sincospi(r1, &s, &c);
    cOut->real(static_cast<double>(r * s));
    cOut->imag(static_cast<double>(r * c));
}

template<typename Td, typename Tc>
static void boxMullerTransform(Td *const out1, Td *const out2, const Tc &r1,
                               const Tc &r2) {
    /*
     * The log of a real value x where 0 < x < 1 is negative.
     */
    Tc r = sycl::sqrt(neg_two<Tc>() * sycl::log(r2));
    Tc s, c;

    // Multiplying by PI instead of 2*PI seems to yeild a better distribution
    // even though the original boxMuller algorithm calls for 2 * PI
    // sincos(two_pi<Tc>() * r1, &s, &c);
    sincospi(r1, &s, &c);
    *out1 = static_cast<Td>(r * s);
    *out2 = static_cast<Td>(r * c);
}
// template<>
//__device__ void boxMullerTransform<arrayfire::common::half, __half>(
//    arrayfire::common::half *const out1, arrayfire::common::half *const out2,
//    const __half &r1, const __half &r2) { float o1, o2; float fr1 =
//    __half2float(r1); float fr2 = __half2float(r2); boxMullerTransform(&o1,
//    &o2, fr1, fr2); *out1 = o1; *out2 = o2;
//}

// Writes without boundary checking
static void writeOut128Bytes(uchar *out, const uint &index, const uint groupSz,
                             const uint &r1, const uint &r2, const uint &r3,
                             const uint &r4) {
    out[index]                = r1;
    out[index + groupSz]      = r1 >> 8;
    out[index + 2 * groupSz]  = r1 >> 16;
    out[index + 3 * groupSz]  = r1 >> 24;
    out[index + 4 * groupSz]  = r2;
    out[index + 5 * groupSz]  = r2 >> 8;
    out[index + 6 * groupSz]  = r2 >> 16;
    out[index + 7 * groupSz]  = r2 >> 24;
    out[index + 8 * groupSz]  = r3;
    out[index + 9 * groupSz]  = r3 >> 8;
    out[index + 10 * groupSz] = r3 >> 16;
    out[index + 11 * groupSz] = r3 >> 24;
    out[index + 12 * groupSz] = r4;
    out[index + 13 * groupSz] = r4 >> 8;
    out[index + 14 * groupSz] = r4 >> 16;
    out[index + 15 * groupSz] = r4 >> 24;
}

static void writeOut128Bytes(char *out, const uint &index, const uint groupSz,
                             const uint &r1, const uint &r2, const uint &r3,
                             const uint &r4) {
    out[index]                = (r1)&0x1;
    out[index + groupSz]      = (r1 >> 8) & 0x1;
    out[index + 2 * groupSz]  = (r1 >> 16) & 0x1;
    out[index + 3 * groupSz]  = (r1 >> 24) & 0x1;
    out[index + 4 * groupSz]  = (r2)&0x1;
    out[index + 5 * groupSz]  = (r2 >> 8) & 0x1;
    out[index + 6 * groupSz]  = (r2 >> 16) & 0x1;
    out[index + 7 * groupSz]  = (r2 >> 24) & 0x1;
    out[index + 8 * groupSz]  = (r3)&0x1;
    out[index + 9 * groupSz]  = (r3 >> 8) & 0x1;
    out[index + 10 * groupSz] = (r3 >> 16) & 0x1;
    out[index + 11 * groupSz] = (r3 >> 24) & 0x1;
    out[index + 12 * groupSz] = (r4)&0x1;
    out[index + 13 * groupSz] = (r4 >> 8) & 0x1;
    out[index + 14 * groupSz] = (r4 >> 16) & 0x1;
    out[index + 15 * groupSz] = (r4 >> 24) & 0x1;
}

static void writeOut128Bytes(short *out, const uint &index, const uint groupSz,
                             const uint &r1, const uint &r2, const uint &r3,
                             const uint &r4) {
    out[index]               = r1;
    out[index + groupSz]     = r1 >> 16;
    out[index + 2 * groupSz] = r2;
    out[index + 3 * groupSz] = r2 >> 16;
    out[index + 4 * groupSz] = r3;
    out[index + 5 * groupSz] = r3 >> 16;
    out[index + 6 * groupSz] = r4;
    out[index + 7 * groupSz] = r4 >> 16;
}

static void writeOut128Bytes(ushort *out, const uint &index, const uint groupSz,
                             const uint &r1, const uint &r2, const uint &r3,
                             const uint &r4) {
    writeOut128Bytes((short *)(out), index, groupSz, r1, r2, r3, r4);
}

static void writeOut128Bytes(int *out, const uint &index, const uint groupSz,
                             const uint &r1, const uint &r2, const uint &r3,
                             const uint &r4) {
    out[index]               = r1;
    out[index + groupSz]     = r2;
    out[index + 2 * groupSz] = r3;
    out[index + 3 * groupSz] = r4;
}

static void writeOut128Bytes(uint *out, const uint &index, const uint groupSz,
                             const uint &r1, const uint &r2, const uint &r3,
                             const uint &r4) {
    writeOut128Bytes((int *)(out), index, groupSz, r1, r2, r3, r4);
}

static void writeOut128Bytes(intl *out, const uint &index, const uint groupSz,
                             const uint &r1, const uint &r2, const uint &r3,
                             const uint &r4) {
    intl c1              = r2;
    c1                   = (c1 << 32) | r1;
    intl c2              = r4;
    c2                   = (c2 << 32) | r3;
    out[index]           = c1;
    out[index + groupSz] = c2;
}

static void writeOut128Bytes(uintl *out, const uint &index, const uint groupSz,
                             const uint &r1, const uint &r2, const uint &r3,
                             const uint &r4) {
    writeOut128Bytes((intl *)(out), index, groupSz, r1, r2, r3, r4);
}

static void writeOut128Bytes(float *out, const uint &index, const uint groupSz,
                             const uint &r1, const uint &r2, const uint &r3,
                             const uint &r4) {
    out[index]               = 1.f - getFloat01(r1);
    out[index + groupSz]     = 1.f - getFloat01(r2);
    out[index + 2 * groupSz] = 1.f - getFloat01(r3);
    out[index + 3 * groupSz] = 1.f - getFloat01(r4);
}

static void writeOut128Bytes(cfloat *out, const uint &index, const uint groupSz,
                             const uint &r1, const uint &r2, const uint &r3,
                             const uint &r4) {
    out[index]           = {1.f - getFloat01(r1), 1.f - getFloat01(r2)};
    out[index + groupSz] = {1.f - getFloat01(r3), 1.f - getFloat01(r4)};
}

static void writeOut128Bytes(double *out, const uint &index, const uint groupSz,
                             const uint &r1, const uint &r2, const uint &r3,
                             const uint &r4) {
    out[index]           = 1.0 - getDouble01(r1, r2);
    out[index + groupSz] = 1.0 - getDouble01(r3, r4);
}

static void writeOut128Bytes(cdouble *out, const uint &index,
                             const uint groupSz, const uint &r1, const uint &r2,
                             const uint &r3, const uint &r4) {
    out[index] = {1.0 - getDouble01(r1, r2), 1.0 - getDouble01(r3, r4)};
}

static void writeOut128Bytes(arrayfire::common::half *out, const uint &index,
                             const uint groupSz, const uint &r1, const uint &r2,
                             const uint &r3, const uint &r4) {
    // out[index]               = oneMinusGetHalf01(r1);
    // out[index + groupSz]     = oneMinusGetHalf01(r1 >> 16);
    // out[index + 2 * groupSz] = oneMinusGetHalf01(r2);
    // out[index + 3 * groupSz] = oneMinusGetHalf01(r2 >> 16);
    // out[index + 4 * groupSz] = oneMinusGetHalf01(r3);
    // out[index + 5 * groupSz] = oneMinusGetHalf01(r3 >> 16);
    // out[index + 6 * groupSz] = oneMinusGetHalf01(r4);
    // out[index + 7 * groupSz] = oneMinusGetHalf01(r4 >> 16);
}

// Normalized writes without boundary checking

static void boxMullerWriteOut128Bytes(float *out, const uint &index,
                                      const uint groupSz, const uint &r1,
                                      const uint &r2, const uint &r3,
                                      const uint &r4) {
    boxMullerTransform(&out[index], &out[index + groupSz],
                       getFloatNegative11(r1), getFloat01(r2));
    boxMullerTransform(&out[index + 2 * groupSz], &out[index + 3 * groupSz],
                       getFloatNegative11(r3), getFloat01(r4));
}

static void boxMullerWriteOut128Bytes(cfloat *out, const uint &index,
                                      const uint groupSz, const uint &r1,
                                      const uint &r2, const uint &r3,
                                      const uint &r4) {
    boxMullerTransform(&out[index], getFloatNegative11(r1), getFloat01(r2));
    boxMullerTransform(&out[index + groupSz], getFloatNegative11(r3),
                       getFloat01(r4));
}

static void boxMullerWriteOut128Bytes(double *out, const uint &index,
                                      const uint groupSz, const uint &r1,
                                      const uint &r2, const uint &r3,
                                      const uint &r4) {
    boxMullerTransform(&out[index], &out[index + groupSz],
                       getDoubleNegative11(r1, r2), getDouble01(r3, r4));
}

static void boxMullerWriteOut128Bytes(cdouble *out, const uint &index,
                                      const uint groupSz, const uint &r1,
                                      const uint &r2, const uint &r3,
                                      const uint &r4) {
    boxMullerTransform(&out[index], getDoubleNegative11(r1, r2),
                       getDouble01(r3, r4));
}

static void boxMullerWriteOut128Bytes(arrayfire::common::half *out,
                                      const uint &index, const uint groupSz,
                                      const uint &r1, const uint &r2,
                                      const uint &r3, const uint &r4) {
    //   boxMullerTransform(&out[index], &out[index + groupSz],
    //                      getHalfNegative11(r1), getHalf01(r1 >> 16));
    //   boxMullerTransform(&out[index + 2 * groupSz],
    //                      &out[index + 3 * groupSz], getHalfNegative11(r2),
    //                      getHalf01(r2 >> 16));
    //   boxMullerTransform(&out[index + 4 * groupSz],
    //                      &out[index + 5 * groupSz], getHalfNegative11(r3),
    //                      getHalf01(r3 >> 16));
    //   boxMullerTransform(&out[index + 6 * groupSz],
    //                      &out[index + 7 * groupSz], getHalfNegative11(r4),
    //                      getHalf01(r4 >> 16));
}

// Writes with boundary checking

static void partialWriteOut128Bytes(uchar *out, const uint &index,
                                    const uint groupSz, const uint &r1,
                                    const uint &r2, const uint &r3,
                                    const uint &r4, const uint &elements) {
    if (index < elements) { out[index] = r1; }
    if (index + groupSz < elements) { out[index + groupSz] = r1 >> 8; }
    if (index + 2 * groupSz < elements) { out[index + 2 * groupSz] = r1 >> 16; }
    if (index + 3 * groupSz < elements) { out[index + 3 * groupSz] = r1 >> 24; }
    if (index + 4 * groupSz < elements) { out[index + 4 * groupSz] = r2; }
    if (index + 5 * groupSz < elements) { out[index + 5 * groupSz] = r2 >> 8; }
    if (index + 6 * groupSz < elements) { out[index + 6 * groupSz] = r2 >> 16; }
    if (index + 7 * groupSz < elements) { out[index + 7 * groupSz] = r2 >> 24; }
    if (index + 8 * groupSz < elements) { out[index + 8 * groupSz] = r3; }
    if (index + 9 * groupSz < elements) { out[index + 9 * groupSz] = r3 >> 8; }
    if (index + 10 * groupSz < elements) {
        out[index + 10 * groupSz] = r3 >> 16;
    }
    if (index + 11 * groupSz < elements) {
        out[index + 11 * groupSz] = r3 >> 24;
    }
    if (index + 12 * groupSz < elements) { out[index + 12 * groupSz] = r4; }
    if (index + 13 * groupSz < elements) {
        out[index + 13 * groupSz] = r4 >> 8;
    }
    if (index + 14 * groupSz < elements) {
        out[index + 14 * groupSz] = r4 >> 16;
    }
    if (index + 15 * groupSz < elements) {
        out[index + 15 * groupSz] = r4 >> 24;
    }
}

static void partialWriteOut128Bytes(char *out, const uint &index,
                                    const uint groupSz, const uint &r1,
                                    const uint &r2, const uint &r3,
                                    const uint &r4, const uint &elements) {
    if (index < elements) { out[index] = (r1)&0x1; }
    if (index + groupSz < elements) { out[index + groupSz] = (r1 >> 8) & 0x1; }
    if (index + 2 * groupSz < elements) {
        out[index + 2 * groupSz] = (r1 >> 16) & 0x1;
    }
    if (index + 3 * groupSz < elements) {
        out[index + 3 * groupSz] = (r1 >> 24) & 0x1;
    }
    if (index + 4 * groupSz < elements) { out[index + 4 * groupSz] = (r2)&0x1; }
    if (index + 5 * groupSz < elements) {
        out[index + 5 * groupSz] = (r2 >> 8) & 0x1;
    }
    if (index + 6 * groupSz < elements) {
        out[index + 6 * groupSz] = (r2 >> 16) & 0x1;
    }
    if (index + 7 * groupSz < elements) {
        out[index + 7 * groupSz] = (r2 >> 24) & 0x1;
    }
    if (index + 8 * groupSz < elements) { out[index + 8 * groupSz] = (r3)&0x1; }
    if (index + 9 * groupSz < elements) {
        out[index + 9 * groupSz] = (r3 >> 8) & 0x1;
    }
    if (index + 10 * groupSz < elements) {
        out[index + 10 * groupSz] = (r3 >> 16) & 0x1;
    }
    if (index + 11 * groupSz < elements) {
        out[index + 11 * groupSz] = (r3 >> 24) & 0x1;
    }
    if (index + 12 * groupSz < elements) {
        out[index + 12 * groupSz] = (r4)&0x1;
    }
    if (index + 13 * groupSz < elements) {
        out[index + 13 * groupSz] = (r4 >> 8) & 0x1;
    }
    if (index + 14 * groupSz < elements) {
        out[index + 14 * groupSz] = (r4 >> 16) & 0x1;
    }
    if (index + 15 * groupSz < elements) {
        out[index + 15 * groupSz] = (r4 >> 24) & 0x1;
    }
}

static void partialWriteOut128Bytes(short *out, const uint &index,
                                    const uint groupSz, const uint &r1,
                                    const uint &r2, const uint &r3,
                                    const uint &r4, const uint &elements) {
    if (index < elements) { out[index] = r1; }
    if (index + groupSz < elements) { out[index + groupSz] = r1 >> 16; }
    if (index + 2 * groupSz < elements) { out[index + 2 * groupSz] = r2; }
    if (index + 3 * groupSz < elements) { out[index + 3 * groupSz] = r2 >> 16; }
    if (index + 4 * groupSz < elements) { out[index + 4 * groupSz] = r3; }
    if (index + 5 * groupSz < elements) { out[index + 5 * groupSz] = r3 >> 16; }
    if (index + 6 * groupSz < elements) { out[index + 6 * groupSz] = r4; }
    if (index + 7 * groupSz < elements) { out[index + 7 * groupSz] = r4 >> 16; }
}

static void partialWriteOut128Bytes(ushort *out, const uint &index,
                                    const uint groupSz, const uint &r1,
                                    const uint &r2, const uint &r3,
                                    const uint &r4, const uint &elements) {
    partialWriteOut128Bytes((short *)(out), index, groupSz, r1, r2, r3, r4,
                            elements);
}

static void partialWriteOut128Bytes(int *out, const uint &index,
                                    const uint groupSz, const uint &r1,
                                    const uint &r2, const uint &r3,
                                    const uint &r4, const uint &elements) {
    if (index < elements) { out[index] = r1; }
    if (index + groupSz < elements) { out[index + groupSz] = r2; }
    if (index + 2 * groupSz < elements) { out[index + 2 * groupSz] = r3; }
    if (index + 3 * groupSz < elements) { out[index + 3 * groupSz] = r4; }
}

static void partialWriteOut128Bytes(uint *out, const uint &index,
                                    const uint groupSz, const uint &r1,
                                    const uint &r2, const uint &r3,
                                    const uint &r4, const uint &elements) {
    partialWriteOut128Bytes((int *)(out), index, groupSz, r1, r2, r3, r4,
                            elements);
}

static void partialWriteOut128Bytes(intl *out, const uint &index,
                                    const uint groupSz, const uint &r1,
                                    const uint &r2, const uint &r3,
                                    const uint &r4, const uint &elements) {
    intl c1 = r2;
    c1      = (c1 << 32) | r1;
    intl c2 = r4;
    c2      = (c2 << 32) | r3;
    if (index < elements) { out[index] = c1; }
    if (index + groupSz < elements) { out[index + groupSz] = c2; }
}

static void partialWriteOut128Bytes(uintl *out, const uint &index,
                                    const uint groupSz, const uint &r1,
                                    const uint &r2, const uint &r3,
                                    const uint &r4, const uint &elements) {
    partialWriteOut128Bytes((intl *)(out), index, groupSz, r1, r2, r3, r4,
                            elements);
}

static void partialWriteOut128Bytes(float *out, const uint &index,
                                    const uint groupSz, const uint &r1,
                                    const uint &r2, const uint &r3,
                                    const uint &r4, const uint &elements) {
    if (index < elements) { out[index] = 1.f - getFloat01(r1); }
    if (index + groupSz < elements) {
        out[index + groupSz] = 1.f - getFloat01(r2);
    }
    if (index + 2 * groupSz < elements) {
        out[index + 2 * groupSz] = 1.f - getFloat01(r3);
    }
    if (index + 3 * groupSz < elements) {
        out[index + 3 * groupSz] = 1.f - getFloat01(r4);
    }
}

static void partialWriteOut128Bytes(cfloat *out, const uint &index,
                                    const uint groupSz, const uint &r1,
                                    const uint &r2, const uint &r3,
                                    const uint &r4, const uint &elements) {
    if (index < elements) {
        out[index] = {1.f - getFloat01(r1), 1.f - getFloat01(r2)};
    }
    if (index + groupSz < elements) {
        out[index + groupSz] = {1.f - getFloat01(r3), 1.f - getFloat01(r4)};
    }
}

static void partialWriteOut128Bytes(double *out, const uint &index,
                                    const uint groupSz, const uint &r1,
                                    const uint &r2, const uint &r3,
                                    const uint &r4, const uint &elements) {
    if (index < elements) { out[index] = 1.0 - getDouble01(r1, r2); }
    if (index + groupSz < elements) {
        out[index + groupSz] = 1.0 - getDouble01(r3, r4);
    }
}

static void partialWriteOut128Bytes(cdouble *out, const uint &index,
                                    const uint groupSz, const uint &r1,
                                    const uint &r2, const uint &r3,
                                    const uint &r4, const uint &elements) {
    if (index < elements) {
        out[index] = {1.0 - getDouble01(r1, r2), 1.0 - getDouble01(r3, r4)};
    }
}

// Normalized writes with boundary checking
static void partialBoxMullerWriteOut128Bytes(float *out, const uint &index,
                                             const uint groupSz, const uint &r1,
                                             const uint &r2, const uint &r3,
                                             const uint &r4,
                                             const uint &elements) {
    float n1, n2, n3, n4;
    boxMullerTransform(&n1, &n2, getFloatNegative11(r1), getFloat01(r2));
    boxMullerTransform(&n3, &n4, getFloatNegative11(r3), getFloat01(r4));
    if (index < elements) { out[index] = n1; }
    if (index + groupSz < elements) { out[index + groupSz] = n2; }
    if (index + 2 * groupSz < elements) { out[index + 2 * groupSz] = n3; }
    if (index + 3 * groupSz < elements) { out[index + 3 * groupSz] = n4; }
}

static void partialBoxMullerWriteOut128Bytes(cfloat *out, const uint &index,
                                             const uint groupSz, const uint &r1,
                                             const uint &r2, const uint &r3,
                                             const uint &r4,
                                             const uint &elements) {
    float n1, n2, n3, n4;
    boxMullerTransform(&n1, &n2, getFloatNegative11(r1), getFloat01(r2));
    boxMullerTransform(&n3, &n4, getFloatNegative11(r3), getFloat01(r4));
    if (index < elements) { out[index] = {n1, n2}; }
    if (index + groupSz < elements) { out[index + groupSz] = {n3, n4}; }
}

static void partialBoxMullerWriteOut128Bytes(double *out, const uint &index,
                                             const uint groupSz, const uint &r1,
                                             const uint &r2, const uint &r3,
                                             const uint &r4,
                                             const uint &elements) {
    double n1, n2;
    boxMullerTransform(&n1, &n2, getDoubleNegative11(r1, r2),
                       getDouble01(r3, r4));
    if (index < elements) { out[index] = n1; }
    if (index + groupSz < elements) { out[index + groupSz] = n2; }
}

static void partialBoxMullerWriteOut128Bytes(cdouble *out, const uint &index,
                                             const uint groupSz, const uint &r1,
                                             const uint &r2, const uint &r3,
                                             const uint &r4,
                                             const uint &elements) {
    double n1, n2;
    boxMullerTransform(&n1, &n2, getDoubleNegative11(r1, r2),
                       getDouble01(r3, r4));
    if (index < elements) { out[index] = {n1, n2}; }
}

static void partialWriteOut128Bytes(arrayfire::common::half *out,
                                    const uint &index, const uint groupSz,
                                    const uint &r1, const uint &r2,
                                    const uint &r3, const uint &r4,
                                    const uint &elements) {
    //  if (index < elements) { out[index] = oneMinusGetHalf01(r1); }
    //  if (index + groupSz < elements) {
    //      out[index + groupSz] = oneMinusGetHalf01(r1 >> 16);
    //  }
    //  if (index + 2 * groupSz < elements) {
    //      out[index + 2 * groupSz] = oneMinusGetHalf01(r2);
    //  }
    //  if (index + 3 * groupSz < elements) {
    //      out[index + 3 * groupSz] = oneMinusGetHalf01(r2 >> 16);
    //  }
    //  if (index + 4 * groupSz < elements) {
    //      out[index + 4 * groupSz] = oneMinusGetHalf01(r3);
    //  }
    //  if (index + 5 * groupSz < elements) {
    //      out[index + 5 * groupSz] = oneMinusGetHalf01(r3 >> 16);
    //  }
    //  if (index + 6 * groupSz < elements) {
    //      out[index + 6 * groupSz] = oneMinusGetHalf01(r4);
    //  }
    //  if (index + 7 * groupSz < elements) {
    //      out[index + 7 * groupSz] = oneMinusGetHalf01(r4 >> 16);
    //  }
}

// Normalized writes with boundary checking
static void partialBoxMullerWriteOut128Bytes(arrayfire::common::half *out,
                                             const uint &index,
                                             const uint groupSz, const uint &r1,
                                             const uint &r2, const uint &r3,
                                             const uint &r4,
                                             const uint &elements) {
    //    arrayfire::common::half n[8];
    //    boxMullerTransform(n + 0, n + 1, getHalfNegative11(r1),
    //                       getHalf01(r1 >> 16));
    //    boxMullerTransform(n + 2, n + 3, getHalfNegative11(r2),
    //                       getHalf01(r2 >> 16));
    //    boxMullerTransform(n + 4, n + 5, getHalfNegative11(r3),
    //                       getHalf01(r3 >> 16));
    //    boxMullerTransform(n + 6, n + 7, getHalfNegative11(r4),
    //                       getHalf01(r4 >> 16));
    //    if (index < elements) { out[index] = n[0]; }
    //    if (index + groupSz < elements) { out[index + groupSz] = n[1]; }
    //    if (index + 2 * groupSz < elements) {
    //        out[index + 2 * groupSz] = n[2];
    //    }
    //    if (index + 3 * groupSz < elements) {
    //        out[index + 3 * groupSz] = n[3];
    //    }
    //    if (index + 4 * groupSz < elements) {
    //        out[index + 4 * groupSz] = n[4];
    //    }
    //    if (index + 5 * groupSz < elements) {
    //        out[index + 5 * groupSz] = n[5];
    //    }
    //    if (index + 6 * groupSz < elements) {
    //        out[index + 6 * groupSz] = n[6];
    //    }
    //    if (index + 7 * groupSz < elements) {
    //        out[index + 7 * groupSz] = n[7];
    //    }
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire

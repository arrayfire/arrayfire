/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <err_cpu.hpp>
#include <kernel/random_engine_mersenne.hpp>
#include <kernel/random_engine_philox.hpp>
#include <kernel/random_engine_threefry.hpp>
#include <types.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

using std::array;
using std::memcpy;

namespace arrayfire {
namespace cpu {
namespace kernel {
// Utils
static const double PI_VAL =
    3.1415926535897932384626433832795028841971693993751058209749445923078164;

// Conversion to half adapted from Random123
constexpr float unsigned_half_factor =
    ((1.0f) / (std::numeric_limits<ushort>::max() + (1.0f)));
constexpr float unsigned_half_half_factor((0.5f) * unsigned_half_factor);

template<typename T>
T transform(uint *val, uint index);

template<>
uintl transform<uintl>(uint *val, uint index) {
    uint index2 = index << 1;
    uintl v     = ((static_cast<uintl>(val[index2]) << 32) |
               (static_cast<uintl>(val[index2 + 1])));
    return v;
}

// Generates rationals in [0, 1)
float getFloat01(uint *val, uint index) {
    // Conversion to floats adapted from Random123
    constexpr float factor =
        ((1.0f) /
         (static_cast<float>(std::numeric_limits<unsigned int>::max()) +
          (1.0f)));
    constexpr float half_factor = ((0.5f) * factor);
    return fmaf(val[index], factor, half_factor);
}

// Generates rationals in (-1, 1]
static float getFloatNegative11(uint *val, uint index) {
    // Conversion to floats adapted from Random123
    constexpr float factor =
        ((1.0) /
         (static_cast<double>(std::numeric_limits<int>::max()) + (1.0)));
    constexpr float half_factor = ((0.5f) * factor);

    return fmaf(static_cast<float>(val[index]), factor, half_factor);
}

// Generates rationals in [0, 1)
arrayfire::common::half getHalf01(uint *val, uint index) {
    float v = val[index >> 1U] >> (16U * (index & 1U)) & 0x0000ffff;
    return static_cast<arrayfire::common::half>(
        fmaf(v, unsigned_half_factor, unsigned_half_half_factor));
}

// Generates rationals in (-1, 1]
static arrayfire::common::half getHalfNegative11(uint *val, uint index) {
    float v = val[index >> 1U] >> (16U * (index & 1U)) & 0x0000ffff;
    // Conversion to half adapted from Random123
    constexpr float factor =
        ((1.0f) / (std::numeric_limits<short>::max() + (1.0f)));
    constexpr float half_factor = ((0.5f) * factor);

    return static_cast<arrayfire::common::half>(fmaf(v, factor, half_factor));
}

// Generates rationals in [0, 1)
double getDouble01(uint *val, uint index) {
    uintl v = transform<uintl>(val, index);
    constexpr double factor =
        ((1.0) / (std::numeric_limits<unsigned long long>::max() +
                  static_cast<long double>(1.0l)));
    constexpr double half_factor((0.5) * factor);
    return fma(v, factor, half_factor);
}

template<>
char transform<char>(uint *val, uint index) {
    char v = 0;
    memcpy(&v, static_cast<char *>(static_cast<void *>(val)) + index,
           sizeof(char));
    v &= 0x1;
    return v;
}

template<>
uchar transform<uchar>(uint *val, uint index) {
    uchar v = 0;
    memcpy(&v, static_cast<uchar *>(static_cast<void *>(val)) + index,
           sizeof(uchar));
    return v;
}

template<>
ushort transform<ushort>(uint *val, uint index) {
    ushort v = val[index >> 1U] >> (16U * (index & 1U)) & 0x0000ffff;
    return v;
}

template<>
short transform<short>(uint *val, uint index) {
    return transform<ushort>(val, index);
}

template<>
uint transform<uint>(uint *val, uint index) {
    return val[index];
}

template<>
int transform<int>(uint *val, uint index) {
    return transform<uint>(val, index);
}

template<>
intl transform<intl>(uint *val, uint index) {
    uintl v = transform<uintl>(val, index);
    intl out;
    memcpy(&out, &v, sizeof(intl));
    return v;
}

template<>
float transform<float>(uint *val, uint index) {
    return 1.f - getFloat01(val, index);
}

template<>
double transform<double>(uint *val, uint index) {
    return 1. - getDouble01(val, index);
}

template<>
arrayfire::common::half transform<arrayfire::common::half>(uint *val,
                                                           uint index) {
    float v = val[index >> 1U] >> (16U * (index & 1U)) & 0x0000ffff;
    return static_cast<arrayfire::common::half>(
        1.f - fmaf(v, unsigned_half_factor, unsigned_half_half_factor));
}

// Generates rationals in [-1, 1)
double getDoubleNegative11(uint *val, uint index) {
    intl v = transform<intl>(val, index);
    // Conversion to doubles adapted from Random123
    constexpr double signed_factor =
        ((1.0l) / (std::numeric_limits<long long>::max() + (1.0l)));
    constexpr double half_factor = ((0.5) * signed_factor);
    return fma(v, signed_factor, half_factor);
}

#define MAX_RESET_CTR_VAL 64
#define WRITE_STRIDE 256

// This implementation aims to emulate the corresponding method in the CUDA
// backend, in order to produce the exact same numbers as CUDA.
// A stride of WRITE_STRIDE (256) is applied between each write
// (emulating the CUDA thread writing to 4 locations with a stride of
// blockDim.x, which is 256).
// ELEMS_PER_ITER correspond to elementsPerBlock in the CUDA backend, so each
// "iter" (iteration) here correspond to a CUDA thread block doing its work.
// This change was prompted by issue #2429
template<typename T>
void philoxUniform(T *out, size_t elements, const uintl seed, uintl counter) {
    uint hi  = seed >> 32;
    uint lo  = seed;
    uint hic = counter >> 32;
    uint loc = counter;

    constexpr size_t RESET_CTR = MAX_RESET_CTR_VAL / sizeof(T);
    constexpr size_t ELEMS_PER_ITER =
        WRITE_STRIDE * 4 * sizeof(uint) / sizeof(T);

    int num_iters = divup(elements, ELEMS_PER_ITER);
    size_t len    = num_iters * ELEMS_PER_ITER;

    constexpr size_t NUM_WRITES = 16 / sizeof(T);
    for (size_t iter = 0; iter < len; iter += ELEMS_PER_ITER) {
        for (size_t i = 0; i < WRITE_STRIDE; i += RESET_CTR) {
            for (size_t j = 0; j < RESET_CTR; ++j) {
                // first_write_idx is the first of the 4 locations that will
                // be written to
                uintptr_t first_write_idx = iter + i + j;
                if (first_write_idx >= elements) { break; }

                // Recalculate key and ctr to emulate how the CUDA backend
                // calculates these per thread
                uint key[2] = {lo, hi};
                uint ctr[4] = {loc + (uint)first_write_idx, 0, 0, 0};
                ctr[1]      = hic + (ctr[0] < loc);
                ctr[2]      = (ctr[1] < hic);
                philox(key, ctr);

                // Use the same ctr array for each of the 4 locations,
                // but each of the location gets a different ctr value
                for (uint buf_idx = 0; buf_idx < NUM_WRITES; ++buf_idx) {
                    size_t out_idx = iter + buf_idx * WRITE_STRIDE + i + j;
                    if (out_idx < elements) {
                        out[out_idx] = transform<T>(ctr, buf_idx);
                    }
                }
            }
        }
    }
}

#undef MAX_RESET_CTR_VAL
#undef WRITE_STRIDE

template<typename T>
void threefryUniform(T *out, size_t elements, const uintl seed, uintl counter) {
    uint hi     = seed >> 32;
    uint lo     = seed;
    uint hic    = counter >> 32;
    uint loc    = counter;
    uint key[2] = {lo, hi};
    uint ctr[2] = {loc, hic};
    uint val[2];

    int reset = (2 * sizeof(uint)) / sizeof(T);
    for (int i = 0; i < (int)elements; i += reset) {
        threefry(key, ctr, val);
        ++ctr[0];
        ctr[1] += (ctr[0] == 0);
        int lim = (reset < (int)(elements - i)) ? reset : (int)(elements - i);
        for (int j = 0; j < lim; ++j) { out[i + j] = transform<T>(val, j); }
    }
}

template<typename T>
void boxMullerTransform(data_t<T> *const out1, data_t<T> *const out2,
                        const T r1, const T r2) {
    /*
     * The log of a real value x where 0 < x < 1 is negative.
     */
    using Tc = compute_t<T>;
    Tc r     = sqrt((Tc)(-2.0) * log(static_cast<Tc>(r2)));
    Tc theta = PI_VAL * (static_cast<Tc>(r1));

    *out1 = r * sin(theta);
    *out2 = r * cos(theta);
}

void boxMullerTransform(uint val[4], double *temp) {
    boxMullerTransform<double>(&temp[0], &temp[1], getDoubleNegative11(val, 0),
                               getDouble01(val, 1));
}

void boxMullerTransform(uint val[4], float *temp) {
    boxMullerTransform<float>(&temp[0], &temp[1], getFloatNegative11(val, 0),
                              getFloat01(val, 1));
    boxMullerTransform<float>(&temp[2], &temp[3], getFloatNegative11(val, 2),
                              getFloat01(val, 3));
}

void boxMullerTransform(uint val[4], arrayfire::common::half *temp) {
    using arrayfire::common::half;
    boxMullerTransform<half>(&temp[0], &temp[1], getHalfNegative11(val, 0),
                             getHalf01(val, 1));
    boxMullerTransform<half>(&temp[2], &temp[3], getHalfNegative11(val, 2),
                             getHalf01(val, 3));
    boxMullerTransform<half>(&temp[4], &temp[5], getHalfNegative11(val, 4),
                             getHalf01(val, 5));
    boxMullerTransform<half>(&temp[6], &temp[7], getHalfNegative11(val, 6),
                             getHalf01(val, 7));
}

template<typename T>
void philoxNormal(T *out, size_t elements, const uintl seed, uintl counter) {
    uint hi     = seed >> 32;
    uint lo     = seed;
    uint hic    = counter >> 32;
    uint loc    = counter;
    uint key[2] = {lo, hi};
    uint ctr[4] = {loc, hic, 0, 0};
    T temp[(4 * sizeof(uint)) / sizeof(T)];

    int reset = (4 * sizeof(uint)) / sizeof(T);
    for (int i = 0; i < (int)elements; i += reset) {
        philox(key, ctr);
        boxMullerTransform(ctr, temp);
        int lim = (reset < (int)(elements - i)) ? reset : (int)(elements - i);
        for (int j = 0; j < lim; ++j) { out[i + j] = temp[j]; }
    }
}

template<typename T>
void threefryNormal(T *out, size_t elements, const uintl seed, uintl counter) {
    uint hi     = seed >> 32;
    uint lo     = seed;
    uint hic    = counter >> 32;
    uint loc    = counter;
    uint key[2] = {lo, hi};
    uint ctr[2] = {loc, hic};
    uint val[4];
    T temp[(4 * sizeof(uint)) / sizeof(T)];

    int reset = (4 * sizeof(uint)) / sizeof(T);
    for (int i = 0; i < (int)elements; i += reset) {
        threefry(key, ctr, val);
        ++ctr[0];
        ctr[1] += (ctr[0] == 0);
        threefry(key, ctr, val + 2);
        ++ctr[0];
        ctr[1] += (ctr[0] == 0);
        boxMullerTransform(val, temp);
        int lim = (reset < (int)(elements - i)) ? reset : (int)(elements - i);
        for (int j = 0; j < lim; ++j) { out[i + j] = temp[j]; }
    }
}

template<typename T>
void uniformDistributionMT(T *out, size_t elements, uint *const state,
                           const uint *const pos, const uint *const sh1,
                           const uint *const sh2, uint mask,
                           const uint *const recursion_table,
                           const uint *const temper_table) {
    uint l_state[STATE_SIZE];
    uint o[4];
    uint lpos = pos[0];
    uint lsh1 = sh1[0];
    uint lsh2 = sh2[0];

    state_read(l_state, state);

    int reset = (4 * sizeof(uint)) / sizeof(T);
    for (int i = 0; i < (int)elements; i += reset) {
        mersenne(o, l_state, i, lpos, lsh1, lsh2, mask, recursion_table,
                 temper_table);
        int lim = (reset < (int)(elements - i)) ? reset : (int)(elements - i);
        for (int j = 0; j < lim; ++j) { out[i + j] = transform<T>(o, j); }
    }

    state_write(state, l_state);
}

template<typename T>
void normalDistributionMT(T *out, size_t elements, uint *const state,
                          const uint *const pos, const uint *const sh1,
                          const uint *const sh2, uint mask,
                          const uint *const recursion_table,
                          const uint *const temper_table) {
    T temp[(4 * sizeof(uint)) / sizeof(T)];
    uint l_state[STATE_SIZE];
    uint o[4];
    uint lpos = pos[0];
    uint lsh1 = sh1[0];
    uint lsh2 = sh2[0];

    state_read(l_state, state);

    int reset = (4 * sizeof(uint)) / sizeof(T);
    for (int i = 0; i < (int)elements; i += reset) {
        mersenne(o, l_state, i, lpos, lsh1, lsh2, mask, recursion_table,
                 temper_table);
        boxMullerTransform(o, temp);
        int lim = (reset < (int)(elements - i)) ? reset : (int)(elements - i);
        for (int j = 0; j < lim; ++j) { out[i + j] = temp[j]; }
    }

    state_write(state, l_state);
}

template<typename T>
void uniformDistributionCBRNG(T *out, size_t elements,
                              af_random_engine_type type, const uintl seed,
                              uintl counter) {
    switch (type) {
        case AF_RANDOM_ENGINE_PHILOX_4X32_10:
            philoxUniform(out, elements, seed, counter);
            break;
        case AF_RANDOM_ENGINE_THREEFRY_2X32_16:
            threefryUniform(out, elements, seed, counter);
            break;
        default:
            AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
    }
}

template<typename T>
void normalDistributionCBRNG(T *out, size_t elements,
                             af_random_engine_type type, const uintl seed,
                             uintl counter) {
    switch (type) {
        case AF_RANDOM_ENGINE_PHILOX_4X32_10:
            philoxNormal(out, elements, seed, counter);
            break;
        case AF_RANDOM_ENGINE_THREEFRY_2X32_16:
            threefryNormal(out, elements, seed, counter);
            break;
        default:
            AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire

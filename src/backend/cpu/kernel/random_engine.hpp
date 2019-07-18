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
#include <cstring>

using std::array;
using std::memcpy;

namespace cpu {
namespace kernel {
// Utils
static const double PI_VAL =
    3.1415926535897932384626433832795028841971693993751058209749445923078164;

// Conversion to half adapted from Random123
#define USHORTMAX 0xffff
#define HALF_FACTOR ((1.0f) / (USHORTMAX + (1.0f)))
#define HALF_HALF_FACTOR ((0.5f) * HALF_FACTOR)

// Conversion to floats adapted from Random123
#define UINTMAX 0xffffffff
#define FLT_FACTOR ((1.0f) / (UINTMAX + (1.0f)))
#define HALF_FLT_FACTOR ((0.5f) * FLT_FACTOR)

#define UINTLMAX 0xffffffffffffffff
#define DBL_FACTOR ((1.0) / (UINTLMAX + (1.0)))
#define HALF_DBL_FACTOR ((0.5) * DBL_FACTOR)

template<typename T>
T transform(uint *val, int index) {
    T *oval = (T *)val;
    return oval[index];
}

template<>
char transform<char>(uint *val, int index) {
    char v = val[index >> 2] >> (8 << (index & 3));
    v      = (v & 0x1) ? 1 : 0;
    return v;
}

template<>
uchar transform<uchar>(uint *val, int index) {
    uchar v = val[index >> 2] >> (index << 3);
    return v;
}

template<>
ushort transform<ushort>(uint *val, int index) {
    ushort v = val[index >> 1U] >> (16U * (index & 1U)) & 0x0000ffff;
    return v;
}

template<>
short transform<short>(uint *val, int index) {
    return transform<ushort>(val, index);
}

template<>
uint transform<uint>(uint *val, int index) {
    return val[index];
}

template<>
int transform<int>(uint *val, int index) {
    return transform<uint>(val, index);
}

template<>
uintl transform<uintl>(uint *val, int index) {
    uintl v = (((uintl)val[index << 1]) << 32) | ((uintl)val[(index << 1) + 1]);
    return v;
}

template<>
intl transform<intl>(uint *val, int index) {
    return transform<uintl>(val, index);
}

// Generates rationals in [0, 1)
template<>
float transform<float>(uint *val, int index) {
    return 1.f - (val[index] * FLT_FACTOR + HALF_FLT_FACTOR);
}

// Generates rationals in [0, 1)
template<>
common::half transform<common::half>(uint *val, int index) {
    float v = val[index >> 1U] >> (16U * (index & 1U)) & 0x0000ffff;
    return static_cast<common::half>(1.f - (v * HALF_FACTOR + HALF_HALF_FACTOR));
}

// Generates rationals in [0, 1)
template<>
double transform<double>(uint *val, int index) {
    uintl v = transform<uintl>(val, index);
    return 1.0 - (v * DBL_FACTOR + HALF_DBL_FACTOR);
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
                uint ctr[4] = {loc + (uint)first_write_idx,
                               hic + (ctr[0] < loc), (ctr[1] < hic), 0};
                philox(key, ctr);

                // Use the same ctr array for each of the 4 locations,
                // but each of the location gets a different ctr value
                for (size_t buf_idx = 0; buf_idx < NUM_WRITES; ++buf_idx) {
                    size_t out_idx = iter + buf_idx * WRITE_STRIDE + i + j;
                    if (out_idx < elements) {
                        out[out_idx] =
                            transform<T>(ctr, buf_idx);
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
        for (int j = 0; j < lim; ++j) {
            out[i + j] = transform<T>(val, j);
        }
    }
}

template<typename T>
void boxMullerTransform(data_t<T> *const out1, data_t<T> *const out2,
                        const compute_t<T> r1, const compute_t<T> r2) {
    /*
     * The log of a real value x where 0 < x < 1 is negative.
     */
    using Tc = compute_t<T>;
    Tc r     = sqrt((Tc)(-2.0) * log((Tc)(1.0) - r1));
    Tc theta = 2 * (Tc)PI_VAL * ((Tc)(1.0) - r2);
    *out1    = r * sin(theta);
    *out2    = r * cos(theta);
}

void boxMullerTransform(uint val[4], double *temp) {
    boxMullerTransform<double>(&temp[0], &temp[1], transform<double>(val, 0),
                               transform<double>(val, 1));
}

void boxMullerTransform(uint val[4], float *temp) {
    boxMullerTransform<float>(&temp[0], &temp[1], transform<float>(val, 0),
                              transform<float>(val, 1));
    boxMullerTransform<float>(&temp[2], &temp[3], transform<float>(val, 2),
                              transform<float>(val, 3));
}

void boxMullerTransform(uint val[4], common::half *temp) {
    using common::half;
    boxMullerTransform<half>(&temp[0], &temp[1], transform<half>(val, 0),
                             transform<half>(val, 1));
    boxMullerTransform<half>(&temp[2], &temp[3], transform<half>(val, 2),
                             transform<half>(val, 3));
    boxMullerTransform<half>(&temp[4], &temp[5], transform<half>(val, 4),
                             transform<half>(val, 5));
    boxMullerTransform<half>(&temp[6], &temp[7], transform<half>(val, 6),
                             transform<half>(val, 7));
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
        for (int j = 0; j < lim; ++j) {
            out[i + j] = transform<T>(o, j);
        }
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

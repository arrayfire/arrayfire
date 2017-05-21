/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <err_cpu.hpp>
#include <kernel/random_engine_philox.hpp>
#include <kernel/random_engine_threefry.hpp>
#include <kernel/random_engine_mersenne.hpp>

namespace cpu
{
namespace kernel
{
    //Utils
    static const double PI_VAL = 3.1415926535897932384626433832795028841971693993751058209749445923078164;

    //Conversion to floats adapted from Random123
    #define UINTMAX 0xffffffff
    #define FLT_FACTOR ((1.0f)/(UINTMAX + (1.0f)))
    #define HALF_FLT_FACTOR ((0.5f)*FLT_FACTOR)

    #define UINTLMAX 0xffffffffffffffff
    #define DBL_FACTOR ((1.0)/(UINTLMAX + (1.0)))
    #define HALF_DBL_FACTOR ((0.5)*DBL_FACTOR)

    template <typename T>
    T transform(uint *val, int index)
    {
        T *oval = (T*)val;
        return oval[index];
    }

    template <> char transform<char>(uint *val, int index)
    {
        char v = val[index>>2]>>(8<<(index & 3));
        v = (v&0x1) ? 1 : 0;
        return v;
    }

    template <> uchar transform<uchar>(uint *val, int index)
    {
        uchar v = val[index>>2]>>(8<<(index & 3));
        return v;
    }

    template <> ushort transform<ushort>(uint *val, int index)
    {
        ushort v = val[index>>1]>>(16<<(index & 1));
        return v;
    }

    template <> short transform<short>(uint *val, int index)
    {
        return transform<ushort>(val, index);
    }

    template <> uint transform<uint>(uint *val, int index)
    {
        return val[index];
    }

    template <> int transform<int>(uint *val, int index)
    {
        return transform<uint>(val, index);
    }

    template <> uintl transform<uintl>(uint *val, int index)
    {
        uintl v = (((uintl)val[index<<1])<<32) | ((uintl)val[(index<<1)+1]);
        return v;
    }

    template <> intl transform<intl>(uint *val, int index)
    {
        return transform<uintl>(val, index);
    }

    template <> float transform<float>(uint *val, int index)
    {
        return (val[index]*FLT_FACTOR + HALF_FLT_FACTOR);
    }

    template <> double transform<double>(uint *val, int index)
    {
        uintl v = transform<uintl>(val, index);
        return (v*DBL_FACTOR + HALF_DBL_FACTOR);
    }

    template <typename T>
    void philoxUniform(T* out, size_t elements, const uintl seed, uintl counter)
    {
        uint hi = seed>>32;
        uint lo = seed;
        uint key[2] = {(uint)counter, hi};
        uint ctr[4] = {(uint)counter, 0, 0, lo};

        int reset = (4*sizeof(uint))/sizeof(T);
        for (int i = 0; i < (int)elements; i += reset) {
            philox(key, ctr);
            int lim = (reset < (int)(elements - i))? reset : (int)(elements - i);
            for (int j = 0; j < lim; ++j) {
                out[i + j] = transform<T>(ctr, j);
            }
        }
    }

    template <typename T>
    void threefryUniform(T* out, size_t elements, const uintl seed, uintl counter)
    {
        uint hi = seed>>32;
        uint lo = seed;
        uint key[2] = {(uint)counter, hi};
        uint ctr[2] = {(uint)counter, lo};
        uint val[2];

        int reset = (2*sizeof(uint))/sizeof(T);
        for (int i = 0; i < (int)elements; i += reset) {
            threefry(key, ctr, val);
            ++ctr[0]; ++key[0];
            int lim = (reset < (int)(elements - i))? reset : (int)(elements - i);
            for (int j = 0; j < lim; ++j) {
                out[i + j] = transform<T>(val, j);
            }
        }
    }

    template <typename T>
    void boxMullerTransform(T * const out1, T * const out2, const T r1, const T r2)
    {
        /*
         * The log of a real value x where 0 < x < 1 is negative.
         */
        T r = sqrt((T)(-2.0) * log(r1));
        T theta = 2 * (T)PI_VAL * (r2);
        *out1 = r*sin(theta);
        *out2 = r*cos(theta);
    }

    void boxMullerTransform(uint val[4], double *temp)
    {
        boxMullerTransform(&temp[0], &temp[1], transform<double>(val, 0), transform<double>(val,1));
    }

    void boxMullerTransform(uint val[4], float *temp)
    {
        boxMullerTransform(&temp[0], &temp[1], transform<float>(val, 0), transform<float>(val, 1));
        boxMullerTransform(&temp[2], &temp[3], transform<float>(val, 2), transform<float>(val, 3));
    }

    template <typename T>
    void philoxNormal(T* out, size_t elements, const uintl seed, uintl counter)
    {
        uint hi = seed>>32;
        uint lo = seed;
        uint key[2] = {(uint)counter, hi};
        uint ctr[4] = {(uint)counter, 0, 0, lo};
        T temp[(4*sizeof(uint))/sizeof(T)];

        int reset = (4*sizeof(uint))/sizeof(T);
        for (int i = 0; i < (int)elements; i += reset) {
            philox(key, ctr);
            boxMullerTransform(ctr, temp);
            int lim = (reset < (int)(elements - i))? reset : (int)(elements - i);
            for (int j = 0; j < lim; ++j) {
                out[i + j] = temp[j];
            }
        }
    }

    template <typename T>
    void threefryNormal(T* out, size_t elements, const uintl seed, uintl counter)
    {
        uint hi = seed>>32;
        uint lo = seed;
        uint key[2] = {(uint)counter, hi};
        uint ctr[2] = {(uint)counter, lo};
        uint val[4];
        T temp[(4*sizeof(uint))/sizeof(T)];

        int reset = (4*sizeof(uint))/sizeof(T);
        for (int i = 0; i < (int)elements; i += reset) {
            threefry(key, ctr, val);
            ++ctr[0]; ++key[0];
            threefry(key, ctr, val+2);
            ++ctr[0]; ++key[0];
            boxMullerTransform(val, temp);
            int lim = (reset < (int)(elements - i))? reset : (int)(elements - i);
            for (int j = 0; j < lim; ++j) {
                out[i + j] = temp[j];
            }
        }
    }

    template <typename T>
    void uniformDistributionMT(T* out, size_t elements,
            uint * const state,
            const uint * const pos,
            const uint * const sh1,
            const uint * const sh2,
            uint mask,
            const uint * const recursion_table,
            const uint * const temper_table)
    {
        uint l_state[STATE_SIZE];
        uint o[4];
        uint lpos = pos[0];
        uint lsh1 = sh1[0];
        uint lsh2 = sh2[0];

        state_read(l_state, state);

        int reset = (4*sizeof(uint))/sizeof(T);
        for (int i = 0; i < (int)elements; i += reset) {
            mersenne(o, l_state, i, lpos, lsh1, lsh2, mask, recursion_table, temper_table);
            int lim = (reset < (int)(elements - i))? reset : (int)(elements - i);
            for (int j = 0; j < lim; ++j) {
                out[i + j] = transform<T>(o, j);
            }
        }

        state_write(state, l_state);
    }

    template <typename T>
    void normalDistributionMT(T* out, size_t elements,
            uint * const state,
            const uint * const pos,
            const uint * const sh1,
            const uint * const sh2,
            uint mask,
            const uint * const recursion_table,
            const uint * const temper_table)
    {
        T temp[(4*sizeof(uint))/sizeof(T)];
        uint l_state[STATE_SIZE];
        uint o[4];
        uint lpos = pos[0];
        uint lsh1 = sh1[0];
        uint lsh2 = sh2[0];

        state_read(l_state, state);

        int reset = (4*sizeof(uint))/sizeof(T);
        for (int i = 0; i < (int)elements; i += reset) {
            mersenne(o, l_state, i, lpos, lsh1, lsh2, mask, recursion_table, temper_table);
            boxMullerTransform(o, temp);
            int lim = (reset < (int)(elements - i))? reset : (int)(elements - i);
            for (int j = 0; j < lim; ++j) {
                out[i + j] = temp[j];
            }
        }

        state_write(state, l_state);
    }

    template <typename T>
    void uniformDistributionCBRNG(T* out, size_t elements, af_random_engine_type type, const uintl seed, uintl counter)
    {
        switch(type) {
            case AF_RANDOM_ENGINE_PHILOX_4X32_10   :   philoxUniform(out, elements, seed, counter); break;
            case AF_RANDOM_ENGINE_THREEFRY_2X32_16 : threefryUniform(out, elements, seed, counter); break;
            default : AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

    template <typename T>
    void normalDistributionCBRNG(T* out, size_t elements, af_random_engine_type type, const uintl seed, uintl counter)
    {
        switch(type) {
            case AF_RANDOM_ENGINE_PHILOX_4X32_10   :   philoxNormal(out, elements, seed, counter); break;
            case AF_RANDOM_ENGINE_THREEFRY_2X32_16 : threefryNormal(out, elements, seed, counter); break;
            default : AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

}
}

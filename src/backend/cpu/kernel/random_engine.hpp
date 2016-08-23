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
    static const float UINTMAXFLOAT = 4294967296.0f;
    static const float UINTLMAXDOUBLE = (4294967296.0*4294967296.0);
    static const double PI_VAL = 3.1415926535897932384626433832795028841971693993751058209749445923078164;

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
        return (float)val[index]/UINTMAXFLOAT;
    }

    template <> double transform<double>(uint *val, int index)
    {
        uintl v = transform<uintl>(val, index);
        return (double)v/UINTLMAXDOUBLE;
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
    void normalizePair(T * const out1, T * const out2, const T r1, const T r2)
    {
        T r = sqrt((T)(-2.0) * log(r1));
        T theta = 2 * (T)PI_VAL * (r2);
        *out1 = r*sin(theta);
        *out2 = r*cos(theta);
    }

    void normalize(uint val[4], double *temp)
    {
        normalizePair(&temp[0], &temp[1], transform<double>(val, 0), transform<double>(val,1));
    }

    void normalize(uint val[4], float *temp)
    {
        normalizePair(&temp[0], &temp[1], transform<float>(val, 0), transform<float>(val, 1));
        normalizePair(&temp[2], &temp[3], transform<float>(val, 2), transform<float>(val, 3));
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
            normalize(ctr, temp);
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
            normalize(val, temp);
            int lim = (reset < (int)(elements - i))? reset : (int)(elements - i);
            for (int j = 0; j < lim; ++j) {
                out[i + j] = temp[j];
            }
        }
    }

    template <typename T>
    void uniformDistributionMT(T* out, size_t elements,
            uint * const state,
            uint const * const pos,
            uint const * const sh1,
            uint const * const sh2,
            uint mask,
            uint const * const recursion_table,
            uint const * const temper_table)
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
            uint const * const pos,
            uint const * const sh1,
            uint const * const sh2,
            uint mask,
            uint const * const recursion_table,
            uint const * const temper_table)
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
            normalize(o, temp);
            int lim = (reset < (int)(elements - i))? reset : (int)(elements - i);
            for (int j = 0; j < lim; ++j) {
                out[i + j] = temp[j];
            }
        }

        state_write(state, l_state);
    }

    template <typename T>
    void uniformDistributionCBRNG(T* out, size_t elements, af_random_type type, const uintl seed, uintl counter)
    {
        switch(type) {
            case AF_RANDOM_PHILOX   :   philoxUniform(out, elements, seed, counter); break;
            case AF_RANDOM_THREEFRY : threefryUniform(out, elements, seed, counter); break;
            default : AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

    template <typename T>
    void normalDistributionCBRNG(T* out, size_t elements, af_random_type type, const uintl seed, uintl counter)
    {
        switch(type) {
            case AF_RANDOM_PHILOX   :   philoxNormal(out, elements, seed, counter); break;
            case AF_RANDOM_THREEFRY : threefryNormal(out, elements, seed, counter); break;
            default : AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

    void initMersenneState(uint * const state, const uint * const tbl, const uintl seed)
    {
        uint hidden_seed = tbl[4] ^ (tbl[8] << 16);
        uint tmp = hidden_seed;
        tmp += tmp >> 16;
        tmp += tmp >> 8;
        tmp &= 0xff;
        tmp |= tmp << 8;
        tmp |= tmp << 16;
        state[0] = seed;
        state[1] = hidden_seed ^ ((uint)(1812433253) * (state[0] ^ (state[0] >> 30)) + 1);
        for (int i = 2; i < N; ++i) {
            state[i] = tmp;
            state[i] ^= (uint)(1812433253) * (state[i-1] ^ (state[i-1] >> 30)) + i;
        }
    }
}
}

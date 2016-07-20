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
#include <kernel/random_engine_philox.hpp>
#include <kernel/random_engine_threefry.hpp>

namespace cpu
{
namespace kernel
{
    //Utils
    #define UINTMAXFLOAT 4294967296.0f
    #define UINTLMAXDOUBLE (4294967296.0*4294967296.0)
    #define PI_VAL 3.1415926535897932384626433832795028841971693993751058209749445923078164

    template <typename T>
    T transform(uint *val, int index)
    {
        T *oval = (T*)val;
        return oval[index];
    }

    template <> char transform<char>(uint *val, int index)
    {
        char v = transform<char>(val, index);
        v = (v&0x1) ? 1 : 0;
        return v;
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
    void philoxUniform(T* out, size_t elements, const uintl seed, uintl &counter)
    {
        uint hi = seed>>32;
        uint lo = seed;
        uint key[2] = {(uint)counter, hi};
        uint ctr[4] = {(uint)counter, 0, 0, lo};

        int fresh = 0;
        unsigned reset = (4*sizeof(uint))/sizeof(T);
        philox(key, ctr);
        ++ctr[0]; ++key[0];
        for (int i = 0; i < (int)elements; ++i) {
            if (fresh == reset) {
                philox(key, ctr);
                ++ctr[0]; ++key[0];
                fresh = 0;
            }
            out[i] = transform<T>(ctr, fresh);
            fresh++;
            ////philox(key, ctr);
            ////++ctr[0]; ++key[0];
            ////int lim = (reset < elements - i)? reset : elements - i;
            ////for (int j = 0; j < lim; ++j) {
            ////    out[i + j] = transform<T>(ctr, j);
            ////}
        }
    }

    template <typename T>
    void threefryUniform(T* out, size_t elements, const uintl seed, uintl &counter)
    {
        uint hi = seed>>32;
        uint lo = seed;
        uint key[2] = {(uint)counter, hi};
        uint ctr[2] = {(uint)counter, lo};
        uint val[2];

        int fresh = 0;
        unsigned reset = (2*sizeof(uint))/sizeof(T);
        threefry(key, ctr, val);
        ++ctr[0]; ++key[0];
        for (int i = 0; i < (int)elements; ++i) {
            if (fresh == reset) {
                threefry(key, ctr, val);
                ++ctr[0]; ++key[0];
                fresh = 0;
            }
            out[i] = transform<T>(val, fresh);
            fresh++;
            ////threefry(key, ctr, val);
            ////++ctr[0]; ++key[0];
            ////int lim = (reset < elements - i)? reset : elements - i;
            ////for (int j = 0; j < lim; ++j) {
            ////    out[i + j] = transform<T>(ctr, j);
            ////}
        }
    }

    template <typename T, af_random_type Type>
    void uniformDistribution(T* out, size_t elements, const uintl seed, uintl &counter)
    {
        switch(Type) {
            case AF_RANDOM_PHILOX   :   philoxUniform(out, elements, seed, counter); break;
            case AF_RANDOM_THREEFRY : threefryUniform(out, elements, seed, counter); break;
        }
    }

    template <typename T>
    void normalizePair(T * const out1, T * const out2, const T r1, const T r2)
    {
#if defined(IS_APPLE) // Because Apple is.. "special"
        T r = sqrt((T)(-2.0) * log10(r1) * (float)log10_val);
#else
        T r = sqrt((T)(-2.0) * log(r1));
#endif
        T theta = 2 * (T)PI_VAL * (r2);
        *out1 = r*sin(theta);
        *out2 = r*cos(theta);
    }

    void normalize(uint val[4], double *temp)
    {
        uintl *v = (uintl*)val;
        normalizePair(&temp[0], &temp[1], v[0]/UINTLMAXDOUBLE, v[1]/UINTLMAXDOUBLE);
    }

    void normalize(uint val[4], float *temp)
    {
        normalizePair(&temp[0], &temp[1], val[0]/UINTMAXFLOAT, val[1]/UINTMAXFLOAT);
        normalizePair(&temp[2], &temp[3], val[2]/UINTMAXFLOAT, val[3]/UINTMAXFLOAT);
    }

    template <typename T>
    void threefryNormal(T* out, size_t elements, const uintl seed, uintl &counter)
    {
        uint hi = seed>>32;
        uint lo = seed;
        uint key[2] = {(uint)counter, hi};
        uint ctr[2] = {(uint)counter, lo};
        uint val[4];
        T temp[(4*sizeof(uint))/sizeof(T)];

        int fresh = 0;
        int reset = (4*sizeof(uint))/sizeof(T);
        threefry(key, ctr, val);
        normalize(val, temp);
        ++ctr[0]; ++key[0];
        for (int i = 0; i < (int)elements; ++i) {
            if (fresh == reset) {
                threefry(key, ctr, val);
                ++ctr[0]; ++key[0];
                threefry(key, ctr, val+2);
                fresh = 0;
                normalize(val, temp);
            }
            out[i] = temp[fresh];
            fresh++;
        }
    }

    template <typename T>
    void philoxNormal(T* out, size_t elements, const uintl seed, uintl &counter)
    {
        uint hi = seed>>32;
        uint lo = seed;
        uint key[2] = {(uint)counter, hi};
        uint ctr[4] = {(uint)counter, 0, 0, lo};
        T temp[(4*sizeof(uint))/sizeof(T)];

        int fresh = 0;
        int reset = (4*sizeof(uint))/sizeof(T);
        philox(key, ctr);
        normalize(ctr, temp);
        for (int i = 0; i < (int)elements; ++i) {
            if (fresh == reset) {
                philox(key, ctr);
                ++ctr[0]; ++key[0];
                fresh = 0;
                normalize(ctr, temp);
            }
            out[i] = temp[fresh];
            fresh++;
        }
    }

    template <typename T, af_random_type Type>
    void normalDistribution(T* out, size_t elements, const uintl seed, uintl &counter)
    {
        switch(Type) {
            case AF_RANDOM_PHILOX   :   philoxNormal(out, elements, seed, counter); break;
            case AF_RANDOM_THREEFRY : threefryNormal(out, elements, seed, counter); break;
        }
    }

}
}

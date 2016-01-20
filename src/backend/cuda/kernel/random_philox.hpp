/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 *
 ********************************************************/

/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

namespace cuda
{
namespace kernel
{
    template <typename T, unsigned int W>
    struct PhiloxCounter
    {
        T v[W];
    };

    template <typename T, unsigned int W>
    struct PhiloxKey
    {
        T v[W/2];
    };

    static const uintl m2x64_0 = uintl(0xD2B74407B1CE6E93);
    static const uintl m4x64_0 = uintl(0xD2E7470EE14C6C93);
    static const uintl m4x64_1 = uintl(0xCA5A826395121157);
    static const uint m2x32_0 = uint(0xD256D193);
    static const uint m4x32_0 = uint(0xD2511F53);
    static const uint m4x32_1 = uint(0xCD9E8D57);
    static const uintl w64_0 = uintl(0x9E3779B97F4A7C15);
    static const uintl w64_1 = uintl(0xBB67AE8584CAA73B);
    static const uint w32_0 = uint(0x9E3779B9);
    static const uint w32_1 = uint(0xBB67AE85);
    static const unsigned int PhiloxDefaultRounds = 10;
    #define UINTMAXFLOAT 4294967296.0f
    #define UINTLMAXDOUBLE 4294967296.0*4294967296.0

    static inline __device__ void mulhilo(const uint &a, const uint &b,
            uint &hi, uint &lo)
    {
        hi = __umulhi(a,b);
        lo = a*b;
    }

    static inline __device__ void mulhilo(const uintl &a, const uintl &b,
            uintl &hi, uintl &lo)
    {
        hi = __umul64hi(a,b);
        lo = a*b;
    }

    template <typename T, T mul>
    static inline __device__ void phRound(PhiloxCounter<T, 2> &ctr,
            const PhiloxKey<T, 2> &key)
    {
        T hi, lo;
        mulhilo(mul, ctr.v[0], hi, lo);
        ctr.v[0] = hi^key.v[0]^ctr.v[1];
        ctr.v[1] = lo;
    }

    template <typename T, T mul0, T mul1>
    static inline __device__ void phRound(PhiloxCounter<T, 4> &ctr,
            const PhiloxKey<T, 4> &key)
    {
        T hi0, lo0, hi1, lo1;
        mulhilo(mul0, ctr.v[0], hi0, lo0);
        mulhilo(mul1, ctr.v[2], hi1, lo1);
        ctr.v[0] = hi1^ctr.v[1]^key.v[0];
        ctr.v[1] = lo1;
        ctr.v[2] = hi0^ctr.v[3]^key.v[1];
        ctr.v[3] = lo0;
    }

    static inline __device__ void philoxRound(PhiloxCounter<uint, 4> &ctr,
            const PhiloxKey<uint, 4> &key)
    {
        phRound<uint, m4x32_0, m4x32_1>(ctr, key);
    }

    static inline __device__ void philoxRound(PhiloxCounter<uintl, 4> &ctr,
            const PhiloxKey<uintl, 4> &key)
    {
        phRound<uintl, m4x64_0, m4x64_1>(ctr, key);
    }

    static inline __device__ void philoxRound(PhiloxCounter<uint, 2> &ctr,
            const PhiloxKey<uint, 2> &key)
    {
        phRound<uint, m2x32_0>(ctr, key);
    }

    static inline __device__ void philoxRound(PhiloxCounter<uintl, 2> &ctr,
            const PhiloxKey<uintl, 2> &key)
    {
        phRound<uintl, m2x64_0>(ctr, key);
    }

    static inline __device__ void philoxBump(PhiloxKey<uint, 2> &key)
    {
        key.v[0] += w32_0;
    }

    static inline __device__ void philoxBump(PhiloxKey<uintl, 2> &key)
    {
        key.v[0] += w64_0;
    }

    static inline __device__ void philoxBump(PhiloxKey<uint, 4> &key)
    {
        key.v[0] += w32_0;
        key.v[1] += w32_1;
    }

    static inline __device__ void philoxBump(PhiloxKey<uintl, 4> &key)
    {
        key.v[0] += w64_0;
        key.v[1] += w64_1;
    }

    template <typename T, unsigned int W, unsigned int R>
    static inline __device__ PhiloxCounter<T,W> philox(const PhiloxCounter<T,W> &inputCtr,
            const PhiloxKey<T,W> &inputKey)
    {
        PhiloxCounter<T,W> ctr = inputCtr;
        PhiloxKey<T,W> key = inputKey;
        if (R>0)    {                   philoxRound(ctr, key);}
        if (R>1)    {philoxBump(key);   philoxRound(ctr, key);}
        if (R>2)    {philoxBump(key);   philoxRound(ctr, key);}
        if (R>3)    {philoxBump(key);   philoxRound(ctr, key);}
        if (R>4)    {philoxBump(key);   philoxRound(ctr, key);}
        if (R>5)    {philoxBump(key);   philoxRound(ctr, key);}
        if (R>6)    {philoxBump(key);   philoxRound(ctr, key);}
        if (R>7)    {philoxBump(key);   philoxRound(ctr, key);}
        if (R>8)    {philoxBump(key);   philoxRound(ctr, key);}
        if (R>9)    {philoxBump(key);   philoxRound(ctr, key);}
        if (R>10)   {philoxBump(key);   philoxRound(ctr, key);}
        if (R>11)   {philoxBump(key);   philoxRound(ctr, key);}
        if (R>12)   {philoxBump(key);   philoxRound(ctr, key);}
        if (R>13)   {philoxBump(key);   philoxRound(ctr, key);}
        if (R>14)   {philoxBump(key);   philoxRound(ctr, key);}
        if (R>15)   {philoxBump(key);   philoxRound(ctr, key);}
        return ctr;
    }

    template <typename T, unsigned int W>
    __device__ PhiloxCounter<T,W> philox(const PhiloxCounter<T,W> &ctr,
            const PhiloxKey<T,W> &key)
    {
        return philox<T,W,PhiloxDefaultRounds>(ctr, key);
    }

    __device__ static float normalizeToFloat(const uint &num)
    {
        return float(num)/UINTMAXFLOAT;
    }

    __device__ static double normalizeToDouble(const uintl &num)
    {
        return double(num)/UINTLMAXDOUBLE;
    }

#define writeOut16(T)                                               \
    __device__ static void writeOut(T *out, const unsigned &index,  \
            const PhiloxCounter<uint, 4> &counter)                  \
    {                                                               \
        out[index]                  = (counter.v[0]&0x00001111);    \
        out[index + blockDim.x]     = (counter.v[0]>>4);            \
        out[index + 2*blockDim.x]   = (counter.v[1]&0x00001111);    \
        out[index + 3*blockDim.x]   = (counter.v[1]>>4);            \
        out[index + 4*blockDim.x]   = (counter.v[2]&0x00001111);    \
        out[index + 5*blockDim.x]   = (counter.v[2]>>4);            \
        out[index + 6*blockDim.x]   = (counter.v[3]&0x00001111);    \
        out[index + 7*blockDim.x]   = (counter.v[3]>>4);            \
    }

#define writeOut32(T)                                                   \
    __device__ static void writeOut(T *out, const unsigned &index,      \
            const PhiloxCounter<uint, 4> &counter)                      \
    {                                                                   \
        out[index]                  = counter.v[0];                     \
        out[index + blockDim.x]     = counter.v[1];                     \
        out[index + 2*blockDim.x]   = counter.v[2];                     \
        out[index + 3*blockDim.x]   = counter.v[3];                     \
    }

#define writeOut64(T)                                                               \
    __device__ static void writeOut(T *out, const unsigned &index,                  \
            const PhiloxCounter<uint, 4> &counter)                                  \
    {                                                                               \
        out[index]              = (uintl(counter.v[0])<<32) | uintl(counter.v[1]);  \
        out[index + blockDim.x] = (uintl(counter.v[2])<<32) | uintl(counter.v[3]);  \
    }

    writeOut16(ushort);
    writeOut16(short);
    writeOut32(uint);
    writeOut32(int);
    writeOut64(uintl);
    writeOut64(intl);

    __device__ static void writeOut(float *out, const unsigned &index,
            const PhiloxCounter<uint, 4> &counter)
    {
        out[index]                  = normalizeToFloat(counter.v[0]);
        out[index + blockDim.x]     = normalizeToFloat(counter.v[1]);
        out[index + 2*blockDim.x]   = normalizeToFloat(counter.v[2]);
        out[index + 3*blockDim.x]   = normalizeToFloat(counter.v[3]);
    }

    __device__ static void writeOut(double *out, const unsigned &index,
            const PhiloxCounter<uint, 4> &counter)
    {
        out[index]              = normalizeToDouble((uintl(counter.v[0])<<32) | uintl(counter.v[1]));
        out[index + blockDim.x] = normalizeToDouble((uintl(counter.v[2])<<32) | uintl(counter.v[3]));
    }

    __device__ static void writeOut(cfloat *out, const unsigned &index,
            const PhiloxCounter<uint, 4> &counter)
    {
        out[index].x               =   normalizeToFloat(counter.v[0]);
        out[index].y               =   normalizeToFloat(counter.v[1]);
        out[index + blockDim.x].x  =   normalizeToFloat(counter.v[2]);
        out[index + blockDim.x].y  =   normalizeToFloat(counter.v[3]);
    }

    __device__ static void writeOut(cdouble *out, const unsigned &index,
            const PhiloxCounter<uint, 4> &counter)
    {
        out[index].x   =   normalizeToDouble((uintl(counter.v[0])<<32) | uintl(counter.v[1]));
        out[index].y   =   normalizeToDouble((uintl(counter.v[2])<<32) | uintl(counter.v[3]));
    }

    __device__ static void writeOut(char *out, const unsigned &index,
            const PhiloxCounter<uint, 4> &counter)
    {
        for (unsigned tid = index, i = 0; (i < 16); tid += blockDim.x, ++i) {
            out[tid] = (counter.v[i>>2] & (1 << (i & 3)))? 1:0;
        }
    }

    __device__ static void writeOut(uchar *out, const unsigned &index,
            const PhiloxCounter<uint, 4> &counter)
    {
        for (unsigned tid = index, i = 0; (i < 16); tid += blockDim.x, ++i) {
            out[tid] = (counter.v[i>>2] >> ((i & 3) << 1)) & 3;
        }
    }

#define writeOutPartial16(T)                                                                                \
    __device__ static void writeOutPartial(T *out, const unsigned &index,                                   \
            const size_t &elements, const PhiloxCounter<uint, 4> &counter)                                  \
    {                                                                                                       \
         if (index                  < elements) {out[index]                  = (counter.v[0]&0x00001111);}  \
         if (index + blockDim.x     < elements) {out[index + blockDim.x]     = (counter.v[0]>>4);}          \
         if (index + 2*blockDim.x   < elements) {out[index + 2*blockDim.x]   = (counter.v[1]&0x00001111);}  \
         if (index + 3*blockDim.x   < elements) {out[index + 3*blockDim.x]   = (counter.v[1]>>4);}          \
         if (index + 4*blockDim.x   < elements) {out[index + 4*blockDim.x]   = (counter.v[2]&0x00001111);}  \
         if (index + 5*blockDim.x   < elements) {out[index + 5*blockDim.x]   = (counter.v[2]>>4);}          \
         if (index + 6*blockDim.x   < elements) {out[index + 6*blockDim.x]   = (counter.v[3]&0x00001111);}  \
         if (index + 7*blockDim.x   < elements) {out[index + 7*blockDim.x]   = (counter.v[3]>>4);}          \
    }

#define writeOutPartial32(T)                                                                    \
    __device__ static void writeOutPartial(T *out, const unsigned &index,                       \
            const size_t &elements, const PhiloxCounter<uint, 4> &counter)                      \
    {                                                                                           \
        if (index                   < elements) {out[index]                  = counter.v[0];}   \
        if (index + blockDim.x      < elements) {out[index + blockDim.x]     = counter.v[1];}   \
        if (index + 2*blockDim.x    < elements) {out[index + 2*blockDim.x]   = counter.v[2];}   \
        if (index + 3*blockDim.x    < elements) {out[index + 3*blockDim.x]   = counter.v[3];}   \
    }

#define writeOutPartial64(T)                                                                                            \
    __device__ static void writeOutPartial(T *out, const unsigned &index,                                               \
            const size_t &elements, const PhiloxCounter<uint, 4> &counter)                                              \
    {                                                                                                                   \
        if (index               < elements) {out[index]              = (uintl(counter.v[0])<<32) | uintl(counter.v[1]);}\
        if (index + blockDim.x  < elements) {out[index + blockDim.x] = (uintl(counter.v[2])<<32) | uintl(counter.v[3]);}\
    }

    writeOutPartial16(ushort);
    writeOutPartial16(short);
    writeOutPartial32(uint);
    writeOutPartial32(int);
    writeOutPartial64(uintl);
    writeOutPartial64(intl);

    __device__ static void writeOutPartial(float *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4> &counter)
    {
        if (index                   < elements) {out[index]                  = normalizeToFloat(counter.v[0]);}
        if (index + blockDim.x      < elements) {out[index + blockDim.x]     = normalizeToFloat(counter.v[1]);}
        if (index + 2*blockDim.x    < elements) {out[index + 2*blockDim.x]   = normalizeToFloat(counter.v[2]);}
        if (index + 3*blockDim.x    < elements) {out[index + 3*blockDim.x]   = normalizeToFloat(counter.v[3]);}
    }

    __device__ static void writeOutPartial(double *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4> &counter)
    {
        if (index               < elements)
            {out[index]              = normalizeToDouble((uintl(counter.v[0])<<32) | uintl(counter.v[1]));}
        if (index + blockDim.x  < elements)
            {out[index + blockDim.x] = normalizeToDouble((uintl(counter.v[2])<<32) | uintl(counter.v[3]));}
    }

    __device__ static void writeOutPartial(cfloat *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4> &counter)
    {
        if (index               < elements) {
            out[index].x               =   normalizeToFloat(counter.v[0]);
            out[index].y               =   normalizeToFloat(counter.v[1]);}
        if (index + blockDim.x  < elements) {
            out[index + blockDim.x].x  =   normalizeToFloat(counter.v[2]);
            out[index + blockDim.x].y  =   normalizeToFloat(counter.v[2]);}
    }

    __device__ static void writeOutPartial(cdouble *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4> &counter)
    {
        if (index < elements) {
            out[index].x   =   normalizeToDouble((uintl(counter.v[0])<<32) | uintl(counter.v[1]));
            out[index].y   =   normalizeToDouble((uintl(counter.v[2])<<32) | uintl(counter.v[3]));}
    }

    __device__ static void writeOutPartial(char *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4> &counter)
    {
        for (unsigned tid = index, i = 0; (i < 16) && (tid < elements); tid += blockDim.x, ++i) {
            out[tid] = (counter.v[i>>2] & (1 << (i & 3)))? 1:0;
        }
    }

    __device__ static void writeOutPartial(uchar *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4> &counter)
    {
        for (unsigned tid = index, i = 0; (i < 16) && (tid < elements); tid += blockDim.x, ++i) {
            out[tid] = (counter.v[i>>2] >> ((i & 3) << 1)) & 3;
        }
    }

    template<typename T>
    __global__ void
    philox_uniform_kernel(T *out, unsigned long long seed, unsigned philoxcounter,
            int elementsPerBlockIteration, size_t elements)
    {
        unsigned index = blockIdx.x*elementsPerBlockIteration + threadIdx.x;
        PhiloxKey<uint, 4> key = {index, uint(seed>>32)};
        PhiloxCounter<uint, 4> counter = {index, uint(seed&0xffffffff), blockIdx.x^index, philoxcounter};
        if (blockIdx.x != (gridDim.x - 1))   {
            PhiloxCounter<uint, 4> r = philox<uint, 4>(counter, key);
            writeOut(out, index, r);
        } else {
            PhiloxCounter<uint, 4> r = philox<uint, 4>(counter, key);
            writeOutPartial(out, index, elements, r);
        }
    }
}
}

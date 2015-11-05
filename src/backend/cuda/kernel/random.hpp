/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <curand_kernel.h>
#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <platform.hpp>
#include <debug_cuda.hpp>

namespace cuda
{
namespace kernel
{

    static const int THREADS = 256;
    static const int BLOCKS  = 64;
    static unsigned long long seed = 0;
    static curandState_t *states[DeviceManager::MAX_DEVICES];
    static bool is_init[DeviceManager::MAX_DEVICES] = {0};

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
    static const float uintmaxfloat = 4294967296.0f;
    static const double uintlmaxdouble = 4294967296.0*4294967296.0;

    inline __device__ void mulhilo(const uint &a, const uint &b,
            uint& hi, uint& lo)
    {
        hi = __umulhi(a,b);
        lo = a*b;
    }

    inline __device__ void mulhilo(const uintl &a, const uintl &b,
            uintl& hi, uintl& lo)
    {
        hi = __umul64hi(a,b);
        lo = a*b;
    }

    template <typename T, T mul>
    inline __device__ void phRound(PhiloxCounter<T, 2> &ctr,
            const PhiloxKey<T, 2> &key)
    {
        T hi, lo;
        mulhilo(mul, ctr.v[0], hi, lo);
        ctr.v[0] = hi^key.v[0]^ctr.v[1];
        ctr.v[1] = lo;
    }

    template <typename T, T mul0, T mul1>
    inline __device__ void phRound(PhiloxCounter<T, 4> &ctr,
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

    inline __device__ void philoxRound(PhiloxCounter<uint, 4> &ctr,
            const PhiloxKey<uint, 4> &key)
    {
        phRound<uint, m4x32_0, m4x32_1>(ctr, key);
    }

    inline __device__ void philoxRound(PhiloxCounter<uintl, 4> &ctr,
            const PhiloxKey<uintl, 4> &key)
    {
        phRound<uintl, m4x64_0, m4x64_1>(ctr, key);
    }

    inline __device__ void philoxRound(PhiloxCounter<uint, 2> &ctr,
            const PhiloxKey<uint, 2> &key)
    {
        phRound<uint, m2x32_0>(ctr, key);
    }

    inline __device__ void philoxRound(PhiloxCounter<uintl, 2> &ctr,
            const PhiloxKey<uintl, 2> &key)
    {
        phRound<uintl, m2x64_0>(ctr, key);
    }

    inline __device__ void philoxBump(PhiloxKey<uint, 2> &key)
    {
        key.v[0] += w32_0;
    }

    inline __device__ void philoxBump(PhiloxKey<uintl, 2> &key)
    {
        key.v[0] += w64_0;
    }

    inline __device__ void philoxBump(PhiloxKey<uint, 4> &key)
    {
        key.v[0] += w32_0;
        key.v[1] += w32_1;
    }

    inline __device__ void philoxBump(PhiloxKey<uintl, 4> &key)
    {
        key.v[0] += w64_0;
        key.v[1] += w64_1;
    }

    template <typename T, unsigned int W, unsigned int R>
    inline __device__ PhiloxCounter<T,W> philox(const PhiloxCounter<T,W> &inputCtr,
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

    template<typename T>
    __device__
    void generate_uniform(T *val, curandState_t *state)
    {
        *val =  (T)curand(state);
    }

    template<> __device__
    void generate_uniform<char>(char *val, curandState_t *state)
    {
        *val = curand_uniform(state) > 0.5;
    }

    template<> __device__
    void generate_uniform<float>(float *val, curandState_t *state)
    {
        *val = curand_uniform(state);
    }

    template<> __device__
    void generate_uniform<double>(double *val, curandState_t *state)
    {
        *val = curand_uniform_double(state);
    }

    template<> __device__
    void generate_uniform<cfloat>(cfloat *cval, curandState_t *state)
    {
        cval->x = curand_uniform(state);
        cval->y = curand_uniform(state);
    }

    template<> __device__
    void generate_uniform<cdouble>(cdouble *cval, curandState_t *state)
    {
        cval->x = curand_uniform_double(state);
        cval->y = curand_uniform_double(state);
    }


    __device__
    void generate_normal(float *val, curandState_t *state)
    {
        *val = curand_normal(state);
    }


    __device__
    void generate_normal(double *val, curandState_t *state)
    {
        *val = curand_normal_double(state);
    }


    __device__
    void generate_normal(cfloat *cval, curandState_t *state)
    {
        cval->x = curand_normal(state);
        cval->y = curand_normal(state);
    }


    __device__
    void generate_normal(cdouble *cval, curandState_t *state)
    {
        cval->x = curand_normal_double(state);
        cval->y = curand_normal_double(state);
    }

    __global__ static void
    setup_kernel(curandState_t *states, unsigned long long seed)
    {
        unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
        curand_init(seed, tid, 0, &states[tid]);
    }

    template<typename T>
    __global__ static void
    uniform_kernel(T *out, curandState_t *states, size_t elements)
    {
        unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
        curandState_t state = states[id];
        for (int tid = id; tid < elements; tid += blockDim.x * gridDim.x) {
            T value;
            generate_uniform<T>(&value, &state);
            out[tid] = value;
        }
        states[id] = state;
    }

    __device__ static float normalizeToFloat(const uint& num)
    {
        return float(num)/uintmaxfloat;
    }

    __device__ static double normalizeToDouble(const uintl& num)
    {
        return double(num)/uintlmaxdouble;
    }

    __device__ static void writeOut(uint *out, const unsigned index,
            const PhiloxCounter<uint, 4>& counter)
    {
        out[index]                  = counter.v[0];
        out[index + blockDim.x]     = counter.v[1];
        out[index + 2*blockDim.x]   = counter.v[2];
        out[index + 3*blockDim.x]   = counter.v[3];
    }

    __device__ static void writeOut(uintl *out, const unsigned &index,
            const PhiloxCounter<uint, 4>& counter)
    {
        out[index]              = (uintl(counter.v[0])<<32) | uintl(counter.v[1]);
        out[index + blockDim.x] = (uintl(counter.v[2])<<32) | uintl(counter.v[3]);
    }

    __device__ static void writeOut(int *out, const unsigned index,
            const PhiloxCounter<uint, 4>& counter)
    {
        out[index]                  = counter.v[0];
        out[index + blockDim.x]     = counter.v[1];
        out[index + 2*blockDim.x]   = counter.v[2];
        out[index + 3*blockDim.x]   = counter.v[3];
    }

    __device__ static void writeOut(intl *out, const unsigned &index,
            const PhiloxCounter<uint, 4>& counter)
    {
        out[index]              = (uintl(counter.v[0])<<32) | uintl(counter.v[1]);
        out[index + blockDim.x] = (uintl(counter.v[2])<<32) | uintl(counter.v[3]);
    }

    __device__ static void writeOut(float *out, const unsigned index,
            const PhiloxCounter<uint, 4>& counter)
    {
        out[index]                  = normalizeToFloat(counter.v[0]);
        out[index + blockDim.x]     = normalizeToFloat(counter.v[1]);
        out[index + 2*blockDim.x]   = normalizeToFloat(counter.v[2]);
        out[index + 3*blockDim.x]   = normalizeToFloat(counter.v[3]);
    }

    __device__ static void writeOut(double *out, const unsigned &index,
            const PhiloxCounter<uint, 4>& counter)
    {
        out[index]              = normalizeToDouble((uintl(counter.v[0])<<32) | uintl(counter.v[1]));
        out[index + blockDim.x] = normalizeToDouble((uintl(counter.v[2])<<32) | uintl(counter.v[3]));
    }

    __device__ static void writeOut(cfloat *out, const unsigned &index,
            const PhiloxCounter<uint, 4>& counter)
    {
        out[index].x               =   normalizeToFloat(counter.v[0]);
        out[index].y               =   normalizeToFloat(counter.v[1]);
        out[index + blockDim.x].x  =   normalizeToFloat(counter.v[2]);
        out[index + blockDim.x].y  =   normalizeToFloat(counter.v[3]);
    }

    __device__ static void writeOut(cdouble *out, const unsigned &index,
            const PhiloxCounter<uint, 4>& counter)
    {
        out[index].x   =   normalizeToDouble((uintl(counter.v[0])<<32) | uintl(counter.v[1]));
        out[index].y   =   normalizeToDouble((uintl(counter.v[2])<<32) | uintl(counter.v[3]));
    }

    __device__ static void writeOut(char *out, const unsigned &index,
            const PhiloxCounter<uint, 4>& counter)
    {
        out[index]                  = (counter.v[0]&0x00000001)? 1:0;
        out[index + blockDim.x]     = (counter.v[0]&0x00000010)? 1:0;
        out[index + 2*blockDim.x]   = (counter.v[0]&0x00000100)? 1:0;
        out[index + 3*blockDim.x]   = (counter.v[0]&0x00001000)? 1:0;
        out[index + 4*blockDim.x]   = (counter.v[1]&0x00000001)? 1:0;
        out[index + 5*blockDim.x]   = (counter.v[1]&0x00000010)? 1:0;
        out[index + 6*blockDim.x]   = (counter.v[1]&0x00000100)? 1:0;
        out[index + 7*blockDim.x]   = (counter.v[1]&0x00001000)? 1:0;
        out[index + 8*blockDim.x]   = (counter.v[2]&0x00000001)? 1:0;
        out[index + 9*blockDim.x]   = (counter.v[2]&0x00000010)? 1:0;
        out[index + 10*blockDim.x]  = (counter.v[2]&0x00000100)? 1:0;
        out[index + 11*blockDim.x]  = (counter.v[2]&0x00001000)? 1:0;
        out[index + 12*blockDim.x]  = (counter.v[3]&0x00000001)? 1:0;
        out[index + 13*blockDim.x]  = (counter.v[3]&0x00000010)? 1:0;
        out[index + 14*blockDim.x]  = (counter.v[3]&0x00000100)? 1:0;
        out[index + 15*blockDim.x]  = (counter.v[3]&0x00001000)? 1:0;
    }

    __device__ static void writeOut(uchar *out, const unsigned &index,
            const PhiloxCounter<uint, 4>& counter)
    {
        out[index]                  = (counter.v[0]&0x00000011);
        out[index + blockDim.x]     = (counter.v[0]&0x00001100)>>2;
        out[index + 2*blockDim.x]   = (counter.v[0]&0x00110000)>>4;
        out[index + 3*blockDim.x]   = (counter.v[0]&0x11000000)>>6;
        out[index + 4*blockDim.x]   = (counter.v[1]&0x00000011);
        out[index + 5*blockDim.x]   = (counter.v[1]&0x00001100)>>2;
        out[index + 6*blockDim.x]   = (counter.v[1]&0x00110000)>>4;
        out[index + 7*blockDim.x]   = (counter.v[1]&0x11000000)>>6;
        out[index + 8*blockDim.x]   = (counter.v[2]&0x00000011);
        out[index + 9*blockDim.x]   = (counter.v[2]&0x00001100)>>2;
        out[index + 10*blockDim.x]  = (counter.v[2]&0x00110000)>>4;
        out[index + 11*blockDim.x]  = (counter.v[2]&0x11000000)>>6;
        out[index + 12*blockDim.x]  = (counter.v[3]&0x00000011);
        out[index + 13*blockDim.x]  = (counter.v[3]&0x00001100)>>2;
        out[index + 14*blockDim.x]  = (counter.v[3]&0x00110000)>>4;
        out[index + 15*blockDim.x]  = (counter.v[3]&0x11000000)>>6;
    }

    __device__ static void writeOut(short *out, const unsigned &index,
            const PhiloxCounter<uint, 4>& counter)
    {
        out[index]                  = (counter.v[0]&0x00001111);
        out[index + blockDim.x]     = (counter.v[0]>>4);
        out[index + 2*blockDim.x]   = (counter.v[1]&0x00001111);
        out[index + 3*blockDim.x]   = (counter.v[1]>>4);
        out[index + 4*blockDim.x]   = (counter.v[2]&0x00001111);
        out[index + 5*blockDim.x]   = (counter.v[2]>>4);
        out[index + 6*blockDim.x]   = (counter.v[3]&0x00001111);
        out[index + 7*blockDim.x]   = (counter.v[3]>>4);
    }

    __device__ static void writeOut(ushort *out, const unsigned &index,
            const PhiloxCounter<uint, 4>& counter)
    {
        out[index]                  = (counter.v[0]&0x00001111);
        out[index + blockDim.x]     = (counter.v[0]>>4);
        out[index + 2*blockDim.x]   = (counter.v[1]&0x00001111);
        out[index + 3*blockDim.x]   = (counter.v[1]>>4);
        out[index + 4*blockDim.x]   = (counter.v[2]&0x00001111);
        out[index + 5*blockDim.x]   = (counter.v[2]>>4);
        out[index + 6*blockDim.x]   = (counter.v[3]&0x00001111);
        out[index + 7*blockDim.x]   = (counter.v[3]>>4);
    }

    __device__ static void writeOutPartial(uint *out, const unsigned index,
            const size_t &elements, const PhiloxCounter<uint, 4>& counter)
    {
        if (index                   < elements) {out[index]                  = counter.v[0];}
        if (index + blockDim.x      < elements) {out[index + blockDim.x]     = counter.v[1];}
        if (index + 2*blockDim.x    < elements) {out[index + 2*blockDim.x]   = counter.v[2];}
        if (index + 3*blockDim.x    < elements) {out[index + 3*blockDim.x]   = counter.v[3];}
    }

    __device__ static void writeOutPartial(uintl *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4>& counter)
    {
        if (index               < elements) {out[index]              = (uintl(counter.v[0])<<32) | uintl(counter.v[1]);}
        if (index + blockDim.x  < elements) {out[index + blockDim.x] = (uintl(counter.v[2])<<32) | uintl(counter.v[3]);}
    }

    __device__ static void writeOutPartial(int *out, const unsigned index,
            const size_t &elements, const PhiloxCounter<uint, 4>& counter)
    {
        if (index                   < elements) {out[index]                  = counter.v[0];}
        if (index + blockDim.x      < elements) {out[index + blockDim.x]     = counter.v[1];}
        if (index + 2*blockDim.x    < elements) {out[index + 2*blockDim.x]   = counter.v[2];}
        if (index + 3*blockDim.x    < elements) {out[index + 3*blockDim.x]   = counter.v[3];}
    }

    __device__ static void writeOutPartial(intl *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4>& counter)
    {
        if (index               < elements) {out[index]              = (uintl(counter.v[0])<<32) | uintl(counter.v[1]);}
        if (index + blockDim.x  < elements) {out[index + blockDim.x] = (uintl(counter.v[2])<<32) | uintl(counter.v[3]);}
    }

    __device__ static void writeOutPartial(float *out, const unsigned index,
            const size_t &elements, const PhiloxCounter<uint, 4>& counter)
    {
        if (index                   < elements) {out[index]                  = normalizeToFloat(counter.v[0]);}
        if (index + blockDim.x      < elements) {out[index + blockDim.x]     = normalizeToFloat(counter.v[1]);}
        if (index + 2*blockDim.x    < elements) {out[index + 2*blockDim.x]   = normalizeToFloat(counter.v[2]);}
        if (index + 3*blockDim.x    < elements) {out[index + 3*blockDim.x]   = normalizeToFloat(counter.v[3]);}
    }

    __device__ static void writeOutPartial(double *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4>& counter)
    {
        if (index               < elements)
            {out[index]              = normalizeToDouble((uintl(counter.v[0])<<32) | uintl(counter.v[1]));}
        if (index + blockDim.x  < elements)
            {out[index + blockDim.x] = normalizeToDouble((uintl(counter.v[2])<<32) | uintl(counter.v[3]));}
    }

    __device__ static void writeOutPartial(cfloat *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4>& counter)
    {
        if (index               < elements) {
            out[index].x               =   normalizeToFloat(counter.v[0]);
            out[index].y               =   normalizeToFloat(counter.v[1]);}
        if (index + blockDim.x  < elements) {
            out[index + blockDim.x].x  =   normalizeToFloat(counter.v[2]);
            out[index + blockDim.x].y  =   normalizeToFloat(counter.v[2]);}
    }

    __device__ static void writeOutPartial(cdouble *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4>& counter)
    {
        if (index < elements) {
            out[index].x   =   normalizeToDouble((uintl(counter.v[0])<<32) | uintl(counter.v[1]));
            out[index].y   =   normalizeToDouble((uintl(counter.v[2])<<32) | uintl(counter.v[3]));}
    }

    __device__ static void writeOutPartial(char *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4>& counter)
    {
        if (index                   < elements) {out[index]                  = (counter.v[0]&0x00000001)? 1:0;}
        if (index + blockDim.x      < elements) {out[index + blockDim.x]     = (counter.v[0]&0x00000010)? 1:0;}
        if (index + 2*blockDim.x    < elements) {out[index + 2*blockDim.x]   = (counter.v[0]&0x00000100)? 1:0;}
        if (index + 3*blockDim.x    < elements) {out[index + 3*blockDim.x]   = (counter.v[0]&0x00001000)? 1:0;}
        if (index + 4*blockDim.x    < elements) {out[index + 4*blockDim.x]   = (counter.v[1]&0x00000001)? 1:0;}
        if (index + 5*blockDim.x    < elements) {out[index + 5*blockDim.x]   = (counter.v[1]&0x00000010)? 1:0;}
        if (index + 6*blockDim.x    < elements) {out[index + 6*blockDim.x]   = (counter.v[1]&0x00000100)? 1:0;}
        if (index + 7*blockDim.x    < elements) {out[index + 7*blockDim.x]   = (counter.v[1]&0x00001000)? 1:0;}
        if (index + 8*blockDim.x    < elements) {out[index + 8*blockDim.x]   = (counter.v[2]&0x00000001)? 1:0;}
        if (index + 9*blockDim.x    < elements) {out[index + 9*blockDim.x]   = (counter.v[2]&0x00000010)? 1:0;}
        if (index + 10*blockDim.x   < elements) {out[index + 10*blockDim.x]  = (counter.v[2]&0x00000100)? 1:0;}
        if (index + 11*blockDim.x   < elements) {out[index + 11*blockDim.x]  = (counter.v[2]&0x00001000)? 1:0;}
        if (index + 12*blockDim.x   < elements) {out[index + 12*blockDim.x]  = (counter.v[3]&0x00000001)? 1:0;}
        if (index + 13*blockDim.x   < elements) {out[index + 13*blockDim.x]  = (counter.v[3]&0x00000010)? 1:0;}
        if (index + 14*blockDim.x   < elements) {out[index + 14*blockDim.x]  = (counter.v[3]&0x00000100)? 1:0;}
        if (index + 15*blockDim.x   < elements) {out[index + 15*blockDim.x]  = (counter.v[3]&0x00001000)? 1:0;}
    }

    __device__ static void writeOutPartial(uchar *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4>& counter)
    {
        if (index                   < elements) {out[index]                  = (counter.v[0]&0x00000011);   }
        if (index + blockDim.x      < elements) {out[index + blockDim.x]     = (counter.v[0]&0x00001100)>>2;}
        if (index + 2*blockDim.x    < elements) {out[index + 2*blockDim.x]   = (counter.v[0]&0x00110000)>>4;}
        if (index + 3*blockDim.x    < elements) {out[index + 3*blockDim.x]   = (counter.v[0]&0x11000000)>>6;}
        if (index + 4*blockDim.x    < elements) {out[index + 4*blockDim.x]   = (counter.v[1]&0x00000011);   }
        if (index + 5*blockDim.x    < elements) {out[index + 5*blockDim.x]   = (counter.v[1]&0x00001100)>>2;}
        if (index + 6*blockDim.x    < elements) {out[index + 6*blockDim.x]   = (counter.v[1]&0x00110000)>>4;}
        if (index + 7*blockDim.x    < elements) {out[index + 7*blockDim.x]   = (counter.v[1]&0x11000000)>>6;}
        if (index + 8*blockDim.x    < elements) {out[index + 8*blockDim.x]   = (counter.v[2]&0x00000011);   }
        if (index + 9*blockDim.x    < elements) {out[index + 9*blockDim.x]   = (counter.v[2]&0x00001100)>>2;}
        if (index + 10*blockDim.x   < elements) {out[index + 10*blockDim.x]  = (counter.v[2]&0x00110000)>>4;}
        if (index + 11*blockDim.x   < elements) {out[index + 11*blockDim.x]  = (counter.v[2]&0x11000000)>>6;}
        if (index + 12*blockDim.x   < elements) {out[index + 12*blockDim.x]  = (counter.v[3]&0x00000011);   }
        if (index + 13*blockDim.x   < elements) {out[index + 13*blockDim.x]  = (counter.v[3]&0x00001100)>>2;}
        if (index + 14*blockDim.x   < elements) {out[index + 14*blockDim.x]  = (counter.v[3]&0x00110000)>>4;}
        if (index + 15*blockDim.x   < elements) {out[index + 15*blockDim.x]  = (counter.v[3]&0x11000000)>>6;}
    }

    __device__ static void writeOutPartial(short *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4>& counter)
    {
         if (index                  < elements) {out[index]                  = (counter.v[0]&0x00001111);}
         if (index + blockDim.x     < elements) {out[index + blockDim.x]     = (counter.v[0]>>4);}
         if (index + 2*blockDim.x   < elements) {out[index + 2*blockDim.x]   = (counter.v[1]&0x00001111);}
         if (index + 3*blockDim.x   < elements) {out[index + 3*blockDim.x]   = (counter.v[1]>>4);}
         if (index + 4*blockDim.x   < elements) {out[index + 4*blockDim.x]   = (counter.v[2]&0x00001111);}
         if (index + 5*blockDim.x   < elements) {out[index + 5*blockDim.x]   = (counter.v[2]>>4);}
         if (index + 6*blockDim.x   < elements) {out[index + 6*blockDim.x]   = (counter.v[3]&0x00001111);}
         if (index + 7*blockDim.x   < elements) {out[index + 7*blockDim.x]   = (counter.v[3]>>4);}
    }

    __device__ static void writeOutPartial(ushort *out, const unsigned &index,
            const size_t &elements, const PhiloxCounter<uint, 4>& counter)
    {
         if (index                  < elements) {out[index]                  = (counter.v[0]&0x00001111);}
         if (index + blockDim.x     < elements) {out[index + blockDim.x]     = (counter.v[0]>>4);}
         if (index + 2*blockDim.x   < elements) {out[index + 2*blockDim.x]   = (counter.v[1]&0x00001111);}
         if (index + 3*blockDim.x   < elements) {out[index + 3*blockDim.x]   = (counter.v[1]>>4);}
         if (index + 4*blockDim.x   < elements) {out[index + 4*blockDim.x]   = (counter.v[2]&0x00001111);}
         if (index + 5*blockDim.x   < elements) {out[index + 5*blockDim.x]   = (counter.v[2]>>4);}
         if (index + 6*blockDim.x   < elements) {out[index + 6*blockDim.x]   = (counter.v[3]&0x00001111);}
         if (index + 7*blockDim.x   < elements) {out[index + 7*blockDim.x]   = (counter.v[3]>>4);}
    }
    template<typename T>
    __global__ static void
    philox_uniform_kernel(T *out, unsigned long long seed,
            int elementsPerBlockIteration, size_t elements)
    {
        unsigned index = blockIdx.x*elementsPerBlockIteration + threadIdx.x;
        PhiloxKey<uint, 4> key = {index, seed>>32};
        PhiloxCounter<uint, 4> counter = {index, (seed&0xffffffff), blockIdx.x^index, 0xa22af12e};
        if (blockIdx.x != (gridDim.x - 1))   {
            PhiloxCounter<uint, 4> r = philox<uint, 4>(counter, key);
            writeOut(out, index, r);
        } else {
            PhiloxCounter<uint, 4> r = philox<uint, 4>(counter, key);
            writeOutPartial(out, index, elements, r);
        }
    }

    template<typename T>
    __global__ static void
    normal_kernel(T *out, curandState_t *states, size_t elements)
    {
        unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
        curandState_t state = states[id];
        for (int tid = id; tid < elements; tid += blockDim.x * gridDim.x) {
            T value;
            generate_normal(&value, &state);
            out[tid] = value;
        }
        states[id] = state;
    }

    void setup_states()
    {
        int device = getActiveDeviceId();

        if (!is_init[device]) {
            CUDA_CHECK(cudaMalloc(&states[device], BLOCKS * THREADS * sizeof(curandState_t)));
        }

        CUDA_LAUNCH((setup_kernel), BLOCKS, THREADS, states[device], seed);
        POST_LAUNCH_CHECK();
        is_init[device] = true;
    }

    template<typename T>
    void randu(T *out, size_t elements, const af::randomType &rtype)
    {
        switch (rtype) {
            case AF_RANDOM_DEFAULT:
                {
                    int device = getActiveDeviceId();
                    int threads = THREADS;
                    int blocks  = divup(elements, THREADS);
                    if (blocks > BLOCKS) blocks = BLOCKS;
                    CUDA_LAUNCH(uniform_kernel, blocks, threads, out, states[device], elements);
                    break;
                }
            case AF_RANDOM_PHILOX:
                {
                    int threads = THREADS;
                    int elementsPerBlockIteration = THREADS*4*sizeof(unsigned)/sizeof(T);
                    int blocks = divup(elements, elementsPerBlockIteration);
                    CUDA_LAUNCH(philox_uniform_kernel, blocks, threads,
                            out, seed, elementsPerBlockIteration, elements);
                    ++seed;
                    break;
                }
        }
        POST_LAUNCH_CHECK();
    }

    template<typename T>
    void randn(T *out, size_t elements)
    {
        int device = getActiveDeviceId();

        int threads = THREADS;
        int blocks  = divup(elements, THREADS);
        if (blocks > BLOCKS) blocks = BLOCKS;

        if (!states[device]) {
            CUDA_CHECK(cudaMalloc(&states[device], BLOCKS * THREADS * sizeof(curandState_t)));

            CUDA_LAUNCH(setup_kernel, BLOCKS, THREADS, states[device], seed);

            POST_LAUNCH_CHECK();
        }

        CUDA_LAUNCH(normal_kernel, blocks, threads, out, states[device], elements);

        POST_LAUNCH_CHECK();
    }
}
}

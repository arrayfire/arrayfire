/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <af/defines.h>
#include <kernel/random_engine_philox.hpp>
#include <kernel/random_engine_threefry.hpp>

namespace cuda
{
namespace kernel
{
    //Utils
    static const int THREADS = 256;
    #define UINTMAXFLOAT 4294967296.0f
    #define UINTLMAXDOUBLE 4294967296.0*4294967296.0
    #define PI_VAL 3.1415926535897932384626433832795028841971693993751058209749445923078164

    __device__ static float getFloat(const uint &num)
    {
        return float(num)/UINTMAXFLOAT;
    }

    __device__ static double getDouble(const uint &num1, const uint &num2)
    {
        uintl num = (((uintl)num1)<<32) | ((uintl)num2);
        return double(num)/UINTLMAXDOUBLE;
    }

    template <typename T>
    __device__ static void normalize(T * const out1, T * const out2, const T &r1, const T &r2)
    {
        T r = sqrt((T)(-2.0) * log(r1));
        T theta = 2 * (T)PI_VAL * r2;
        *out1 = r*sin(theta);
        *out2 = r*cos(theta);
    }

    //Writes without boundary checking

    __device__ static void writeOut256Bytes(uchar *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        out[index]                 =     r1;
        out[index +    blockDim.x] =  r1>>8;
        out[index +  2*blockDim.x] = r1>>16;
        out[index +  3*blockDim.x] = r1>>24;
        out[index +  4*blockDim.x] =     r2;
        out[index +  5*blockDim.x] =  r2>>8;
        out[index +  6*blockDim.x] = r2>>16;
        out[index +  7*blockDim.x] = r2>>24;
        out[index +  8*blockDim.x] =     r3;
        out[index +  9*blockDim.x] =  r3>>8;
        out[index + 10*blockDim.x] = r3>>16;
        out[index + 11*blockDim.x] = r3>>24;
        out[index + 12*blockDim.x] =     r4;
        out[index + 13*blockDim.x] =  r4>>8;
        out[index + 14*blockDim.x] = r4>>16;
        out[index + 15*blockDim.x] = r4>>24;
    }

    __device__ static void writeOut256Bytes(char *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        out[index]                 = (r1   )&0x1;
        out[index +    blockDim.x] = (r1>>1)&0x1;
        out[index +  2*blockDim.x] = (r1>>2)&0x1;
        out[index +  3*blockDim.x] = (r1>>3)&0x1;
        out[index +  4*blockDim.x] = (r2   )&0x1;
        out[index +  5*blockDim.x] = (r2>>1)&0x1;
        out[index +  6*blockDim.x] = (r2>>2)&0x1;
        out[index +  7*blockDim.x] = (r2>>3)&0x1;
        out[index +  8*blockDim.x] = (r3   )&0x1;
        out[index +  9*blockDim.x] = (r3>>1)&0x1;
        out[index + 10*blockDim.x] = (r3>>2)&0x1;
        out[index + 11*blockDim.x] = (r3>>3)&0x1;
        out[index + 12*blockDim.x] = (r4   )&0x1;
        out[index + 13*blockDim.x] = (r4>>1)&0x1;
        out[index + 14*blockDim.x] = (r4>>2)&0x1;
        out[index + 15*blockDim.x] = (r4>>3)&0x1;
    }

    __device__ static void writeOut256Bytes(short *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        out[index]                =     r1;
        out[index +   blockDim.x] = r1>>16;
        out[index + 2*blockDim.x] =     r2;
        out[index + 3*blockDim.x] = r2>>16;
        out[index + 4*blockDim.x] =     r3;
        out[index + 5*blockDim.x] = r3>>16;
        out[index + 6*blockDim.x] =     r4;
        out[index + 7*blockDim.x] = r4>>16;
    }

    __device__ static void writeOut256Bytes(ushort *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        writeOut256Bytes((short*)(out), index, r1, r2, r3, r4);
    }

    __device__ static void writeOut256Bytes(int *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        out[index]                = r1;
        out[index +   blockDim.x] = r2;
        out[index + 2*blockDim.x] = r3;
        out[index + 3*blockDim.x] = r4;
    }

    __device__ static void writeOut256Bytes(uint *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        writeOut256Bytes((int*)(out), index, r1, r2, r3, r4);
    }

    __device__ static void writeOut256Bytes(intl *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        intl c1 = r2;
        c1 = (c1<<32) | r1;
        intl c2 = r4;
        c2 = (c2<<32) | r3;
        out[index]              = c1;
        out[index + blockDim.x] = c2;
    }

    __device__ static void writeOut256Bytes(uintl *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        writeOut256Bytes((intl*)(out), index, r1, r2, r3, r4);
    }

    __device__ static void writeOut256Bytes(float *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        out[index]                = getFloat(r1);
        out[index +   blockDim.x] = getFloat(r2);
        out[index + 2*blockDim.x] = getFloat(r3);
        out[index + 3*blockDim.x] = getFloat(r4);
    }

    __device__ static void writeOut256Bytes(cfloat *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        out[index].x              = getFloat(r1);
        out[index].y              = getFloat(r2);
        out[index + blockDim.x].x = getFloat(r3);
        out[index + blockDim.x].y = getFloat(r4);
    }

    __device__ static void writeOut256Bytes(double *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        out[index]              = getDouble(r1, r2);
        out[index + blockDim.x] = getDouble(r3, r4);
    }

    __device__ static void writeOut256Bytes(cdouble *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        out[index].x = getDouble(r1, r2);
        out[index].y = getDouble(r3, r4);
    }

    //Normalized writes without boundary checking

    __device__ static void normalizedWriteOut256Bytes(float *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        normalize(&out[index]               , &out[index +   blockDim.x], getFloat(r1), getFloat(r2));
        normalize(&out[index + 2*blockDim.x], &out[index + 3*blockDim.x], getFloat(r1), getFloat(r2));
    }

    __device__ static void normalizedWriteOut256Bytes(cfloat *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        normalize(&out[index].x             , &out[index].y             , getFloat(r1), getFloat(r2));
        normalize(&out[index + blockDim.x].x, &out[index + blockDim.x].y, getFloat(r3), getFloat(r4));
    }

    __device__ static void normalizedWriteOut256Bytes(double *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        normalize(&out[index], &out[index + blockDim.x], getDouble(r1, r2), getDouble(r3, r4));
    }

    __device__ static void normalizedWriteOut256Bytes(cdouble *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        normalize(&out[index].x, &out[index].y, getDouble(r1, r2), getDouble(r3, r4));
    }

    //Writes with boundary checking

    __device__ static void partialWriteOut256Bytes(uchar *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        if (index                 < elements) {out[index]                 =     r1;}
        if (index +    blockDim.x < elements) {out[index +    blockDim.x] =  r1>>8;}
        if (index +  2*blockDim.x < elements) {out[index +  2*blockDim.x] = r1>>16;}
        if (index +  3*blockDim.x < elements) {out[index +  3*blockDim.x] = r1>>24;}
        if (index +  4*blockDim.x < elements) {out[index +  4*blockDim.x] =     r2;}
        if (index +  5*blockDim.x < elements) {out[index +  5*blockDim.x] =  r2>>8;}
        if (index +  6*blockDim.x < elements) {out[index +  6*blockDim.x] = r2>>16;}
        if (index +  7*blockDim.x < elements) {out[index +  7*blockDim.x] = r2>>24;}
        if (index +  8*blockDim.x < elements) {out[index +  8*blockDim.x] =     r3;}
        if (index +  9*blockDim.x < elements) {out[index +  9*blockDim.x] =  r3>>8;}
        if (index + 10*blockDim.x < elements) {out[index + 10*blockDim.x] = r3>>16;}
        if (index + 11*blockDim.x < elements) {out[index + 11*blockDim.x] = r3>>24;}
        if (index + 12*blockDim.x < elements) {out[index + 12*blockDim.x] =     r4;}
        if (index + 13*blockDim.x < elements) {out[index + 13*blockDim.x] =  r4>>8;}
        if (index + 14*blockDim.x < elements) {out[index + 14*blockDim.x] = r4>>16;}
        if (index + 15*blockDim.x < elements) {out[index + 15*blockDim.x] = r4>>24;}
    }

    __device__ static void partialWriteOut256Bytes(char *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        if (index                 < elements) {out[index]                 = (r1   )&0x1;}
        if (index +    blockDim.x < elements) {out[index +    blockDim.x] = (r1>>1)&0x1;}
        if (index +  2*blockDim.x < elements) {out[index +  2*blockDim.x] = (r1>>2)&0x1;}
        if (index +  3*blockDim.x < elements) {out[index +  3*blockDim.x] = (r1>>3)&0x1;}
        if (index +  4*blockDim.x < elements) {out[index +  4*blockDim.x] = (r2   )&0x1;}
        if (index +  5*blockDim.x < elements) {out[index +  5*blockDim.x] = (r2>>1)&0x1;}
        if (index +  6*blockDim.x < elements) {out[index +  6*blockDim.x] = (r2>>2)&0x1;}
        if (index +  7*blockDim.x < elements) {out[index +  7*blockDim.x] = (r2>>3)&0x1;}
        if (index +  8*blockDim.x < elements) {out[index +  8*blockDim.x] = (r3   )&0x1;}
        if (index +  9*blockDim.x < elements) {out[index +  9*blockDim.x] = (r3>>1)&0x1;}
        if (index + 10*blockDim.x < elements) {out[index + 10*blockDim.x] = (r3>>2)&0x1;}
        if (index + 11*blockDim.x < elements) {out[index + 11*blockDim.x] = (r3>>3)&0x1;}
        if (index + 12*blockDim.x < elements) {out[index + 12*blockDim.x] = (r4   )&0x1;}
        if (index + 13*blockDim.x < elements) {out[index + 13*blockDim.x] = (r4>>1)&0x1;}
        if (index + 14*blockDim.x < elements) {out[index + 14*blockDim.x] = (r4>>2)&0x1;}
        if (index + 15*blockDim.x < elements) {out[index + 15*blockDim.x] = (r4>>3)&0x1;}
    }

    __device__ static void partialWriteOut256Bytes(short *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        if (index                < elements) {out[index]                =     r1;}
        if (index +   blockDim.x < elements) {out[index +   blockDim.x] = r1>>16;}
        if (index + 2*blockDim.x < elements) {out[index + 2*blockDim.x] =     r2;}
        if (index + 3*blockDim.x < elements) {out[index + 3*blockDim.x] = r2>>16;}
        if (index + 4*blockDim.x < elements) {out[index + 4*blockDim.x] =     r3;}
        if (index + 5*blockDim.x < elements) {out[index + 5*blockDim.x] = r3>>16;}
        if (index + 6*blockDim.x < elements) {out[index + 6*blockDim.x] =     r4;}
        if (index + 7*blockDim.x < elements) {out[index + 7*blockDim.x] = r4>>16;}
    }

    __device__ static void partialWriteOut256Bytes(ushort *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        partialWriteOut256Bytes((short*)(out), index, r1, r2, r3, r4, elements);
    }

    __device__ static void partialWriteOut256Bytes(int *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        if (index                < elements) {out[index]                = r1;}
        if (index +   blockDim.x < elements) {out[index +   blockDim.x] = r2;}
        if (index + 2*blockDim.x < elements) {out[index + 2*blockDim.x] = r3;}
        if (index + 3*blockDim.x < elements) {out[index + 3*blockDim.x] = r4;}
    }

    __device__ static void partialWriteOut256Bytes(uint *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        partialWriteOut256Bytes((int*)(out), index, r1, r2, r3, r4, elements);
    }

    __device__ static void partialWriteOut256Bytes(intl *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        intl c1 = r2;
        c1 = (c1<<32) | r1;
        intl c2 = r4;
        c2 = (c2<<32) | r3;
        if (index              < elements) {out[index]              = c1;}
        if (index + blockDim.x < elements) {out[index + blockDim.x] = c2;}
    }

    __device__ static void partialWriteOut256Bytes(uintl *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        partialWriteOut256Bytes((intl*)(out), index, r1, r2, r3, r4, elements);
    }

    __device__ static void partialWriteOut256Bytes(float *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        if (index                < elements) {out[index]                = getFloat(r1);}
        if (index +   blockDim.x < elements) {out[index +   blockDim.x] = getFloat(r2);}
        if (index + 2*blockDim.x < elements) {out[index + 2*blockDim.x] = getFloat(r3);}
        if (index + 3*blockDim.x < elements) {out[index + 3*blockDim.x] = getFloat(r4);}
    }

    __device__ static void partialWriteOut256Bytes(cfloat *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        if (index              < elements) {
            out[index].x              = getFloat(r1);
            out[index].y              = getFloat(r2);
        }
        if (index + blockDim.x < elements) {
            out[index + blockDim.x].x = getFloat(r3);
            out[index + blockDim.x].y = getFloat(r4);
        }
    }

    __device__ static void partialWriteOut256Bytes(double *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        if (index              < elements) {out[index]              = getDouble(r1, r2);}
        if (index + blockDim.x < elements) {out[index + blockDim.x] = getDouble(r3, r4);}
    }

    __device__ static void partialWriteOut256Bytes(cdouble *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        if (index < elements) {
            out[index].x = getDouble(r1, r2);
            out[index].y = getDouble(r3, r4);
        }
    }

    //Normalized writes with boundary checking

    __device__ static void partialNormalizedWriteOut256Bytes(float *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        float n1, n2, n3, n4;
        normalize(&n1, &n2, getFloat(r1), getFloat(r2));
        normalize(&n3, &n4, getFloat(r3), getFloat(r4));
        if (index                < elements) {out[index]                = n1;}
        if (index +   blockDim.x < elements) {out[index +   blockDim.x] = n2;}
        if (index + 2*blockDim.x < elements) {out[index + 2*blockDim.x] = n3;}
        if (index + 3*blockDim.x < elements) {out[index + 3*blockDim.x] = n4;}
    }

    __device__ static void partialNormalizedWriteOut256Bytes(cfloat *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        float n1, n2, n3, n4;
        normalize(&n1, &n2, getFloat(r1), getFloat(r2));
        normalize(&n3, &n4, getFloat(r3), getFloat(r4));
        if (index              < elements) {
            out[index].x              = n1;
            out[index].y              = n2;
        }
        if (index + blockDim.x < elements) {
            out[index + blockDim.x].x = n3;
            out[index + blockDim.x].y = n4;
        }
    }

    __device__ static void partialNormalizedWriteOut256Bytes(double *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        double n1, n2;
        normalize(&n1, &n2, getDouble(r1, r2), getDouble(r3, r4));
        if (index              < elements) {out[index]              = n1;}
        if (index + blockDim.x < elements) {out[index + blockDim.x] = n2;}
    }

    __device__ static void partialNormalizedWriteOut256Bytes(cdouble *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        double n1, n2;
        normalize(&n1, &n2, getDouble(r1, r2), getDouble(r3, r4));
        if (index < elements) {
            out[index].x = n1;
            out[index].y = n2;
        }
    }

    template <typename T>
    __global__ void uniformPhilox(T *out, uint hi, uint lo, uint counter, uint elementsPerBlock, uint elements)
    {
        uint index = blockIdx.x*elementsPerBlock + threadIdx.x;
        uint key[2] = {index, hi};
        uint ctr[4] = {index+counter, 0, 0, lo};
        if (blockIdx.x != (gridDim.x - 1)) {
            philox(key, ctr);
            writeOut256Bytes(out, index, ctr[0], ctr[1], ctr[2], ctr[3]);
        } else {
            philox(key, ctr);
            partialWriteOut256Bytes(out, index, ctr[0], ctr[1], ctr[2], ctr[3], elements);
        }
    }

    template <typename T>
    __global__ void uniformThreefry(T *out, uint hi, uint lo, uint counter, uint elementsPerBlock, uint elements)
    {
        uint index = blockIdx.x*elementsPerBlock + threadIdx.x;
        uint key[2] = {index, hi};
        uint ctr[2] = {index+counter, lo};
        uint o[4];
        if (blockIdx.x != (gridDim.x - 1)) {
            threefry(key, ctr, o);
            ctr[1] += elements;
            threefry(key, ctr, o + 2);
            writeOut256Bytes(out, index, o[0], o[1], o[2], o[3]);
        } else {
            threefry(key, ctr, o);
            ctr[1] += elements;
            threefry(key, ctr, o + 2);
            partialWriteOut256Bytes(out, index, o[0], o[1], o[2], o[3], elements);
        }
    }

    template <typename T, af_random_type Type>
    void uniformDistribution(T *out, size_t elements, const uintl seed, uintl &counter)
    {
        int threads = THREADS;
        int elementsPerBlock = threads*4*sizeof(uint)/sizeof(T);
        int blocks = divup(elements, elementsPerBlock);
        uint hi = seed>>32;
        uint lo = seed;
        uintl count = counter;
        switch (Type) {
        case AF_RANDOM_PHILOX : CUDA_LAUNCH(uniformPhilox, blocks, threads,
                out, hi, lo, count, elementsPerBlock, elements); break;
        case AF_RANDOM_THREEFRY : CUDA_LAUNCH(uniformThreefry, blocks, threads,
                out, hi, lo, count, elementsPerBlock, elements); break;
        }
        counter += elements;
    }
}
}

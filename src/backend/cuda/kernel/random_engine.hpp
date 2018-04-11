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
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <af/defines.h>
#include <kernel/random_engine_philox.hpp>
#include <kernel/random_engine_threefry.hpp>
#include <kernel/random_engine_mersenne.hpp>
#include <random_engine.hpp>

namespace cuda
{
namespace kernel
{
    //Utils

    static const int THREADS = 256;
    #define PI_VAL 3.1415926535897932384626433832795028841971693993751058209749445923078164

    //Conversion to floats adapted from Random123
    #define UINTMAX 0xffffffff
    #define FLT_FACTOR ((1.0f)/(UINTMAX + (1.0f)))
    #define HALF_FLT_FACTOR ((0.5f)*FLT_FACTOR)

    #define UINTLMAX 0xffffffffffffffff
    #define DBL_FACTOR ((1.0)/(UINTLMAX + (1.0)))
    #define HALF_DBL_FACTOR ((0.5)*DBL_FACTOR)

    //Generates rationals in (0, 1]
    __device__ static float getFloat(const uint &num)
    {
        return (num*FLT_FACTOR + HALF_FLT_FACTOR);
    }

    //Generates rationals in (0, 1]
    __device__ static double getDouble(const uint &num1, const uint &num2)
    {
        uintl num = (((uintl)num1)<<32) | ((uintl)num2);
        return (num*DBL_FACTOR + HALF_DBL_FACTOR);
    }

    template <typename T>
    __device__ static void boxMullerTransform(T * const out1, T * const out2, const T &r1, const T &r2)
    {
        /*
         * The log of a real value x where 0 < x < 1 is negative.
         */
        T r = sqrt((T)(-2.0) * log(r1));
        T theta = 2 * (T)PI_VAL * r2;
        *out1 = r*sin(theta);
        *out2 = r*cos(theta);
    }

    //Writes without boundary checking

    __device__ static void writeOut128Bytes(uchar *out, const uint &index,
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

    __device__ static void writeOut128Bytes(char *out, const uint &index,
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

    __device__ static void writeOut128Bytes(short *out, const uint &index,
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

    __device__ static void writeOut128Bytes(ushort *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        writeOut128Bytes((short*)(out), index, r1, r2, r3, r4);
    }

    __device__ static void writeOut128Bytes(int *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        out[index]                = r1;
        out[index +   blockDim.x] = r2;
        out[index + 2*blockDim.x] = r3;
        out[index + 3*blockDim.x] = r4;
    }

    __device__ static void writeOut128Bytes(uint *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        writeOut128Bytes((int*)(out), index, r1, r2, r3, r4);
    }

    __device__ static void writeOut128Bytes(intl *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        intl c1 = r2;
        c1 = (c1<<32) | r1;
        intl c2 = r4;
        c2 = (c2<<32) | r3;
        out[index]              = c1;
        out[index + blockDim.x] = c2;
    }

    __device__ static void writeOut128Bytes(uintl *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        writeOut128Bytes((intl*)(out), index, r1, r2, r3, r4);
    }

    __device__ static void writeOut128Bytes(float *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        out[index]                = 1.f - getFloat(r1);
        out[index +   blockDim.x] = 1.f - getFloat(r2);
        out[index + 2*blockDim.x] = 1.f - getFloat(r3);
        out[index + 3*blockDim.x] = 1.f - getFloat(r4);
    }

    __device__ static void writeOut128Bytes(cfloat *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        out[index].x              = 1.f - getFloat(r1);
        out[index].y              = 1.f - getFloat(r2);
        out[index + blockDim.x].x = 1.f - getFloat(r3);
        out[index + blockDim.x].y = 1.f - getFloat(r4);
    }

    __device__ static void writeOut128Bytes(double *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        out[index]              = 1.0 - getDouble(r1, r2);
        out[index + blockDim.x] = 1.0 - getDouble(r3, r4);
    }

    __device__ static void writeOut128Bytes(cdouble *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        out[index].x = 1.0 - getDouble(r1, r2);
        out[index].y = 1.0 - getDouble(r3, r4);
    }

    //Normalized writes without boundary checking

    __device__ static void boxMullerWriteOut128Bytes(float *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        boxMullerTransform(&out[index]               , &out[index +   blockDim.x], getFloat(r1), getFloat(r2));
        boxMullerTransform(&out[index + 2*blockDim.x], &out[index + 3*blockDim.x], getFloat(r1), getFloat(r2));
    }

    __device__ static void boxMullerWriteOut128Bytes(cfloat *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        boxMullerTransform(&out[index].x             , &out[index].y             , getFloat(r1), getFloat(r2));
        boxMullerTransform(&out[index + blockDim.x].x, &out[index + blockDim.x].y, getFloat(r3), getFloat(r4));
    }

    __device__ static void boxMullerWriteOut128Bytes(double *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        boxMullerTransform(&out[index], &out[index + blockDim.x], getDouble(r1, r2), getDouble(r3, r4));
    }

    __device__ static void boxMullerWriteOut128Bytes(cdouble *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4)
    {
        boxMullerTransform(&out[index].x, &out[index].y, getDouble(r1, r2), getDouble(r3, r4));
    }

    //Writes with boundary checking

    __device__ static void partialWriteOut128Bytes(uchar *out, const uint &index,
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

    __device__ static void partialWriteOut128Bytes(char *out, const uint &index,
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

    __device__ static void partialWriteOut128Bytes(short *out, const uint &index,
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

    __device__ static void partialWriteOut128Bytes(ushort *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        partialWriteOut128Bytes((short*)(out), index, r1, r2, r3, r4, elements);
    }

    __device__ static void partialWriteOut128Bytes(int *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        if (index                < elements) {out[index]                = r1;}
        if (index +   blockDim.x < elements) {out[index +   blockDim.x] = r2;}
        if (index + 2*blockDim.x < elements) {out[index + 2*blockDim.x] = r3;}
        if (index + 3*blockDim.x < elements) {out[index + 3*blockDim.x] = r4;}
    }

    __device__ static void partialWriteOut128Bytes(uint *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        partialWriteOut128Bytes((int*)(out), index, r1, r2, r3, r4, elements);
    }

    __device__ static void partialWriteOut128Bytes(intl *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        intl c1 = r2;
        c1 = (c1<<32) | r1;
        intl c2 = r4;
        c2 = (c2<<32) | r3;
        if (index              < elements) {out[index]              = c1;}
        if (index + blockDim.x < elements) {out[index + blockDim.x] = c2;}
    }

    __device__ static void partialWriteOut128Bytes(uintl *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        partialWriteOut128Bytes((intl*)(out), index, r1, r2, r3, r4, elements);
    }

    __device__ static void partialWriteOut128Bytes(float *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        if (index                < elements) {out[index]                = 1.f - getFloat(r1);}
        if (index +   blockDim.x < elements) {out[index +   blockDim.x] = 1.f - getFloat(r2);}
        if (index + 2*blockDim.x < elements) {out[index + 2*blockDim.x] = 1.f - getFloat(r3);}
        if (index + 3*blockDim.x < elements) {out[index + 3*blockDim.x] = 1.f - getFloat(r4);}
    }

    __device__ static void partialWriteOut128Bytes(cfloat *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        if (index              < elements) {
            out[index].x              = 1.f - getFloat(r1);
            out[index].y              = 1.f - getFloat(r2);
        }
        if (index + blockDim.x < elements) {
            out[index + blockDim.x].x = 1.f - getFloat(r3);
            out[index + blockDim.x].y = 1.f - getFloat(r4);
        }
    }

    __device__ static void partialWriteOut128Bytes(double *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        if (index              < elements) {out[index]              = 1.0 - getDouble(r1, r2);}
        if (index + blockDim.x < elements) {out[index + blockDim.x] = 1.0 - getDouble(r3, r4);}
    }

    __device__ static void partialWriteOut128Bytes(cdouble *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        if (index < elements) {
            out[index].x = 1.0 - getDouble(r1, r2);
            out[index].y = 1.0 - getDouble(r3, r4);
        }
    }

    //Normalized writes with boundary checking

    __device__ static void partialBoxMullerWriteOut128Bytes(float *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        float n1, n2, n3, n4;
        boxMullerTransform(&n1, &n2, getFloat(r1), getFloat(r2));
        boxMullerTransform(&n3, &n4, getFloat(r3), getFloat(r4));
        if (index                < elements) {out[index]                = n1;}
        if (index +   blockDim.x < elements) {out[index +   blockDim.x] = n2;}
        if (index + 2*blockDim.x < elements) {out[index + 2*blockDim.x] = n3;}
        if (index + 3*blockDim.x < elements) {out[index + 3*blockDim.x] = n4;}
    }

    __device__ static void partialBoxMullerWriteOut128Bytes(cfloat *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        float n1, n2, n3, n4;
        boxMullerTransform(&n1, &n2, getFloat(r1), getFloat(r2));
        boxMullerTransform(&n3, &n4, getFloat(r3), getFloat(r4));
        if (index              < elements) {
            out[index].x              = n1;
            out[index].y              = n2;
        }
        if (index + blockDim.x < elements) {
            out[index + blockDim.x].x = n3;
            out[index + blockDim.x].y = n4;
        }
    }

    __device__ static void partialBoxMullerWriteOut128Bytes(double *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        double n1, n2;
        boxMullerTransform(&n1, &n2, getDouble(r1, r2), getDouble(r3, r4));
        if (index              < elements) {out[index]              = n1;}
        if (index + blockDim.x < elements) {out[index + blockDim.x] = n2;}
    }

    __device__ static void partialBoxMullerWriteOut128Bytes(cdouble *out, const uint &index,
            const uint &r1, const uint &r2, const uint &r3, const uint &r4, const uint &elements)
    {
        double n1, n2;
        boxMullerTransform(&n1, &n2, getDouble(r1, r2), getDouble(r3, r4));
        if (index < elements) {
            out[index].x = n1;
            out[index].y = n2;
        }
    }

    template <typename T>
    __global__ void uniformPhilox(T *out, uint hi, uint lo, uint hic, uint loc, uint elementsPerBlock, uint elements)
    {
        uint index = blockIdx.x*elementsPerBlock + threadIdx.x;
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
            partialWriteOut128Bytes(out, index, ctr[0], ctr[1], ctr[2], ctr[3], elements);
        }
    }

    template <typename T>
    __global__ void uniformThreefry(T *out, uint hi, uint lo, uint hic, uint loc, uint elementsPerBlock, uint elements)
    {
        uint index = blockIdx.x*elementsPerBlock + threadIdx.x;
        uint key[2] = {lo, hi};
        uint ctr[2] = {loc, hic};
        ctr[0] += index;
        ctr[1] += (ctr[0] < loc);
        uint o[4];
        if (blockIdx.x != (gridDim.x - 1)) {
            threefry(key, ctr, o);
            ctr[0] += elements;
            threefry(key, ctr, o + 2);
            writeOut128Bytes(out, index, o[0], o[1], o[2], o[3]);
        } else {
            threefry(key, ctr, o);
            ctr[0] += elements;
            threefry(key, ctr, o + 2);
            partialWriteOut128Bytes(out, index, o[0], o[1], o[2], o[3], elements);
        }
    }

    template <typename T>
    __global__ void uniformMersenne(T * const out,
            uint * const gState,
            const uint * const pos_tbl,
            const uint * const sh1_tbl,
            const uint * const sh2_tbl,
            uint mask,
            const uint * const g_recursion_table,
            const uint * const g_temper_table,
            uint elementsPerBlock, size_t elements)
    {
        __shared__ uint state[STATE_SIZE];
        __shared__ uint recursion_table[TABLE_SIZE];
        __shared__ uint temper_table[TABLE_SIZE];
        uint start = blockIdx.x*elementsPerBlock;
        uint end = start + elementsPerBlock;
        end = (end > elements)? elements : end;
        int elementsPerBlockIteration = (blockDim.x*4*sizeof(uint))/sizeof(T);
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
        int offsetX1 = (STATE_SIZE - N + threadIdx.x          ) % STATE_SIZE;
        int offsetX2 = (STATE_SIZE - N + threadIdx.x + 1      ) % STATE_SIZE;
        int offsetY  = (STATE_SIZE - N + threadIdx.x + pos    ) % STATE_SIZE;
        int offsetT  = (STATE_SIZE - N + threadIdx.x + pos - 1) % STATE_SIZE;
        int offsetO  = threadIdx.x;

        for (int i = 0; i < iter; ++i) {
            for (int ii = 0; ii < 4; ++ii) {
                uint r = recursion(recursion_table, mask, sh1, sh2,
                        state[offsetX1],
                        state[offsetX2],
                        state[offsetY ]);
                state[offsetO] = r;
                o[ii] = temper(temper_table, r, state[offsetT]);
                offsetX1 = (offsetX1 + blockDim.x) % STATE_SIZE;
                offsetX2 = (offsetX2 + blockDim.x) % STATE_SIZE;
                offsetY  = (offsetY  + blockDim.x) % STATE_SIZE;
                offsetT  = (offsetT  + blockDim.x) % STATE_SIZE;
                offsetO  = (offsetO  + blockDim.x) % STATE_SIZE;
                __syncthreads();
            }
            if (i == iter - 1) {
                partialWriteOut128Bytes(out, index+threadIdx.x, o[0], o[1], o[2], o[3], elements);
            } else {
                writeOut128Bytes(out, index+threadIdx.x, o[0], o[1], o[2], o[3]);
            }
            index += elementsPerBlockIteration;
        }
        state_write(gState, state);
    }

    template <typename T>
    __global__ void normalPhilox(T *out, uint hi, uint lo, uint hic, uint loc, uint elementsPerBlock, uint elements)
    {
        uint index = blockIdx.x*elementsPerBlock + threadIdx.x;
        uint key[2] = {lo, hi};
        uint ctr[4] = {loc, hic, 0, 0};
        ctr[0] += index;
        ctr[1] += (ctr[0] < loc);
        ctr[2] += (ctr[1] < hic);
        if (blockIdx.x != (gridDim.x - 1)) {
            philox(key, ctr);
            boxMullerWriteOut128Bytes(out, index, ctr[0], ctr[1], ctr[2], ctr[3]);
        } else {
            philox(key, ctr);
            partialBoxMullerWriteOut128Bytes(out, index, ctr[0], ctr[1], ctr[2], ctr[3], elements);
        }
    }

    template <typename T>
    __global__ void normalThreefry(T *out, uint hi, uint lo, uint hic, uint loc, uint elementsPerBlock, uint elements)
    {
        uint index = blockIdx.x*elementsPerBlock + threadIdx.x;
        uint key[2] = {lo, hi};
        uint ctr[2] = {loc, hic};
        ctr[0] += index;
        ctr[1] += (ctr[0] < loc);
        uint o[4];
        if (blockIdx.x != (gridDim.x - 1)) {
            threefry(key, ctr, o);
            ctr[0] += elements;
            threefry(key, ctr, o + 2);
            boxMullerWriteOut128Bytes(out, index, o[0], o[1], o[2], o[3]);
        } else {
            threefry(key, ctr, o);
            ctr[0] += elements;
            threefry(key, ctr, o + 2);
            partialBoxMullerWriteOut128Bytes(out, index, o[0], o[1], o[2], o[3], elements);
        }
    }

    template <typename T>
    __global__ void normalMersenne(T * const out,
            uint * const gState,
            const uint * const pos_tbl,
            const uint * const sh1_tbl,
            const uint * const sh2_tbl,
            uint mask,
            const uint * const g_recursion_table,
            const uint * const g_temper_table,
            uint elementsPerBlock, uint elements)
    {

        __shared__ uint state[STATE_SIZE];
        __shared__ uint recursion_table[TABLE_SIZE];
        __shared__ uint temper_table[TABLE_SIZE];
        uint start = blockIdx.x*elementsPerBlock;
        uint end = start + elementsPerBlock;
        end = (end > elements)? elements : end;
        int iter = divup((end - start)*sizeof(T), blockDim.x*4*sizeof(uint));

        uint pos = pos_tbl[blockIdx.x];
        uint sh1 = sh1_tbl[blockIdx.x];
        uint sh2 = sh2_tbl[blockIdx.x];
        state_read(state, gState);
        read_table(recursion_table, g_recursion_table);
        read_table(temper_table, g_temper_table);
        __syncthreads();

        uint index = start;
        int elementsPerBlockIteration = blockDim.x*4*sizeof(uint)/sizeof(T);
        uint o[4];
        int offsetX1 = (STATE_SIZE - N + threadIdx.x          ) % STATE_SIZE;
        int offsetX2 = (STATE_SIZE - N + threadIdx.x + 1      ) % STATE_SIZE;
        int offsetY  = (STATE_SIZE - N + threadIdx.x + pos    ) % STATE_SIZE;
        int offsetT  = (STATE_SIZE - N + threadIdx.x + pos - 1) % STATE_SIZE;
        int offsetO  = threadIdx.x;

        for (int i = 0; i < iter; ++i) {
            for (int ii = 0; ii < 4; ++ii) {
                uint r = recursion(recursion_table, mask, sh1, sh2,
                        state[offsetX1],
                        state[offsetX2],
                        state[offsetY ]);
                state[offsetO] = r;
                o[ii] = temper(temper_table, r, state[offsetT]);
                offsetX1 = (offsetX1 + blockDim.x) % STATE_SIZE;
                offsetX2 = (offsetX2 + blockDim.x) % STATE_SIZE;
                offsetY  = (offsetY  + blockDim.x) % STATE_SIZE;
                offsetT  = (offsetT  + blockDim.x) % STATE_SIZE;
                offsetO  = (offsetO  + blockDim.x) % STATE_SIZE;
                __syncthreads();
            }
            if (i == iter - 1) {
                partialBoxMullerWriteOut128Bytes(out, index+threadIdx.x, o[0], o[1], o[2], o[3], elements);
            } else {
                boxMullerWriteOut128Bytes(out, index+threadIdx.x, o[0], o[1], o[2], o[3]);
            }
            index += elementsPerBlockIteration;
        }
        state_write(gState, state);
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
        int threads = THREADS;
        int min_elements_per_block = 32*threads*4*sizeof(uint)/sizeof(T);
        int blocks = divup(elements, min_elements_per_block);
        blocks = (blocks > BLOCKS)? BLOCKS : blocks;
        uint elementsPerBlock = divup(elements, blocks);
        CUDA_LAUNCH(uniformMersenne, blocks, threads, out, state, pos, sh1, sh2, mask, recursion_table, temper_table, elementsPerBlock, elements);
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
        int threads = THREADS;
        int min_elements_per_block = 32*threads*4*sizeof(uint)/sizeof(T);
        int blocks = divup(elements, min_elements_per_block);
        blocks = (blocks > BLOCKS)? BLOCKS : blocks;
        uint elementsPerBlock = divup(elements, blocks);
        CUDA_LAUNCH(normalMersenne, blocks, threads, out, state, pos, sh1, sh2, mask, recursion_table, temper_table, elementsPerBlock, elements);
    }

    template <typename T>
    void uniformDistributionCBRNG(T* out, size_t elements, const af_random_engine_type type, const uintl &seed, uintl &counter)
    {
        int threads = THREADS;
        int elementsPerBlock = threads*4*sizeof(uint)/sizeof(T);
        int blocks = divup(elements, elementsPerBlock);
        uint hi = seed>>32;
        uint lo = seed;
        uint hic = counter>>32;
        uint loc = counter;
        switch (type) {
        case AF_RANDOM_ENGINE_PHILOX_4X32_10   :
            CUDA_LAUNCH(uniformPhilox, blocks, threads, out, hi, lo, hic, loc, elementsPerBlock, elements); break;
        case AF_RANDOM_ENGINE_THREEFRY_2X32_16 :
            CUDA_LAUNCH(uniformThreefry, blocks, threads, out, hi, lo, hic, loc, elementsPerBlock, elements); break;
        default : AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
        }
        counter += elements;
    }

    template <typename T>
    void normalDistributionCBRNG(T *out, size_t elements, const af_random_engine_type type, const uintl &seed, uintl &counter)
    {
        int threads = THREADS;
        int elementsPerBlock = threads*4*sizeof(uint)/sizeof(T);
        int blocks = divup(elements, elementsPerBlock);
        uint hi = seed>>32;
        uint lo = seed;
        uint hic = counter>>32;
        uint loc = counter;
        switch (type) {
        case AF_RANDOM_ENGINE_PHILOX_4X32_10   :
            CUDA_LAUNCH(normalPhilox, blocks, threads, out, hi, lo, hic, loc, elementsPerBlock, elements); break;
        case AF_RANDOM_ENGINE_THREEFRY_2X32_16 :
            CUDA_LAUNCH(normalThreefry, blocks, threads, out, hi, lo, hic, loc, elementsPerBlock, elements); break;
        default : AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
        }
        counter += elements;
    }
}
}

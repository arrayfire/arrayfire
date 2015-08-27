/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

namespace cuda
{
    namespace kernel
    {
        template<typename T>
        __device__ T atomicAdd(T *ptr, T val)
        {
            return ::atomicAdd(ptr, val);
        }

#define SPECIALIZE(T, fn1, fn2)                                         \
        template<>                                                      \
        __device__ T atomicAdd<T>(T* ptr, T val)                        \
        {                                                               \
            unsigned long long int* ptr_as_ull =                        \
                (unsigned long long int*)ptr;                           \
            unsigned long long int old = *ptr_as_ull, assumed;          \
            do {                                                        \
                assumed = old;                                          \
                old = atomicCAS(ptr_as_ull, assumed,                    \
                                fn2(val + fn1(assumed)));               \
            } while (assumed != old);                                   \
            return fn1(old);                                            \
        }                                                               \

        SPECIALIZE(double, __longlong_as_double, __double_as_longlong)
        SPECIALIZE(intl, intl, uintl)
        SPECIALIZE(uintl, uintl, uintl)

        template<>
        __device__ cfloat atomicAdd<cfloat>(cfloat *ptr, cfloat val)
        {
            float *fptr = (float *)(ptr);
            cfloat res;
            res.x = ::atomicAdd(fptr + 0, val.x);
            res.y = ::atomicAdd(fptr + 1, val.y);
            return res;
        }

        template<>
        __device__ cdouble atomicAdd<cdouble>(cdouble *ptr, cdouble val)
        {
            double *fptr = (double *)(ptr);
            cdouble res;
            res.x = atomicAdd(fptr + 0, val.x);
            res.y = atomicAdd(fptr + 1, val.y);
            return res;
        }
    }
}

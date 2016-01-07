/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <Array.hpp>
#include <random.hpp>
#include <err_cpu.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/random.hpp>

namespace cpu
{

template<typename T>
Array<T> randu(const af::dim4 &dims, const af::randomType &rtype)
{
    switch (rtype)  {
        case AF_RANDOM_PHILOX:
            CPU_NOT_SUPPORTED();
            break;
    }
    Array<T> outArray = createEmptyArray<T>(dims);
    getQueue().enqueue(kernel::randu<T>, outArray);
    return outArray;
}

#define INSTANTIATE_UNIFORM(T)                              \
    template Array<T>  randu<T>    (const af::dim4 &dims, const af::randomType &rtype);

INSTANTIATE_UNIFORM(float)
INSTANTIATE_UNIFORM(double)
INSTANTIATE_UNIFORM(cfloat)
INSTANTIATE_UNIFORM(cdouble)
INSTANTIATE_UNIFORM(int)
INSTANTIATE_UNIFORM(uint)
INSTANTIATE_UNIFORM(intl)
INSTANTIATE_UNIFORM(uintl)
INSTANTIATE_UNIFORM(uchar)
INSTANTIATE_UNIFORM(short)
INSTANTIATE_UNIFORM(ushort)

template<typename T>
Array<T> randn(const af::dim4 &dims)
{
    Array<T> outArray = createEmptyArray<T>(dims);
    getQueue().enqueue(kernel::randn<T>, outArray);
    return outArray;
}

#define INSTANTIATE_NORMAL(T)                              \
    template Array<T>  randn<T>(const af::dim4 &dims);

INSTANTIATE_NORMAL(float)
INSTANTIATE_NORMAL(double)
INSTANTIATE_NORMAL(cfloat)
INSTANTIATE_NORMAL(cdouble)

template<>
Array<char> randu(const af::dim4 &dims, const af::randomType &rtype)
{
    static unsigned long long my_seed = 0;
    if (kernel::is_first) {
        setSeed(kernel::gen_seed);
        my_seed = kernel::gen_seed;
    }

    static auto gen = kernel::urand<float>(kernel::generator);

    if (my_seed != kernel::gen_seed) {
        gen = kernel::urand<float>(kernel::generator);
        my_seed = kernel::gen_seed;
    }

    Array<char> outArray = createEmptyArray<char>(dims);
    auto func = [=](Array<char> outArray) {
        char *outPtr = outArray.get();
        for (int i = 0; i < (int)outArray.elements(); i++) {
            outPtr[i] = gen() > 0.5;
        }
    };
    getQueue().enqueue(func, outArray);

    return outArray;
}

void setSeed(const uintl seed)
{
    auto f = [=](const uintl seed){
        kernel::generator.seed(seed);
        kernel::is_first = false;
        kernel::gen_seed = seed;
    };
    getQueue().enqueue(f, seed);
}

uintl getSeed()
{
    getQueue().sync();
    return kernel::gen_seed;
}

}

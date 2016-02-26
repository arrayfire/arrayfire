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
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/random.hpp>

namespace cpu
{

template<typename T>
Array<T> randu(const af::dim4 &dims)
{
    Array<T> outArray = createEmptyArray<T>(dims);
    getQueue().enqueue(kernel::randu<T>, outArray);
    return outArray;
}

#define INSTANTIATE_UNIFORM(T)                              \
    template Array<T>  randu<T>    (const af::dim4 &dims);

INSTANTIATE_UNIFORM(float)
INSTANTIATE_UNIFORM(double)
INSTANTIATE_UNIFORM(cfloat)
INSTANTIATE_UNIFORM(cdouble)
INSTANTIATE_UNIFORM(int)
INSTANTIATE_UNIFORM(uint)
INSTANTIATE_UNIFORM(intl)
INSTANTIATE_UNIFORM(uintl)
INSTANTIATE_UNIFORM(uchar)
INSTANTIATE_UNIFORM(char)
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

void setSeed(const uintl seed)
{
    getQueue().enqueue(kernel::setSeed, seed);
}

uintl getSeed()
{
    uintl seed = 0;
    getQueue().enqueue(kernel::getSeedPtr, &seed);
    getQueue().sync();
    return seed;
}

}

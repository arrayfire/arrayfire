/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <type_traits>
#include <random>
#include <algorithm>
#include <functional>
#include <limits>
#include <type_traits>
#include <af/array.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <Array.hpp>
#include <random.hpp>

namespace cpu
{

using namespace std;

template<typename T>
using is_arithmetic_t       = typename enable_if< is_arithmetic<T>::value,      function<T()>>::type;
template<typename T>
using is_complex_t          = typename enable_if< is_complex<T>::value,         function<T()>>::type;
template<typename T>
using is_floating_point_t   = typename enable_if< is_floating_point<T>::value,  function<T()>>::type;

template<typename T, typename GenType>
is_arithmetic_t<T>
urand(GenType &generator)
{
    typedef typename conditional<   is_floating_point<T>::value,
                                    uniform_real_distribution<T>,
#if OS_WIN
                                    uniform_int_distribution<unsigned>>::type dist;
#else
                                    uniform_int_distribution<T >> ::type dist;
#endif
    return bind(dist(), generator);
}

template<typename T, typename GenType>
is_complex_t<T>
urand(GenType &generator)
{
    auto func = urand<typename T::value_type>(generator);
    return [func] () { return T(func(), func());};
}

template<typename T, typename GenType>
is_floating_point_t<T>
nrand(GenType &generator)
{
    return bind(normal_distribution<T>(), generator);
}

template<typename T, typename GenType>
is_complex_t<T>
nrand(GenType &generator)
{
    auto func = nrand<typename T::value_type>(generator);
    return [func] () { return T(func(), func());};
}

static default_random_engine generator;
static unsigned long long gen_seed = 0;
static bool is_first = true;
#define GLOBAL 1

template<typename T>
Array<T> randn(const af::dim4 &dims)
{
    static unsigned long long my_seed = 0;
    if (is_first) {
        setSeed(gen_seed);
        my_seed = gen_seed;
    }

    static auto gen = nrand<T>(generator);

    if (my_seed != gen_seed) {
        gen = nrand<T>(generator);
        my_seed = gen_seed;
    }

    Array<T> outArray = createEmptyArray<T>(dims);
    T *outPtr = outArray.get();
    for (int i = 0; i < (int)outArray.elements(); i++) {
        outPtr[i] = gen();
    }
    return outArray;
}

template<typename T>
Array<T> randu(const af::dim4 &dims)
{
    static unsigned long long my_seed = 0;
    if (is_first) {
        setSeed(gen_seed);
        my_seed = gen_seed;
    }

    static auto gen = urand<T>(generator);

    if (my_seed != gen_seed) {
        gen = urand<T>(generator);
        my_seed = gen_seed;
    }

    Array<T> outArray = createEmptyArray<T>(dims);
    T *outPtr = outArray.get();
    for (int i = 0; i < (int)outArray.elements(); i++) {
        outPtr[i] = gen();
    }
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

#define INSTANTIATE_NORMAL(T)                              \
    template Array<T>  randn<T>(const af::dim4 &dims);

INSTANTIATE_NORMAL(float)
INSTANTIATE_NORMAL(double)
INSTANTIATE_NORMAL(cfloat)
INSTANTIATE_NORMAL(cdouble)


template<>
Array<char> randu(const af::dim4 &dims)
{
    static unsigned long long my_seed = 0;
    if (is_first) {
        setSeed(gen_seed);
        my_seed = gen_seed;
    }

    static auto gen = urand<float>(generator);

    if (my_seed != gen_seed) {
        gen = urand<float>(generator);
        my_seed = gen_seed;
    }

    Array<char> outArray = createEmptyArray<char>(dims);
    char *outPtr = outArray.get();
    for (int i = 0; i < (int)outArray.elements(); i++) {
        outPtr[i] = gen() > 0.5;
    }
    return outArray;
}

void setSeed(const uintl seed)
{
    generator.seed(seed);
    is_first = false;
    gen_seed = seed;
}

uintl getSeed()
{
    return gen_seed;
}

}

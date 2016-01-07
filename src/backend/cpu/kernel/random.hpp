/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <Array.hpp>
#include <type_traits>
#include <random>
#include <algorithm>
#include <functional>
#include <limits>
#include <type_traits>

namespace cpu
{
namespace kernel
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

static mt19937 generator;
static unsigned long long gen_seed = 0;
static bool is_first = true;
#define GLOBAL 1

template<typename T>
void randn(Array<T> out)
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

    T *outPtr = out.get();
    for (int i = 0; i < (int)out.elements(); i++) {
        outPtr[i] = gen();
    }
}

template<typename T>
void randu(Array<T> out)
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

    T *outPtr = out.get();
    for (int i = 0; i < (int)out.elements(); i++) {
        outPtr[i] = gen();
    }
}

template<>
void randu(Array<char> out)
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

    char *outPtr = out.get();
    for (int i = 0; i < (int)out.elements(); i++) {
        outPtr[i] = gen() > 0.5;
    }
}

}
}

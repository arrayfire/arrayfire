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

#if defined(_WIN32)
    #define __THREAD_LOCAL static __declspec(thread)
#else
    #define __THREAD_LOCAL static __thread
#endif

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

mt19937& getGenerator()
{
    // FIXME: This abomination of a work around is brought to you
    // by incomplete standards from Xcode and Visual Studio
    // Should ideally be using thread_local on object instead of pointer
    __THREAD_LOCAL mt19937 *generator = NULL;
    if (generator == NULL) generator = new mt19937();
    return *generator;
}

unsigned long long& getSeed()
{
    __THREAD_LOCAL unsigned long long gen_seed = 0;
    return gen_seed;
}

void getSeedPtr(unsigned long long *seed)
{
    *seed = getSeed();
}

bool& isFirst()
{
    __THREAD_LOCAL bool is_first = true;
    return is_first;
}

void setSeed(const uintl seed)
{
    getGenerator().seed(seed);
    getSeed() = seed;
    isFirst() = false;
}

//FIXME: See if we can use functors instead of function pointer directly
template<typename T>
struct RandomDistribution
{
    std::function<T()> func;
    RandomDistribution(std::function<T()> dist_func) : func(dist_func)
    {
    }
};

template<typename T>
void randn(Array<T> out)
{
    __THREAD_LOCAL unsigned long long my_seed = 0;
    if (isFirst()) {
        my_seed = getSeed();
        setSeed(my_seed);
    }

    // FIXME: This abomination of a work around is brought to you
    // by incomplete standards from Xcode and Visual Studio
    // Should ideally be using thread_local on object instead of pointer
    __THREAD_LOCAL RandomDistribution<T> *distPtr = NULL;

    if (!distPtr || my_seed != getSeed()) {
        if (distPtr) delete distPtr;
        distPtr = new RandomDistribution<T>(nrand<T>(getGenerator()));
        my_seed = getSeed();
    }

    T *outPtr = out.get();
    for (int i = 0; i < (int)out.elements(); i++) {
        outPtr[i] = distPtr->func();
    }
}

template<typename T>
void randu(Array<T> out)
{
    __THREAD_LOCAL unsigned long long my_seed = 0;
    if (isFirst()) {
        my_seed = getSeed();
        setSeed(my_seed);
    }

    // FIXME: This abomination of a work around is brought to you
    // by incomplete standards from Xcode and Visual Studio
    // Should ideally be using thread_local on object instead of pointer
    __THREAD_LOCAL RandomDistribution<T> *distPtr = NULL;

    if (!distPtr || my_seed != getSeed()) {
        if (distPtr) delete distPtr;
        distPtr = new RandomDistribution<T>(urand<T>(getGenerator()));
        my_seed = getSeed();
    }

    T *outPtr = out.get();
    for (int i = 0; i < (int)out.elements(); i++) {
        outPtr[i] = distPtr->func();
    }
}

template<>
void randu(Array<char> out)
{
    __THREAD_LOCAL unsigned long long my_seed = 0;
    if (isFirst()) {
        my_seed = getSeed();
        setSeed(my_seed);
    }

    // FIXME: This abomination of a work around is brought to you
    // by incomplete standards from Xcode and Visual Studio
    // Should ideally be using thread_local on object instead of pointer
    __THREAD_LOCAL RandomDistribution<float> *distPtr = NULL;

    if (!distPtr || my_seed != getSeed()) {
        if (distPtr) delete distPtr;
        distPtr = new RandomDistribution<float>(nrand<float>(getGenerator()));
        my_seed = getSeed();
    }

    char *outPtr = out.get();
    for (int i = 0; i < (int)out.elements(); i++) {
        outPtr[i] = distPtr->func() > 0.5;
    }
}

}
}

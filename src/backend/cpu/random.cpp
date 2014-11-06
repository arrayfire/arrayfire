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

template<typename T>
Array<T>* randn(const af::dim4 &dims)
{
    Array<T> *outArray = createValueArray(dims, T(0));
    T *outPtr = outArray->get();

    default_random_engine generator;
    generate(outPtr, outPtr + outArray->elements(), nrand<T>(generator));

    return outArray;
}

template<typename T>
Array<T>* randu(const af::dim4 &dims)
{
    Array<T> *outArray = createValueArray(dims, T(0));
    T *outPtr = outArray->get();

    default_random_engine generator;
    generate(outPtr, outPtr + outArray->elements(), urand<T>(generator));

    return outArray;
}

#define INSTANTIATE_UNIFORM(T)                              \
    template Array<T>*  randu<T>    (const af::dim4 &dims);

INSTANTIATE_UNIFORM(float)
INSTANTIATE_UNIFORM(double)
INSTANTIATE_UNIFORM(cfloat)
INSTANTIATE_UNIFORM(cdouble)
INSTANTIATE_UNIFORM(int)
INSTANTIATE_UNIFORM(uint)
INSTANTIATE_UNIFORM(char)
INSTANTIATE_UNIFORM(uchar)

#define INSTANTIATE_NORMAL(T)                              \
    template Array<T>*  randn<T>(const af::dim4 &dims);

INSTANTIATE_NORMAL(float)
INSTANTIATE_NORMAL(double)
INSTANTIATE_NORMAL(cfloat)
INSTANTIATE_NORMAL(cdouble)

}

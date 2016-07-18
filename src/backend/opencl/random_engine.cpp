/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <kernel/random_engine.hpp>
#include <cassert>

namespace opencl
{
    template<typename T>
    Array<T> uniformDistribution(const af::dim4 &dims, const af_random_type type, const uintl seed, uintl &counter)
    {
        verifyDoubleSupport<T>();
        Array<T> out = createEmptyArray<T>(dims);

        switch(type) {
        case AF_RANDOM_PHILOX: kernel::uniformDistribution<T, AF_RANDOM_PHILOX>(*out.get(), out.elements(), seed, counter); break;
        case AF_RANDOM_THREEFRY: kernel::uniformDistribution<T, AF_RANDOM_THREEFRY>(*out.get(), out.elements(), seed, counter); break;
        }
        return out;
    }

#define COMPLEX_UNIFORM_DISTRIBUTION(T, TR)\
    template<>\
    Array<T> uniformDistribution<T>(const af::dim4 &dims, const af_random_type type, const uintl seed, uintl &counter)\
    {\
        verifyDoubleSupport<T>();\
        Array<T> out = createEmptyArray<T>(dims);\
\
        switch(type) {\
        case AF_RANDOM_PHILOX: kernel::uniformDistribution<TR, AF_RANDOM_PHILOX>(*out.get(), out.elements()*2, seed, counter); break;\
        case AF_RANDOM_THREEFRY: break;\
        }\
        return out;\
    }\

    COMPLEX_UNIFORM_DISTRIBUTION(cdouble, double)
    COMPLEX_UNIFORM_DISTRIBUTION(cfloat, float)

    template Array<float>   uniformDistribution<float>   (const af::dim4 &dim, const af_random_type type, const uintl seed, uintl &counter);
    template Array<double>  uniformDistribution<double>  (const af::dim4 &dim, const af_random_type type, const uintl seed, uintl &counter);
    template Array<int>     uniformDistribution<int>     (const af::dim4 &dim, const af_random_type type, const uintl seed, uintl &counter);
    template Array<uint>    uniformDistribution<uint>    (const af::dim4 &dim, const af_random_type type, const uintl seed, uintl &counter);
    template Array<intl>    uniformDistribution<intl>    (const af::dim4 &dim, const af_random_type type, const uintl seed, uintl &counter);
    template Array<uintl>   uniformDistribution<uintl>   (const af::dim4 &dim, const af_random_type type, const uintl seed, uintl &counter);
    template Array<char>    uniformDistribution<char>    (const af::dim4 &dim, const af_random_type type, const uintl seed, uintl &counter);
    template Array<uchar>   uniformDistribution<uchar>   (const af::dim4 &dim, const af_random_type type, const uintl seed, uintl &counter);
    template Array<short>   uniformDistribution<short>   (const af::dim4 &dim, const af_random_type type, const uintl seed, uintl &counter);
    template Array<ushort>  uniformDistribution<ushort>  (const af::dim4 &dim, const af_random_type type, const uintl seed, uintl &counter);

}

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
    Array<T> uniformDistribution(const af::dim4 &dims, const af_random_type type, const unsigned long long seed, unsigned long long &counter)
    {
        verifyDoubleSupport<T>();
        Array<T> out = createEmptyArray<T>(dims);

        switch(type) {
        case AF_RANDOM_PHILOX: kernel::uniformDistribution<T, AF_RANDOM_PHILOX>(*out.get(), out.elements(), seed, counter); break;
        case AF_RANDOM_THREEFRY: break;
        }
        return out;
    }

    template Array<float>   uniformDistribution<float>   (const af::dim4 &dim, const af_random_type type, const unsigned long long seed, unsigned long long &counter);
    template Array<double>  uniformDistribution<double>  (const af::dim4 &dim, const af_random_type type, const unsigned long long seed, unsigned long long &counter);
    template Array<cfloat>  uniformDistribution<cfloat>  (const af::dim4 &dim, const af_random_type type, const unsigned long long seed, unsigned long long &counter);
    template Array<cdouble> uniformDistribution<cdouble> (const af::dim4 &dim, const af_random_type type, const unsigned long long seed, unsigned long long &counter);
    template Array<int>     uniformDistribution<int>     (const af::dim4 &dim, const af_random_type type, const unsigned long long seed, unsigned long long &counter);
    template Array<uint>    uniformDistribution<uint>    (const af::dim4 &dim, const af_random_type type, const unsigned long long seed, unsigned long long &counter);
    template Array<intl>    uniformDistribution<intl>    (const af::dim4 &dim, const af_random_type type, const unsigned long long seed, unsigned long long &counter);
    template Array<uintl>   uniformDistribution<uintl>   (const af::dim4 &dim, const af_random_type type, const unsigned long long seed, unsigned long long &counter);
    template Array<char>    uniformDistribution<char>    (const af::dim4 &dim, const af_random_type type, const unsigned long long seed, unsigned long long &counter);
    template Array<uchar>   uniformDistribution<uchar>   (const af::dim4 &dim, const af_random_type type, const unsigned long long seed, unsigned long long &counter);
    template Array<short>   uniformDistribution<short>   (const af::dim4 &dim, const af_random_type type, const unsigned long long seed, unsigned long long &counter);
    template Array<ushort>  uniformDistribution<ushort>  (const af::dim4 &dim, const af_random_type type, const unsigned long long seed, unsigned long long &counter);

}

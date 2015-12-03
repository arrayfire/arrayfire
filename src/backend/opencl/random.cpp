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
#include <Array.hpp>
#include <random.hpp>
#include <cassert>
#include <kernel/random.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>

namespace opencl
{
    template<typename T>
    Array<T> randu(const af::dim4 &dims, const af::randomType &rtype)
    {
        /*
        switch (rtype)  {
            case AF_RANDOM_PHILOX:
                OPENCL_NOT_SUPPORTED();
                break;
        }
        */
        verifyDoubleSupport<T>();
        Array<T> out = createEmptyArray<T>(dims);
        kernel::random<T, true>(*out.get(), out.elements(), rtype);
        return out;
    }

    template<typename T>
    Array<T> randn(const af::dim4 &dims)
    {
        verifyDoubleSupport<T>();
        Array<T> out = createEmptyArray<T>(dims);
        kernel::random<T, false>(*out.get(), out.elements());
        return out;
    }

    template Array<float>  randu<float>   (const af::dim4 &dims, const af::randomType &rtype);
    template Array<double> randu<double>  (const af::dim4 &dims, const af::randomType &rtype);
    template Array<int>    randu<int>     (const af::dim4 &dims, const af::randomType &rtype);
    template Array<uint>   randu<uint>    (const af::dim4 &dims, const af::randomType &rtype);
    template Array<intl>   randu<intl>    (const af::dim4 &dims, const af::randomType &rtype);
    template Array<uintl>  randu<uintl>   (const af::dim4 &dims, const af::randomType &rtype);
    template Array<short>  randu<short>   (const af::dim4 &dims, const af::randomType &rtype);
    template Array<ushort> randu<ushort>  (const af::dim4 &dims, const af::randomType &rtype);
    template Array<char>   randu<char>    (const af::dim4 &dims, const af::randomType &rtype);
    template Array<uchar>  randu<uchar>   (const af::dim4 &dims, const af::randomType &rtype);

    template Array<float>  randn<float>   (const af::dim4 &dims);
    template Array<double> randn<double>  (const af::dim4 &dims);

#define COMPLEX_RANDU(T, TR)                                \
    template<> Array<T> randu<T>(const af::dim4 &dims,      \
            const af::randomType &rtype)                    \
    {                                                       \
        Array<T> out = createEmptyArray<T>(dims);           \
        dim_t elements = out.elements() * 2;                \
        kernel::random<TR, true>(*out.get(), elements);     \
        return out;                                         \
    }                                                       \

#define COMPLEX_RANDN(T, TR)                                \
    template<> Array<T> randn<T>(const af::dim4 &dims)      \
    {                                                       \
        Array<T> out = createEmptyArray<T>(dims);           \
        dim_t elements = out.elements() * 2;                \
        kernel::random<TR, false>(*out.get(), elements);    \
        return out;                                         \
    }                                                       \

    COMPLEX_RANDU(cfloat, float)
    COMPLEX_RANDU(cdouble, double)
    COMPLEX_RANDN(cfloat, float)
    COMPLEX_RANDN(cdouble, double)


    void setSeed(const uintl seed)
    {
        uintl hi = (seed & 0xffffffff00000000) >> 32;
        uintl lo = (seed & 0x00000000ffffffff);
        kernel::random_seed[0] = (unsigned)hi;
        kernel::random_seed[1] = (unsigned)lo;
        kernel::counter = 0;
    }

    uintl getSeed()
    {
        uintl hi = kernel::random_seed[0];
        uintl lo = kernel::random_seed[1];
        return hi << 32 | lo;
    }
}

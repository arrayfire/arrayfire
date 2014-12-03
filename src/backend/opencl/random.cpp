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

namespace opencl
{
    template<typename T>
    Array<T>* randu(const af::dim4 &dims)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !isDoubleSupported(getActiveDeviceId())) {
            OPENCL_NOT_SUPPORTED();
        }
        Array<T> *out = createEmptyArray<T>(dims);
        kernel::random<T, true>(out->get(), out->elements());
        return out;
    }

    template<typename T>
    Array<T>* randn(const af::dim4 &dims)
    {
        if ((std::is_same<T, double>::value || std::is_same<T, cdouble>::value) &&
            !isDoubleSupported(getActiveDeviceId())) {
            OPENCL_NOT_SUPPORTED();
        }
        Array<T> *out = createEmptyArray<T>(dims);
        kernel::random<T, false>(out->get(), out->elements());
        return out;
    }

    template Array<float>  * randu<float>   (const af::dim4 &dims);
    template Array<double> * randu<double>  (const af::dim4 &dims);
    template Array<int>    * randu<int>     (const af::dim4 &dims);
    template Array<uint>   * randu<uint>    (const af::dim4 &dims);
    template Array<char>   * randu<char>    (const af::dim4 &dims);
    template Array<uchar>  * randu<uchar>   (const af::dim4 &dims);

    template Array<float>  * randn<float>   (const af::dim4 &dims);
    template Array<double> * randn<double>  (const af::dim4 &dims);

#define COMPLEX_RANDOM(fn, T, TR)                           \
    template<> Array<T>* fn<T>(const af::dim4 &dims)        \
    {                                                       \
        Array<T> *out = createEmptyArray<T>(dims);          \
        dim_type elements = out->elements() * 2;            \
        kernel::random<TR, false>(out->get(), elements);    \
        return out;                                         \
    }                                                       \

    COMPLEX_RANDOM(randu, cfloat, float)
    COMPLEX_RANDOM(randu, cdouble, double)
    COMPLEX_RANDOM(randn, cfloat, float)
    COMPLEX_RANDOM(randn, cdouble, double)

}

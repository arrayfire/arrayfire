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
#include <kernel/random.hpp>
#include <cassert>

namespace cuda
{
    template<typename T>
    Array<T> randu(const af::dim4 &dims)
    {
        if (!kernel::is_init[getActiveDeviceId()]) kernel::setup_states();
        Array<T> out = createEmptyArray<T>(dims);
        kernel::randu(out.get(), out.elements());
        return out;
    }

    template<typename T>
    Array<T> randn(const af::dim4 &dims)
    {
        if (!kernel::is_init[getActiveDeviceId()]) kernel::setup_states();
        Array<T> out  = createEmptyArray<T>(dims);
        kernel::randn(out.get(), out.elements());
        return out;
    }

    template Array<float>   randu<float>   (const af::dim4 &dims);
    template Array<double>  randu<double>  (const af::dim4 &dims);
    template Array<cfloat>  randu<cfloat>  (const af::dim4 &dims);
    template Array<cdouble> randu<cdouble> (const af::dim4 &dims);
    template Array<int>     randu<int>     (const af::dim4 &dims);
    template Array<uint>    randu<uint>    (const af::dim4 &dims);
    template Array<intl>    randu<intl>    (const af::dim4 &dims);
    template Array<uintl>   randu<uintl>   (const af::dim4 &dims);
    template Array<char>    randu<char>    (const af::dim4 &dims);
    template Array<uchar>   randu<uchar>   (const af::dim4 &dims);

    template Array<float>   randn<float>   (const af::dim4 &dims);
    template Array<double>  randn<double>  (const af::dim4 &dims);
    template Array<cfloat>  randn<cfloat>  (const af::dim4 &dims);
    template Array<cdouble> randn<cdouble> (const af::dim4 &dims);


    void setSeed(const uintl seed)
    {
        kernel::seed = seed;
        kernel::setup_states();
    }

    uintl getSeed()
    {
        return kernel::seed;
    }


}

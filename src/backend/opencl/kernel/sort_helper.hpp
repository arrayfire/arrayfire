/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <type_traits>

namespace opencl
{
    namespace kernel
    {
        using std::conditional;
        using std::is_same;

        // If type is cdouble, return std::complex<double>, else return T
        template<typename T>
        using ztype_t = typename conditional<is_same<T, cdouble>::value,
                                             std::complex<double>, T
                                            >::type;

        // If type is cfloat, return std::complex<float>, else return ztype_t
        template<typename T>
        using ctype_t = typename conditional<is_same<T, cfloat>::value,
                                             std::complex<float>, ztype_t<T>
                                            >::type;

        // If type is intl, return cl_long, else return ctype_t
        template<typename T>
        using ltype_t = typename conditional<is_same<T, intl>::value,
                                             cl_long, ctype_t<T>
                                            >::type;

        // If type is uintl, return cl_ulong, else return ltype_t
        template<typename T>
        using type_t = typename conditional<is_same<T, uintl>::value,
                                            cl_ulong, ltype_t<T>
                                           >::type;
    }
}


/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/traits.hpp>
#include <cuComplex.h>

namespace af {

template<>
struct dtype_traits<cuFloatComplex> {
    enum { af_type = c32 };
    typedef float base_type;
    static const char* getName() { return "cuFloatComplex"; }
};

template<>
struct dtype_traits<cuDoubleComplex> {
    enum { af_type = c64 };
    typedef double base_type;
    static const char* getName() { return "cuDoubleComplex"; }
};

}

using af::dtype_traits;

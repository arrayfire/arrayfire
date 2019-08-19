/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/traits.hpp>
#include <cuComplex.h>
#include <cuda_fp16.h>

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

template<>
struct dtype_traits<__half> {
    enum { af_type = f16 };
    typedef __half base_type;
    static const char* getName() { return "__half"; }
};

}  // namespace af

using af::dtype_traits;

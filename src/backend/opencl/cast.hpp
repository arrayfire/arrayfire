/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <common/jit/UnaryNode.hpp>
#include <err_opencl.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <traits.hpp>
#include <types.hpp>
#include <af/dim4.hpp>
#include <complex>

namespace arrayfire {
namespace opencl {

template<typename To, typename Ti>
struct CastOp {
    const char *name() { return ""; }
};

#define CAST_FN(TYPE)                                   \
    template<typename Ti>                               \
    struct CastOp<TYPE, Ti> {                           \
        const char *name() { return "convert_" #TYPE; } \
    };

CAST_FN(int)
CAST_FN(uint)
CAST_FN(uchar)
CAST_FN(float)
CAST_FN(double)

#define CAST_CFN(TYPE)                                    \
    template<typename Ti>                                 \
    struct CastOp<TYPE, Ti> {                             \
        const char *name() { return "__convert_" #TYPE; } \
    };

CAST_CFN(cfloat)
CAST_CFN(cdouble)
CAST_CFN(char)

template<>
struct CastOp<cfloat, cdouble> {
    const char *name() { return "__convert_z2c"; }
};

template<>
struct CastOp<cdouble, cfloat> {
    const char *name() { return "__convert_c2z"; }
};

template<>
struct CastOp<cfloat, cfloat> {
    const char *name() { return "__convert_c2c"; }
};

template<>
struct CastOp<cdouble, cdouble> {
    const char *name() { return "__convert_z2z"; }
};

#undef CAST_FN
#undef CAST_CFN

}  // namespace opencl
}  // namespace arrayfire

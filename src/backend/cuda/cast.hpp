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
#include <common/half.hpp>
#include <common/jit/UnaryNode.hpp>
#include <err_cuda.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <types.hpp>
#include <af/dim4.hpp>
#include <complex>

namespace cuda {

template<typename To, typename Ti>
struct CastOp {
    const char *name() { return ""; }
};

#define CAST_FN(TYPE)                                \
    template<typename Ti>                            \
    struct CastOp<TYPE, Ti> {                        \
        const char *name() { return "(" #TYPE ")"; } \
    };

CAST_FN(int)
CAST_FN(unsigned int)
CAST_FN(unsigned char)
CAST_FN(unsigned short)
CAST_FN(short)
CAST_FN(float)
CAST_FN(double)

template<typename Ti>
struct CastOp<common::half, Ti> {
    const char *name() { return "(__half)"; }
};

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

// Casting from half to unsigned char causes compilation issues. First convert
// to short then to half
template<>
struct CastOp<unsigned char, common::half> {
    const char *name() { return "(short)"; }
};

#undef CAST_FN
#undef CAST_CFN

template<typename To, typename Ti>
struct CastWrapper {
    Array<To> operator()(const Array<Ti> &in) {
        CastOp<To, Ti> cop;
        common::Node_ptr in_node = in.getNode();
        common::UnaryNode *node =
            new common::UnaryNode(getFullName<To>(), shortname<To>(true),
                                  cop.name(), in_node, af_cast_t);
        return createNodeArray<To>(in.dims(), common::Node_ptr(node));
    }
};

template<typename T>
struct CastWrapper<T, T> {
    Array<T> operator()(const Array<T> &in) { return in; }
};

template<typename To, typename Ti>
Array<To> cast(const Array<Ti> &in) {
    CastWrapper<To, Ti> cast_op;
    return cast_op(in);
}

}  // namespace cuda

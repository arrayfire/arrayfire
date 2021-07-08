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

template<typename To, typename Ti>
struct CastWrapper {
    Array<To> operator()(const Array<Ti> &in) {
        CastOp<To, Ti> cop;
        common::Node_ptr in_node = in.getNode();
        af::dtype to_dtype = static_cast<af::dtype>(dtype_traits<To>::af_type);

        // JIT optimization in the cast of multiple sequential casts that become
        // idempotent - check to see if the previous operation was also a cast
        // TODO: handle arbitrarily long chains of casts
        auto in_node_unary =
            std::dynamic_pointer_cast<common::UnaryNode>(in_node);
        if (in_node_unary && in_node_unary->getOp() == af_cast_t) {
            // child child's output type is the input type of the child
            auto in_in_node = in_node_unary->getChildren()[0];
            if (in_in_node->getType() == to_dtype) {
                // ignore the input node and simply connect a noop node from the
                // child's child to produce this op's output
                return createNodeArray<To>(in.dims(), in_in_node);
            }
        }

        common::UnaryNode *node =
            new common::UnaryNode(to_dtype, cop.name(), in_node, af_cast_t);
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

}  // namespace opencl

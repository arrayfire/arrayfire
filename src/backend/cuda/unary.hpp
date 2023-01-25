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
#include <common/jit/NaryNode.hpp>
#include <common/jit/UnaryNode.hpp>
#include <math.hpp>
#include <optypes.hpp>

namespace arrayfire {
namespace cuda {

template<af_op_t op>
static const char *unaryName();

#define UNARY_DECL(OP, FNAME)                     \
    template<>                                    \
    inline const char *unaryName<af_##OP##_t>() { \
        return FNAME;                             \
    }

#define UNARY_FN(OP) UNARY_DECL(OP, #OP)

UNARY_FN(sin)
UNARY_FN(cos)
UNARY_FN(tan)

UNARY_FN(asin)
UNARY_FN(acos)
UNARY_FN(atan)

UNARY_FN(sinh)
UNARY_FN(cosh)
UNARY_FN(tanh)

UNARY_FN(asinh)
UNARY_FN(acosh)
UNARY_FN(atanh)

UNARY_FN(exp)
UNARY_DECL(sigmoid, "__sigmoid")
UNARY_FN(expm1)
UNARY_FN(erf)
UNARY_FN(erfc)

UNARY_FN(tgamma)
UNARY_FN(lgamma)

UNARY_FN(log)
UNARY_FN(log1p)
UNARY_FN(log10)
UNARY_FN(log2)

UNARY_FN(sqrt)
UNARY_FN(rsqrt)
UNARY_FN(cbrt)

UNARY_FN(trunc)
UNARY_FN(round)
UNARY_FN(signbit)
UNARY_FN(ceil)
UNARY_FN(floor)

UNARY_DECL(bitnot, "__bitnot")
UNARY_DECL(isinf, "__isinf")
UNARY_DECL(isnan, "__isnan")
UNARY_FN(iszero)
UNARY_DECL(noop, "__noop")

#undef UNARY_DECL
#undef UNARY_FN

template<typename T, af_op_t op>
Array<T> unaryOp(const Array<T> &in, dim4 outDim = dim4(-1, -1, -1, -1)) {
    using arrayfire::common::Node;
    using arrayfire::common::Node_ptr;
    using std::array;

    auto createUnary = [](array<Node_ptr, 1> &operands) {
        return common::Node_ptr(new common::UnaryNode(
            static_cast<af::dtype>(af::dtype_traits<T>::af_type),
            unaryName<op>(), operands[0], op));
    };

    if (outDim == dim4(-1, -1, -1, -1)) { outDim = in.dims(); }
    Node_ptr out = common::createNaryNode<T, 1>(outDim, createUnary, {&in});
    return createNodeArray<T>(outDim, out);
}

template<typename T, af_op_t op>
Array<char> checkOp(const Array<T> &in, dim4 outDim = dim4(-1, -1, -1, -1)) {
    using arrayfire::common::Node_ptr;

    auto createUnary = [](std::array<Node_ptr, 1> &operands) {
        return Node_ptr(new common::UnaryNode(
            static_cast<af::dtype>(dtype_traits<char>::af_type),
            unaryName<op>(), operands[0], op));
    };

    if (outDim == dim4(-1, -1, -1, -1)) { outDim = in.dims(); }
    Node_ptr out = common::createNaryNode<T, 1>(outDim, createUnary, {&in});
    return createNodeArray<char>(outDim, out);
}

}  // namespace cuda
}  // namespace arrayfire

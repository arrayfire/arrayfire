/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <optypes.hpp>
#include <math.hpp>
#include <JIT/UnaryNode.hpp>

namespace opencl
{

template<af_op_t op>
static const char *unaryName() { return "noop"; }

#define UNARY_DECL(OP, FNAME)                   \
    template<>                                  \
    const char *unaryName<af_##OP##_t>()        \
    {                                           \
        return FNAME;                           \
    }                                           \


UNARY_DECL(abs, "fabs")

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
UNARY_FN(expm1)
UNARY_FN(erf)
UNARY_FN(erfc)

UNARY_FN(tgamma)
UNARY_FN(lgamma)

UNARY_FN(log)
UNARY_FN(log1p)
UNARY_FN(log10)

UNARY_FN(sqrt)
UNARY_FN(cbrt)

UNARY_FN(round)
UNARY_FN(ceil)
UNARY_FN(floor)

template<typename T, af_op_t op>
Array<T>* unaryOp(const Array<T> &in)
{
    JIT::Node *in_node = in.getNode();

    JIT::UnaryNode *node = new JIT::UnaryNode(dtype_traits<T>::getName(),
                                              unaryName<op>(),
                                              in_node, op);

    return createNodeArray<T>(in.dims(), reinterpret_cast<JIT::Node *>(node));
}

}

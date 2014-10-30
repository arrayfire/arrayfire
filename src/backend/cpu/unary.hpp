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
#include <err_cpu.hpp>
#include <TNJ/UnaryNode.hpp>
#include <cmath>

namespace cpu
{

#define UNARY_FN(op)                            \
    template<typename T>                        \
    struct UnOp<T, T, af_##op##_t>              \
    {                                           \
        T eval(T in)                            \
        {                                       \
            return op(in);                      \
        }                                       \
    };                                          \

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

UNARY_FN(log)
UNARY_FN(log10)
UNARY_FN(log1p)

UNARY_FN(sqrt)
UNARY_FN(cbrt)

UNARY_FN(tgamma)
UNARY_FN(lgamma)

#undef UNARY_FN

    template<typename T, af_op_t op>
    Array<T>* unaryOp(const Array<T> &in)
    {
        TNJ::Node *in_node = in.getNode();
        TNJ::UnaryNode<T, T, op> *node = new TNJ::UnaryNode<T, T, op>(in_node);

        return createNodeArray<T>(in.dims(), reinterpret_cast<TNJ::Node *>(node));
    }

}

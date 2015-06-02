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
#include <err_cpu.hpp>
#include <TNJ/UnaryNode.hpp>
#include <cmath>

namespace cpu
{
#define sign(in) std::signbit(in)

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

UNARY_FN(round)
UNARY_FN(trunc)
UNARY_FN(sign )
UNARY_FN(floor)
UNARY_FN(ceil)

UNARY_FN(exp)
UNARY_FN(expm1)
UNARY_FN(erf)
UNARY_FN(erfc)

UNARY_FN(log)
UNARY_FN(log10)
UNARY_FN(log1p)
UNARY_FN(log2)

UNARY_FN(sqrt)
UNARY_FN(cbrt)

UNARY_FN(tgamma)
UNARY_FN(lgamma)

#undef UNARY_FN
#undef sign

    template<typename T, af_op_t op>
    Array<T> unaryOp(const Array<T> &in)
    {
        TNJ::Node_ptr in_node = in.getNode();
        TNJ::UnaryNode<T, T, op> *node = new TNJ::UnaryNode<T, T, op>(in_node);

        return createNodeArray<T>(in.dims(),
                                  TNJ::Node_ptr(reinterpret_cast<TNJ::Node *>(node)));
    }

#define iszero(a) ((a) == 0)

#define CHECK_FN(name ,op)                      \
    template<typename T>                        \
    struct UnOp<char, T, af_##name##_t>         \
    {                                           \
        char eval(T in)                         \
        {                                       \
            return op(in);                      \
        }                                       \
    };                                          \

    CHECK_FN(isinf, std::isinf)
    CHECK_FN(isnan, std::isnan)
    CHECK_FN(iszero, iszero)

    template<typename T, af_op_t op>
    Array<char> checkOp(const Array<T> &in)
    {
        TNJ::Node_ptr in_node = in.getNode();
        TNJ::UnaryNode<char, T, op> *node = new TNJ::UnaryNode<char, T, op>(in_node);

        return createNodeArray<char>(in.dims(),
                                     TNJ::Node_ptr(reinterpret_cast<TNJ::Node *>(node)));
    }

}

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
#include <err_cuda.hpp>
#include <JIT/UnaryNode.hpp>

namespace cuda
{

template<typename T, af_op_t op>
struct UnOp
{
    const char *name()
    {
        return "noop";
    }
};

#define UNARY_FN(fn)                            \
    template<typename T>                        \
    struct UnOp<T, af_##fn##_t>                 \
    {                                           \
        std::string sn;                         \
        UnOp() : sn(shortname<T>(false)) {      \
            sn = std::string("@___"#fn) + sn;   \
        }                                       \
        const char *name()                      \
        {                                       \
            return sn.c_str();                  \
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
UNARY_FN(abs)

#undef UNARY_FN

    template<typename T, af_op_t op>
    Array<T>* unaryOp(const Array<T> &in)
    {

        UnOp<T, op> uop;

        JIT::Node *in_node = in.getNode();

        JIT::UnaryNode *node = new JIT::UnaryNode(irname<T>(),
                                                  uop.name(),
                                                  in_node, op);

        return createNodeArray<T>(in.dims(), reinterpret_cast<JIT::Node *>(node));
    }

}

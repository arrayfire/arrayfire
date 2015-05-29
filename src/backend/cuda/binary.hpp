/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/dim4.hpp>
#include <Array.hpp>
#include <optypes.hpp>
#include <math.hpp>
#include <JIT/BinaryNode.hpp>

namespace cuda
{


template<typename To, typename Ti, af_op_t op>
struct BinOp
{
    const char *name()
    {
        return "noop";
    }
};

#define BINARY(fn)                                  \
    template<typename To, typename Ti>              \
    struct BinOp<To, Ti, af_##fn##_t>               \
    {                                               \
        std::string res;                            \
        BinOp() :                                   \
            res(cuMangledName<Ti, true>("___"#fn))  \
        {}                                          \
        const std::string name()                    \
        {                                           \
            return res;                             \
        }                                           \
    };

BINARY(add)
BINARY(sub)
BINARY(mul)
BINARY(div)
BINARY(and)
BINARY(or)
BINARY(bitand)
BINARY(bitor)
BINARY(bitxor)
BINARY(bitshiftl)
BINARY(bitshiftr)

BINARY(lt)
BINARY(gt)
BINARY(le)
BINARY(ge)
BINARY(eq)
BINARY(neq)

BINARY(max)
BINARY(min)
BINARY(pow)
BINARY(mod)
BINARY(rem)
BINARY(atan2)
BINARY(hypot)

#undef BINARY

template<typename To, typename Ti, af_op_t op>
Array<To> createBinaryNode(const Array<Ti> &lhs, const Array<Ti> &rhs, const af::dim4 &odims)
{
    BinOp<To, Ti, op> bop;

    JIT::Node_ptr lhs_node = lhs.getNode();
    JIT::Node_ptr rhs_node = rhs.getNode();

    JIT::BinaryNode *node = new JIT::BinaryNode(irname<To>(),
                                                afShortName<To>(),
                                                bop.name(),
                                                lhs_node,
                                                rhs_node, (int)(op));

    return createNodeArray<To>(odims, JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
}

}

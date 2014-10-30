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
        std::string sn;                             \
        BinOp() : sn(shortname<Ti>(false)) {        \
            sn = std::string("@___"#fn) + sn + sn;  \
        }                                           \
        const char *name()                          \
        {                                           \
            return sn.c_str();                      \
        }                                           \
    };

BINARY(add)
BINARY(sub)
BINARY(mul)
BINARY(div)

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

#undef BINARY

template<typename To, typename Ti, af_op_t op>
Array<To> *createBinaryNode(const Array<Ti> &lhs, const Array<Ti> &rhs)
{
    BinOp<To, Ti, op> bop;

    JIT::Node *lhs_node = lhs.getNode();
    JIT::Node *rhs_node = rhs.getNode();

    JIT::BinaryNode *node = new JIT::BinaryNode(irname<To>(),
                                                bop.name(),
                                                lhs_node,
                                                rhs_node, (int)(op));

    return createNodeArray<To>(lhs.dims(), reinterpret_cast<JIT::Node *>(node));
}

}

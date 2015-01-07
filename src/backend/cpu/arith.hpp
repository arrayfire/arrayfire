/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <optypes.hpp>
#include <err_cpu.hpp>
#include <cmath>
#include <TNJ/BinaryNode.hpp>

namespace cpu
{

#define ARITH_FN(OP, op)                        \
    template<typename T>                        \
    struct BinOp<T, T, OP>                      \
    {                                           \
        T eval(T lhs, T rhs)                    \
        {                                       \
            return lhs op rhs;                  \
        }                                       \
    };                                          \


ARITH_FN(af_add_t, +)
ARITH_FN(af_sub_t, -)
ARITH_FN(af_mul_t, *)
ARITH_FN(af_div_t, /)

#undef ARITH_FN

#define NUMERIC_FN(OP, FN)                      \
    template<typename T>                        \
    struct BinOp<T, T, OP>                      \
    {                                           \
        T eval(T lhs, T rhs)                    \
        {                                       \
            return FN(lhs, rhs);                \
        }                                       \
    };                                          \

NUMERIC_FN(af_max_t, ::max)
NUMERIC_FN(af_min_t, ::min)
NUMERIC_FN(af_mod_t, fmod)
NUMERIC_FN(af_pow_t, pow)
NUMERIC_FN(af_rem_t, remainder)
NUMERIC_FN(af_atan2_t, atan2)
NUMERIC_FN(af_hypot_t, hypot)

    template<typename T, af_op_t op>
    Array<T>* arithOp(const Array<T> &lhs, const Array<T> &rhs, const af::dim4 &odims)
    {
        TNJ::Node_ptr lhs_node = lhs.getNode();
        TNJ::Node_ptr rhs_node = rhs.getNode();

        TNJ::BinaryNode<T, T, op> *node = new TNJ::BinaryNode<T, T, op>(lhs_node, rhs_node);

        return createNodeArray<T>(odims, TNJ::Node_ptr(
                                      reinterpret_cast<TNJ::Node *>(node)));
    }
}

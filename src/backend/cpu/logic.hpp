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
#include <Array.hpp>
#include <optypes.hpp>
#include <err_cpu.hpp>
#include <types.hpp>
#include <TNJ/BinaryNode.hpp>

namespace cpu
{

#define LOGIC_FN(OP, op)                        \
    template<typename T>                        \
    struct BinOp<uchar, T, OP>                  \
    {                                           \
        uchar eval(T lhs, T rhs)                \
        {                                       \
            return lhs op rhs;                  \
        }                                       \
    };                                          \


LOGIC_FN(af_eq_t, ==)
LOGIC_FN(af_neq_t, !=)
LOGIC_FN(af_lt_t, <)
LOGIC_FN(af_gt_t, >)
LOGIC_FN(af_le_t, <=)
LOGIC_FN(af_ge_t, >=)

#undef LOGIC_FN

#define LOGIC_CPLX_FN(T, OP, op)                    \
    template<>                                      \
    struct BinOp<uchar, std::complex<T>, OP>        \
    {                                               \
        uchar eval(std::complex<T> lhs,             \
                   std::complex<T> rhs)             \
        {                                           \
            return std::abs(lhs) op std::abs(rhs);  \
        }                                           \
    };                                              \

LOGIC_CPLX_FN(float, af_lt_t, <)
LOGIC_CPLX_FN(float, af_le_t, <=)
LOGIC_CPLX_FN(float, af_gt_t, >)
LOGIC_CPLX_FN(float, af_ge_t, >=)


LOGIC_CPLX_FN(double, af_lt_t, <)
LOGIC_CPLX_FN(double, af_le_t, <=)
LOGIC_CPLX_FN(double, af_gt_t, >)
LOGIC_CPLX_FN(double, af_ge_t, >=)

#undef LOGIC_CPLX_FN

    template<typename T, af_op_t op>
    Array<uchar>* logicOp(const Array<T> &lhs, const Array<T> &rhs)
    {
        TNJ::Node *lhs_node = lhs.getNode();
        TNJ::Node *rhs_node = rhs.getNode();

        TNJ::BinaryNode<uchar, T, op> *node = new TNJ::BinaryNode<uchar, T, op>(lhs_node, rhs_node);

        return createNodeArray<uchar>(lhs.dims(), reinterpret_cast<TNJ::Node *>(node));
    }
}

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <optypes.hpp>
#include <err_cpu.hpp>
#include <types.hpp>
#include <TNJ/BinaryNode.hpp>

namespace cpu
{

#define LOGIC_FN(OP, op)                        \
    template<typename T>                        \
    struct BinOp<char, T, OP>                   \
    {                                           \
        void eval(TNJ::array<char> &out,        \
                  const TNJ::array<T> &lhs,     \
                  const TNJ::array<T> &rhs,     \
                  int lim)                      \
        {                                       \
            for (int i = 0; i < lim; i++) {     \
                out[i] = lhs[i] op rhs[i];      \
            }                                   \
        }                                       \
    };                                          \


    LOGIC_FN(af_eq_t, ==)
    LOGIC_FN(af_neq_t, !=)
    LOGIC_FN(af_lt_t, <)
    LOGIC_FN(af_gt_t, >)
    LOGIC_FN(af_le_t, <=)
    LOGIC_FN(af_ge_t, >=)
    LOGIC_FN(af_and_t, &&)
    LOGIC_FN(af_or_t, ||)

#undef LOGIC_FN

#define LOGIC_CPLX_FN(T, OP, op)                \
    template<>                                  \
    struct BinOp<char, std::complex<T>, OP>     \
    {                                           \
        typedef std::complex<T> Ti;             \
        void eval(TNJ::array<char> &out,        \
                  const TNJ::array<Ti> &lhs,    \
                  const TNJ::array<Ti> &rhs,    \
                  int lim)                      \
        {                                       \
            for (int i = 0; i < lim; i++) {     \
                T lhs_mag = std::abs(lhs[i]);   \
                T rhs_mag = std::abs(rhs[i]);   \
                out[i] = lhs_mag op rhs_mag;    \
            }                                   \
        }                                       \
    };                                          \

LOGIC_CPLX_FN(float, af_lt_t, <)
LOGIC_CPLX_FN(float, af_le_t, <=)
LOGIC_CPLX_FN(float, af_gt_t, >)
LOGIC_CPLX_FN(float, af_ge_t, >=)
LOGIC_CPLX_FN(float, af_and_t, &&)
LOGIC_CPLX_FN(float, af_or_t, ||)


LOGIC_CPLX_FN(double, af_lt_t, <)
LOGIC_CPLX_FN(double, af_le_t, <=)
LOGIC_CPLX_FN(double, af_gt_t, >)
LOGIC_CPLX_FN(double, af_ge_t, >=)
LOGIC_CPLX_FN(double, af_and_t, &&)
LOGIC_CPLX_FN(double, af_or_t, ||)

#undef LOGIC_CPLX_FN

    template<typename T, af_op_t op>
    Array<char> logicOp(const Array<T> &lhs, const Array<T> &rhs, const af::dim4 &odims)
    {
        TNJ::Node_ptr lhs_node = lhs.getNode();
        TNJ::Node_ptr rhs_node = rhs.getNode();

        TNJ::BinaryNode<char, T, op> *node = new TNJ::BinaryNode<char, T, op>(lhs_node, rhs_node);

        return createNodeArray<char>(odims, TNJ::Node_ptr(
                                          reinterpret_cast<TNJ::Node *>(node)));
    }



#define BITWISE_FN(OP, op)                      \
    template<typename T>                        \
    struct BinOp<T, T, OP>                      \
    {                                           \
        void eval(TNJ::array<T> &out,           \
                  const TNJ::array<T> &lhs,     \
                  const TNJ::array<T> &rhs,     \
                  int lim)                      \
        {                                       \
            for (int i = 0; i < lim; i++) {     \
                out[i] = lhs[i] op rhs[i];      \
            }                                   \
        }                                       \
    };                                          \

    BITWISE_FN(af_bitor_t, |)
    BITWISE_FN(af_bitand_t, &)
    BITWISE_FN(af_bitxor_t, ^)
    BITWISE_FN(af_bitshiftl_t, <<)
    BITWISE_FN(af_bitshiftr_t, >>)

#undef BITWISE_FN

    template<typename T, af_op_t op>
    Array<T> bitOp(const Array<T> &lhs, const Array<T> &rhs, const af::dim4 &odims)
    {
        TNJ::Node_ptr lhs_node = lhs.getNode();
        TNJ::Node_ptr rhs_node = rhs.getNode();

        TNJ::BinaryNode<T, T, op> *node = new TNJ::BinaryNode<T, T, op>(lhs_node, rhs_node);

        return createNodeArray<T>(odims, TNJ::Node_ptr(
                                      reinterpret_cast<TNJ::Node *>(node)));
    }
}

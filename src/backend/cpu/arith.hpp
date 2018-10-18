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
#include <cmath>
#include <JIT/BinaryNode.hpp>

namespace cpu
{

#define ARITH_FN(OP, op)                        \
    template<typename T>                        \
    struct BinOp<T, T, OP>                      \
    {                                           \
        void eval(JIT::array<T> &out,           \
                  const JIT::array<T> &lhs,     \
                  const JIT::array<T> &rhs,     \
                  int lim) const                \
        {                                       \
            for (int i = 0; i < lim; i++) {     \
                out[i] = lhs[i] op rhs[i];      \
            }                                   \
        }                                       \
    };                                          \


ARITH_FN(af_add_t, +)
ARITH_FN(af_sub_t, -)
ARITH_FN(af_mul_t, *)
ARITH_FN(af_div_t, /)

#undef ARITH_FN

template<typename T> static T __mod(T lhs, T rhs)
{
    T res = lhs % rhs;
    return (res < 0) ? abs(rhs - res) : res;
}

template<typename T> static T __rem(T lhs, T rhs) { return lhs % rhs; }

template<> STATIC_ float __mod<float>(float lhs, float rhs) { return fmod(lhs, rhs); }
template<> STATIC_ double __mod<double>(double lhs, double rhs) { return fmod(lhs, rhs); }
template<> STATIC_ float __rem<float>(float lhs, float rhs) { return remainder(lhs, rhs); }
template<> STATIC_ double __rem<double>(double lhs, double rhs) { return remainder(lhs, rhs); }


#define NUMERIC_FN(OP, FN)                      \
    template<typename T>                        \
    struct BinOp<T, T, OP>                      \
    {                                           \
        void eval(JIT::array<T> &out,           \
                  const JIT::array<T> &lhs,     \
                  const JIT::array<T> &rhs,     \
                  int lim)                      \
        {                                       \
            for (int i = 0; i < lim; i++) {     \
                out[i] = FN(lhs[i] , rhs[i]);   \
            }                                   \
        }                                       \
    };                                          \

NUMERIC_FN(af_max_t, max)
NUMERIC_FN(af_min_t, min)
NUMERIC_FN(af_mod_t, __mod)
NUMERIC_FN(af_pow_t, pow)
NUMERIC_FN(af_rem_t, __rem)
NUMERIC_FN(af_atan2_t, atan2)
NUMERIC_FN(af_hypot_t, hypot)

template<typename T, af_op_t op>
Array<T> arithOp(const Array<T> &lhs, const Array<T> &rhs, const af::dim4 &odims)
{
    JIT::Node_ptr lhs_node = lhs.getNode();
    JIT::Node_ptr rhs_node = rhs.getNode();

    JIT::BinaryNode<T, T, op> *node = new JIT::BinaryNode<T, T, op>(lhs_node, rhs_node);

    return createNodeArray<T>(odims,
                              JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
}

}

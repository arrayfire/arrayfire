/*******************************************************
 * Copyright (c) 2021, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <jit/Node.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <types.hpp>
#include <cmath>

namespace arrayfire {
namespace cpu {

template<typename To, typename Ti, af_op_t op>
struct BinOp;

#define ARITH_FN(OP, op)                                                 \
    template<typename T>                                                 \
    struct BinOp<T, T, OP> {                                             \
        void eval(jit::array<compute_t<T>> &out,                         \
                  const jit::array<compute_t<T>> &lhs,                   \
                  const jit::array<compute_t<T>> &rhs, int lim) const {  \
            for (int i = 0; i < lim; i++) { out[i] = lhs[i] op rhs[i]; } \
        }                                                                \
    };

ARITH_FN(af_add_t, +)
ARITH_FN(af_sub_t, -)
ARITH_FN(af_mul_t, *)
ARITH_FN(af_div_t, /)

#undef ARITH_FN

#define LOGIC_FN(OP, op)                                                      \
    template<typename T>                                                      \
    struct BinOp<char, T, OP> {                                               \
        void eval(jit::array<char> &out, const jit::array<compute_t<T>> &lhs, \
                  const jit::array<compute_t<T>> &rhs, int lim) {             \
            for (int i = 0; i < lim; i++) { out[i] = lhs[i] op rhs[i]; }      \
        }                                                                     \
    };

LOGIC_FN(af_eq_t, ==)
LOGIC_FN(af_neq_t, !=)
LOGIC_FN(af_lt_t, <)
LOGIC_FN(af_gt_t, >)
LOGIC_FN(af_le_t, <=)
LOGIC_FN(af_ge_t, >=)
LOGIC_FN(af_and_t, &&)
LOGIC_FN(af_or_t, ||)

#undef LOGIC_FN

#define LOGIC_CPLX_FN(T, OP, op)                                               \
    template<>                                                                 \
    struct BinOp<char, std::complex<T>, OP> {                                  \
        typedef std::complex<T> Ti;                                            \
        void eval(jit::array<char> &out, const jit::array<compute_t<Ti>> &lhs, \
                  const jit::array<compute_t<Ti>> &rhs, int lim) {             \
            for (int i = 0; i < lim; i++) {                                    \
                T lhs_mag = std::abs(lhs[i]);                                  \
                T rhs_mag = std::abs(rhs[i]);                                  \
                out[i]    = lhs_mag op rhs_mag;                                \
            }                                                                  \
        }                                                                      \
    };

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

template<typename T>
static T __mod(T lhs, T rhs) {
    T res = lhs % rhs;
    return (res < 0) ? abs(rhs - res) : res;
}

template<typename T>
static T __rem(T lhs, T rhs) {
    return lhs % rhs;
}

template<>
inline float __mod<float>(float lhs, float rhs) {
    return fmod(lhs, rhs);
}
template<>
inline double __mod<double>(double lhs, double rhs) {
    return fmod(lhs, rhs);
}
template<>
inline float __rem<float>(float lhs, float rhs) {
    return remainder(lhs, rhs);
}
template<>
inline double __rem<double>(double lhs, double rhs) {
    return remainder(lhs, rhs);
}

#define BITWISE_FN(OP, op)                                               \
    template<typename T>                                                 \
    struct BinOp<T, T, OP> {                                             \
        void eval(jit::array<compute_t<T>> &out,                         \
                  const jit::array<compute_t<T>> &lhs,                   \
                  const jit::array<compute_t<T>> &rhs, int lim) {        \
            for (int i = 0; i < lim; i++) { out[i] = lhs[i] op rhs[i]; } \
        }                                                                \
    };

BITWISE_FN(af_bitor_t, |)
BITWISE_FN(af_bitand_t, &)
BITWISE_FN(af_bitxor_t, ^)
BITWISE_FN(af_bitshiftl_t, <<)
BITWISE_FN(af_bitshiftr_t, >>)

#undef BITWISE_FN

#define NUMERIC_FN(OP, FN)                                                 \
    template<typename T>                                                   \
    struct BinOp<T, T, OP> {                                               \
        void eval(jit::array<compute_t<T>> &out,                           \
                  const jit::array<compute_t<T>> &lhs,                     \
                  const jit::array<compute_t<T>> &rhs, int lim) {          \
            for (int i = 0; i < lim; i++) { out[i] = FN(lhs[i], rhs[i]); } \
        }                                                                  \
    };

NUMERIC_FN(af_max_t, max)
NUMERIC_FN(af_min_t, min)
NUMERIC_FN(af_mod_t, __mod)
NUMERIC_FN(af_pow_t, pow)
NUMERIC_FN(af_rem_t, __rem)
NUMERIC_FN(af_atan2_t, atan2)
NUMERIC_FN(af_hypot_t, hypot)

}  // namespace cpu
}  // namespace arrayfire

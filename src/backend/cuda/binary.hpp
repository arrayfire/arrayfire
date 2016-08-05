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
    std::string name;
    int call_type;
    BinOp() :
        name("noop"),
        call_type(0)
    {}
};

#define BINARY(fn)                                      \
    template<typename To, typename Ti>                  \
    struct BinOp<To, Ti, af_##fn##_t>                   \
    {                                                   \
        std::string name;                               \
        int call_type;                                  \
        BinOp() :                                       \
            name(cuMangledName<Ti, true>("___"#fn)),    \
            call_type(0)                                \
            {}                                          \
    };

#if defined(USE_LIBDEVICE)
#define NVVM_ARITH_OP(T, fn, fname)             \
    template<>                                  \
    struct BinOp<T, T, af_##fn##_t>             \
    {                                           \
        std::string name;                       \
        int call_type;                          \
        BinOp() :                               \
            name(fname),                        \
            call_type(1)                        \
            {}                                  \
    };                                          \

#define NVVM_COMPARE_OP(T, fn, fname)           \
    template<>                                  \
    struct BinOp<char, T, af_##fn##_t>          \
    {                                           \
        std::string name;                       \
        int call_type;                          \
        BinOp() :                               \
            name(fname),                        \
            call_type(2)                        \
            {}                                  \
    };                                          \

#define NVVM_BINARY_FUNC(T, fn, fname)          \
    template<>                                  \
    struct BinOp<T, T, af_##fn##_t>             \
    {                                           \
        std::string name;                       \
        int call_type;                          \
        BinOp() :                               \
            name("@__nv_"#fname),               \
            call_type(0)                        \
            {}                                  \
    };                                          \

#else

#define NVVM_ARITH_OP(T, fn, fname)    // No specialization
#define NVVM_COMPARE_OP(T, fn, fname)  // No specialization
#define NVVM_BINARY_FUNC(T, fn, fname) // No specialization

#endif

#define NVVM_ARITH_OP_INT(fn, fname)         \
    NVVM_ARITH_OP(int, fn, fname)            \
    NVVM_ARITH_OP(short, fn, fname)          \
    NVVM_ARITH_OP(intl, fn, fname)           \

#define NVVM_ARITH_OP_UINT(fn, fname)        \
    NVVM_ARITH_OP(uint, fn, fname)           \
    NVVM_ARITH_OP(ushort, fn, fname)         \
    NVVM_ARITH_OP(uintl, fn, fname)          \

#define NVVM_ARITH_OP_FLOAT(fn, fname)       \
    NVVM_ARITH_OP(float, fn, fname)          \
    NVVM_ARITH_OP(double, fn, fname)         \

#define NVVM_ARITH_OP_CPLX(fn, fname)        \
    NVVM_ARITH_OP(cfloat, fn, fname)         \
    NVVM_ARITH_OP(cdouble, fn, fname)        \

#define NVVM_COMPARE_OP_INT(fn, fname)       \
    NVVM_COMPARE_OP(int, fn, fname)          \
    NVVM_COMPARE_OP(short, fn, fname)        \
    NVVM_COMPARE_OP(intl, fn, fname)         \

#define NVVM_COMPARE_OP_UINT(fn, fname)      \
    NVVM_COMPARE_OP(uint, fn, fname)         \
    NVVM_COMPARE_OP(ushort, fn, fname)       \
    NVVM_COMPARE_OP(uintl, fn, fname)        \

#define NVVM_COMPARE_OP_FLOAT(fn, fname)     \
    NVVM_COMPARE_OP(float, fn, fname)        \
    NVVM_COMPARE_OP(double, fn, fname)       \

BINARY(add)
NVVM_ARITH_OP_INT(add, "add")
NVVM_ARITH_OP_UINT(add, "add")
NVVM_ARITH_OP_FLOAT(add, "fadd")
NVVM_ARITH_OP_CPLX(add, "fadd")

BINARY(sub)
NVVM_ARITH_OP_INT(sub, "sub")
NVVM_ARITH_OP_UINT(sub, "sub")
NVVM_ARITH_OP_FLOAT(sub, "fsub")
NVVM_ARITH_OP_CPLX(sub, "fsub")

BINARY(mul)
NVVM_ARITH_OP_INT(mul, "mul")
NVVM_ARITH_OP_UINT(mul, "mul")
NVVM_ARITH_OP_FLOAT(mul, "fmul")

BINARY(div)
NVVM_ARITH_OP_INT(div, "sdiv")
NVVM_ARITH_OP_UINT(div, "udiv")
NVVM_ARITH_OP_FLOAT(div, "fdiv")

BINARY(bitand)
NVVM_ARITH_OP_INT(bitand, "and")
NVVM_ARITH_OP_UINT(bitand, "and")

BINARY(bitor)
NVVM_ARITH_OP_INT(bitor, "or")
NVVM_ARITH_OP_UINT(bitor, "or")

BINARY(bitxor)
NVVM_ARITH_OP_INT(bitxor, "xor")
NVVM_ARITH_OP_UINT(bitxor, "xor")

BINARY(bitshiftl)
NVVM_ARITH_OP_INT(bitshiftl, "shl")
NVVM_ARITH_OP_UINT(bitshiftl, "shl")

BINARY(bitshiftr)
NVVM_ARITH_OP_INT(bitshiftr, "lshr")
NVVM_ARITH_OP_UINT(bitshiftr, "lshr")


BINARY(and)
BINARY(or)

BINARY(lt)
NVVM_COMPARE_OP_INT(lt, "icmp ult")
NVVM_COMPARE_OP_UINT(lt, "icmp slt")
NVVM_COMPARE_OP_FLOAT(lt, "fcmp olt")

BINARY(gt)
NVVM_COMPARE_OP_INT(gt, "icmp ugt")
NVVM_COMPARE_OP_UINT(gt, "icmp sgt")
NVVM_COMPARE_OP_FLOAT(gt, "fcmp ogt")

BINARY(le)
NVVM_COMPARE_OP_INT(le, "icmp ule")
NVVM_COMPARE_OP_UINT(le, "icmp sle")
NVVM_COMPARE_OP_FLOAT(le, "fcmp ole")

BINARY(ge)
NVVM_COMPARE_OP_INT(ge, "icmp uge")
NVVM_COMPARE_OP_UINT(ge, "icmp sge")
NVVM_COMPARE_OP_FLOAT(ge, "fcmp oge")

BINARY(eq)
NVVM_COMPARE_OP_INT(eq, "icmp ueq")
NVVM_COMPARE_OP_UINT(eq, "icmp seq")
NVVM_COMPARE_OP_FLOAT(eq, "fcmp oeq")

BINARY(neq)
NVVM_COMPARE_OP_INT(neq, "icmp une")
NVVM_COMPARE_OP_UINT(neq, "icmp sne")
NVVM_COMPARE_OP_FLOAT(neq, "fcmp one")

BINARY(max)
NVVM_BINARY_FUNC(float, max, fmaxf)
NVVM_BINARY_FUNC(double, max, fmax)
NVVM_BINARY_FUNC(int, max, max)
NVVM_BINARY_FUNC(uint, max, umax)
NVVM_BINARY_FUNC(intl, max, llmax)
NVVM_BINARY_FUNC(uintl, max, ullmax)

BINARY(min)
NVVM_BINARY_FUNC(float, min, fminf)
NVVM_BINARY_FUNC(double, min, fmin)
NVVM_BINARY_FUNC(int, min, min)
NVVM_BINARY_FUNC(uint, min, umin)
NVVM_BINARY_FUNC(intl, min, llmin)
NVVM_BINARY_FUNC(uintl, min, ullmin)

BINARY(pow)
NVVM_BINARY_FUNC(float, pow, powf)
NVVM_BINARY_FUNC(double, pow, pow)

BINARY(mod)
NVVM_BINARY_FUNC(float, mod, fmodf)
NVVM_BINARY_FUNC(double, mod, fmod)

BINARY(rem)
NVVM_BINARY_FUNC(float, rem, remainderf)
NVVM_BINARY_FUNC(double, rem, remainder)

BINARY(atan2)
NVVM_BINARY_FUNC(float, atan2, atan2f)
NVVM_BINARY_FUNC(double, atan2, atan2)

BINARY(hypot)
NVVM_BINARY_FUNC(float, hypot, hypotf)
NVVM_BINARY_FUNC(double, hypot, hypot)

#undef BINARY

template<typename To, typename Ti, af_op_t op>
Array<To> createBinaryNode(const Array<Ti> &lhs, const Array<Ti> &rhs, const af::dim4 &odims)
{
    BinOp<To, Ti, op> bop;

    JIT::Node_ptr lhs_node = lhs.getNode();
    JIT::Node_ptr rhs_node = rhs.getNode();

    JIT::BinaryNode *node = new JIT::BinaryNode(irname<To>(),
                                                afShortName<To>(),
                                                bop.name,
                                                lhs_node,
                                                rhs_node,
                                                (int)(op),
                                                bop.call_type);

    return createNodeArray<To>(odims, JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
}

}

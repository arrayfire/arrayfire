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

#define SPECIALIZE_ARITH(T, fn, fname)          \
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

#define SPECIALIZE_ARITH_INT(fn, fname)         \
    SPECIALIZE_ARITH(int, fn, fname)            \
    SPECIALIZE_ARITH(short, fn, fname)          \
    SPECIALIZE_ARITH(intl, fn, fname)           \

#define SPECIALIZE_ARITH_UINT(fn, fname)        \
    SPECIALIZE_ARITH(uint, fn, fname)           \
    SPECIALIZE_ARITH(ushort, fn, fname)         \
    SPECIALIZE_ARITH(uintl, fn, fname)          \

#define SPECIALIZE_ARITH_FLOAT(fn, fname)       \
    SPECIALIZE_ARITH(float, fn, fname)          \
    SPECIALIZE_ARITH(double, fn, fname)         \

#define SPECIALIZE_ARITH_CPLX(fn, fname)        \
    SPECIALIZE_ARITH(cfloat, fn, fname)         \
    SPECIALIZE_ARITH(cdouble, fn, fname)        \


BINARY(add)
SPECIALIZE_ARITH_INT(add, "add")
SPECIALIZE_ARITH_UINT(add, "add")
SPECIALIZE_ARITH_FLOAT(add, "fadd")
SPECIALIZE_ARITH_CPLX(add, "fadd")

BINARY(sub)
SPECIALIZE_ARITH_INT(sub, "sub")
SPECIALIZE_ARITH_UINT(sub, "sub")
SPECIALIZE_ARITH_FLOAT(sub, "fsub")
SPECIALIZE_ARITH_CPLX(sub, "fsub")

BINARY(mul)
SPECIALIZE_ARITH_INT(mul, "mul")
SPECIALIZE_ARITH_UINT(mul, "mul")
SPECIALIZE_ARITH_FLOAT(mul, "fmul")

BINARY(div)
SPECIALIZE_ARITH_INT(div, "sdiv")
SPECIALIZE_ARITH_UINT(div, "udiv")
SPECIALIZE_ARITH_FLOAT(div, "fdiv")

BINARY(bitand)
SPECIALIZE_ARITH_INT(bitand, "and")
SPECIALIZE_ARITH_UINT(bitand, "and")

BINARY(bitor)
SPECIALIZE_ARITH_INT(bitor, "or")
SPECIALIZE_ARITH_UINT(bitor, "or")

BINARY(bitxor)
SPECIALIZE_ARITH_INT(bitxor, "xor")
SPECIALIZE_ARITH_UINT(bitxor, "xor")

BINARY(bitshiftl)
SPECIALIZE_ARITH_INT(bitshiftl, "shl")
SPECIALIZE_ARITH_UINT(bitshiftl, "shl")

BINARY(bitshiftr)
SPECIALIZE_ARITH_INT(bitshiftr, "lshr")
SPECIALIZE_ARITH_UINT(bitshiftr, "lshr")


BINARY(and)
BINARY(or)


#define SPECIALIZE_COMPARE(T, fn, fname)        \
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

#define SPECIALIZE_COMPARE_INT(fn, fname)       \
    SPECIALIZE_COMPARE(int, fn, fname)          \
    SPECIALIZE_COMPARE(short, fn, fname)        \
    SPECIALIZE_COMPARE(intl, fn, fname)         \

#define SPECIALIZE_COMPARE_UINT(fn, fname)      \
    SPECIALIZE_COMPARE(uint, fn, fname)         \
    SPECIALIZE_COMPARE(ushort, fn, fname)       \
    SPECIALIZE_COMPARE(uintl, fn, fname)        \

#define SPECIALIZE_COMPARE_FLOAT(fn, fname)     \
    SPECIALIZE_COMPARE(float, fn, fname)        \
    SPECIALIZE_COMPARE(double, fn, fname)       \


BINARY(lt)
SPECIALIZE_COMPARE_INT(lt, "icmp ult")
SPECIALIZE_COMPARE_UINT(lt, "icmp slt")
SPECIALIZE_COMPARE_FLOAT(lt, "fcmp olt")

BINARY(gt)
SPECIALIZE_COMPARE_INT(gt, "icmp ugt")
SPECIALIZE_COMPARE_UINT(gt, "icmp sgt")
SPECIALIZE_COMPARE_FLOAT(gt, "fcmp ogt")

BINARY(le)
SPECIALIZE_COMPARE_INT(le, "icmp ule")
SPECIALIZE_COMPARE_UINT(le, "icmp sle")
SPECIALIZE_COMPARE_FLOAT(le, "fcmp ole")

BINARY(ge)
SPECIALIZE_COMPARE_INT(ge, "icmp uge")
SPECIALIZE_COMPARE_UINT(ge, "icmp sge")
SPECIALIZE_COMPARE_FLOAT(ge, "fcmp oge")

BINARY(eq)
SPECIALIZE_COMPARE_INT(eq, "icmp ueq")
SPECIALIZE_COMPARE_UINT(eq, "icmp seq")
SPECIALIZE_COMPARE_FLOAT(eq, "fcmp oeq")

BINARY(neq)
SPECIALIZE_COMPARE_INT(neq, "icmp une")
SPECIALIZE_COMPARE_UINT(neq, "icmp sne")
SPECIALIZE_COMPARE_FLOAT(neq, "fcmp one")

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
                                                bop.name,
                                                lhs_node,
                                                rhs_node,
                                                (int)(op),
                                                bop.call_type);

    return createNodeArray<To>(odims, JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
}

}

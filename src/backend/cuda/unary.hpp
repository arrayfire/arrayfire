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

#define UNARY_FN(fn)                                \
    template<typename T>                            \
    struct UnOp<T, af_##fn##_t>                     \
    {                                               \
        std::string res;                            \
        bool is_check;                              \
        UnOp() :                                    \
            res(cuMangledName<T, false>("___"#fn)), \
            is_check(false)                         \
        {                                           \
        }                                           \
        const std::string name()                    \
        {                                           \
            return res;                             \
        }                                           \
    };                                              \

#define UNARY_FN_NAME(op, fn)                       \
    template<typename T>                            \
    struct UnOp<T, af_##op##_t>                     \
    {                                               \
        std::string res;                            \
        bool is_check;                              \
        UnOp() :                                    \
            res(cuMangledName<T, false>("___"#fn)), \
            is_check(false)                         \
        {                                           \
        }                                           \
        const std::string name()                    \
        {                                           \
            return res;                             \
        }                                           \
    };                                              \

#if defined(USE_LIBDEVICE)
#define NVVM_SPECIALIZE_TYPE(T, fn, fname)      \
    template<>                                  \
    struct UnOp<T, af_##fn##_t>                 \
    {                                           \
        std::string res;                        \
        bool is_check;                          \
        UnOp() :                                \
            res("@__nv_"#fname),                \
            is_check(false)                     \
        {                                       \
        }                                       \
        const std::string name()                \
        {                                       \
            return res;                         \
        }                                       \
    };                                          \

#define NVVM_SPECIALIZE_CHECK(T, fn, fname)     \
    template<>                                  \
    struct UnOp<T, af_##fn##_t>                 \
    {                                           \
        std::string res;                        \
        bool is_check;                          \
        UnOp() :                                \
            res("@__nv_"#fname),                \
            is_check(true)                      \
        {                                       \
        }                                       \
        const std::string name()                \
        {                                       \
            return res;                         \
        }                                       \
    };                                          \

#else
#define #define NVVM_SPECIALIZE_TYPE(T, fn, fname)  // no specialization
#define #define NVVM_SPECIALIZE_CHECK(T, fn, fname)  // no specialization
#endif

#define NVVM_SPECIALIZE_FLOATING_NAME(fn, fname)    \
    UNARY_FN(fn)                                    \
    NVVM_SPECIALIZE_TYPE(float, fn, fname##f)       \
    NVVM_SPECIALIZE_TYPE(double, fn, fname)         \


#define NVVM_SPECIALIZE_FLOATING(fn)            \
    NVVM_SPECIALIZE_FLOATING_NAME(fn, fn)

NVVM_SPECIALIZE_FLOATING(sin)
NVVM_SPECIALIZE_FLOATING(cos)
NVVM_SPECIALIZE_FLOATING(tan)
NVVM_SPECIALIZE_FLOATING(asin)
NVVM_SPECIALIZE_FLOATING(acos)
NVVM_SPECIALIZE_FLOATING(atan)
NVVM_SPECIALIZE_FLOATING(sinh)
NVVM_SPECIALIZE_FLOATING(cosh)
NVVM_SPECIALIZE_FLOATING(tanh)
NVVM_SPECIALIZE_FLOATING(asinh)
NVVM_SPECIALIZE_FLOATING(acosh)
NVVM_SPECIALIZE_FLOATING(atanh)
NVVM_SPECIALIZE_FLOATING(exp)
NVVM_SPECIALIZE_FLOATING(expm1)
NVVM_SPECIALIZE_FLOATING(erf)
NVVM_SPECIALIZE_FLOATING(erfc)
NVVM_SPECIALIZE_FLOATING(tgamma)
NVVM_SPECIALIZE_FLOATING(lgamma)
NVVM_SPECIALIZE_FLOATING(log)
NVVM_SPECIALIZE_FLOATING(log1p)
NVVM_SPECIALIZE_FLOATING(log10)
NVVM_SPECIALIZE_FLOATING(log2)
NVVM_SPECIALIZE_FLOATING(sqrt)
NVVM_SPECIALIZE_FLOATING(cbrt)
NVVM_SPECIALIZE_FLOATING(round)
NVVM_SPECIALIZE_FLOATING(trunc)
NVVM_SPECIALIZE_FLOATING(ceil)
NVVM_SPECIALIZE_FLOATING(floor)

UNARY_FN(sign )
NVVM_SPECIALIZE_CHECK(float , sign, signbitf)
NVVM_SPECIALIZE_CHECK(double, sign, signbitd)

UNARY_FN_NAME(isnan, isNaN)
NVVM_SPECIALIZE_CHECK(float , isnan, isnanf)
NVVM_SPECIALIZE_CHECK(double, isnan, isnand)

UNARY_FN_NAME(isinf, isINF)
NVVM_SPECIALIZE_CHECK(float , isinf, isinff)
NVVM_SPECIALIZE_CHECK(double, isinf, isinfd)

UNARY_FN_NAME(iszero, iszero)
UNARY_FN(sigmoid)

#undef UNARY_FN

    template<typename T, af_op_t op>
    Array<T> unaryOp(const Array<T> &in)
    {

        UnOp<T, op> uop;

        JIT::Node_ptr in_node = in.getNode();

        JIT::UnaryNode *node = new JIT::UnaryNode(irname<T>(),
                                                  afShortName<T>(),
                                                  uop.name(),
                                                  in_node, op, uop.is_check);

        return createNodeArray<T>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename T, af_op_t op>
    Array<char> checkOp(const Array<T> &in)
    {
        UnOp<T, op> uop;

        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(irname<char>(),
                                                  afShortName<char>(),
                                                  uop.name(),
                                                  in_node, op, uop.is_check);
        return createNodeArray<char>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }
}

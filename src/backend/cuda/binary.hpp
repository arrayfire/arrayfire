/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <common/jit/BinaryNode.hpp>
#include <common/jit/NaryNode.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <af/dim4.hpp>

namespace cuda {

template<typename To, typename Ti, af_op_t op>
struct BinOp {
    const char *name() { return "__invalid"; }
};

#define BINARY_TYPE_1(fn)                            \
    template<typename To, typename Ti>               \
    struct BinOp<To, Ti, af_##fn##_t> {              \
        const char *name() { return "__" #fn; }      \
    };                                               \
                                                     \
    template<typename To>                            \
    struct BinOp<To, cfloat, af_##fn##_t> {          \
        const char *name() { return "__c" #fn "f"; } \
    };                                               \
                                                     \
    template<typename To>                            \
    struct BinOp<To, cdouble, af_##fn##_t> {         \
        const char *name() { return "__c" #fn; }     \
    };

BINARY_TYPE_1(eq)
BINARY_TYPE_1(neq)
BINARY_TYPE_1(lt)
BINARY_TYPE_1(le)
BINARY_TYPE_1(gt)
BINARY_TYPE_1(ge)
BINARY_TYPE_1(add)
BINARY_TYPE_1(sub)
BINARY_TYPE_1(mul)
BINARY_TYPE_1(div)
BINARY_TYPE_1(and)
BINARY_TYPE_1(or)
BINARY_TYPE_1(bitand)
BINARY_TYPE_1(bitor)
BINARY_TYPE_1(bitxor)
BINARY_TYPE_1(bitshiftl)
BINARY_TYPE_1(bitshiftr)

#undef BINARY_TYPE_1

#define BINARY_TYPE_2(fn)                            \
    template<typename To, typename Ti>               \
    struct BinOp<To, Ti, af_##fn##_t> {              \
        const char *name() { return "__" #fn; }      \
    };                                               \
    template<typename To>                            \
    struct BinOp<To, float, af_##fn##_t> {           \
        const char *name() { return "f" #fn; }       \
    };                                               \
    template<typename To>                            \
    struct BinOp<To, double, af_##fn##_t> {          \
        const char *name() { return "f" #fn; }       \
    };                                               \
    template<typename To>                            \
    struct BinOp<To, cfloat, af_##fn##_t> {          \
        const char *name() { return "__c" #fn "f"; } \
    };                                               \
    template<typename To>                            \
    struct BinOp<To, cdouble, af_##fn##_t> {         \
        const char *name() { return "__c" #fn; }     \
    };

BINARY_TYPE_2(min)
BINARY_TYPE_2(max)
BINARY_TYPE_2(rem)
BINARY_TYPE_2(mod)

template<typename To, typename Ti>
struct BinOp<To, Ti, af_pow_t> {
    const char *name() { return "__pow"; }
};

#define POW_BINARY_OP(INTYPE, OPNAME)         \
    template<typename To>                     \
    struct BinOp<To, INTYPE, af_pow_t> {      \
        const char *name() { return OPNAME; } \
    };

POW_BINARY_OP(double, "pow")
POW_BINARY_OP(float, "powf")
POW_BINARY_OP(intl, "__powll")
POW_BINARY_OP(uintl, "__powul")
POW_BINARY_OP(uint, "__powui")
POW_BINARY_OP(int, "__powsi")

#undef POW_BINARY_OP

template<typename Ti>
struct BinOp<cfloat, Ti, af_cplx2_t> {
    const char *name() { return "__cplx2f"; }
};

template<typename Ti>
struct BinOp<cdouble, Ti, af_cplx2_t> {
    const char *name() { return "__cplx2"; }
};

template<typename To, typename Ti>
struct BinOp<To, Ti, af_cplx2_t> {
    const char *name() { return "noop"; }
};

template<typename To, typename Ti>
struct BinOp<To, Ti, af_atan2_t> {
    const char *name() { return "atan2"; }
};

template<typename To, typename Ti>
struct BinOp<To, Ti, af_hypot_t> {
    const char *name() { return "hypot"; }
};

template<typename To, typename Ti, af_op_t op>
Array<To> createBinaryNode(const Array<Ti> &lhs, const Array<Ti> &rhs,
                           const af::dim4 &odims) {
    using common::Node;
    using common::Node_ptr;

    auto createBinary = [](std::array<Node_ptr, 2> &operands) -> Node_ptr {
        BinOp<To, Ti, op> bop;
        return Node_ptr(new common::BinaryNode(
            getFullName<To>(), shortname<To>(true), bop.name(), operands[0],
            operands[1], (int)(op)));
    };

    Node_ptr out =
        common::createNaryNode<Ti, 2>(odims, createBinary, {&lhs, &rhs});
    return createNodeArray<To>(odims, out);
}

}  // namespace cuda

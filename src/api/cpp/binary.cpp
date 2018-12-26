/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/arith.h>
#include <af/array.h>
#include <af/data.h>
#include <af/gfor.h>
#include "error.hpp"

namespace af {

#define INSTANTIATE(cppfunc, cfunc)                             \
    array cppfunc(const array &lhs, const array &rhs) {         \
        af_array out = 0;                                       \
        AF_THROW(cfunc(&out, lhs.get(), rhs.get(), gforGet())); \
        return array(out);                                      \
    }

INSTANTIATE(min, af_minof)
INSTANTIATE(max, af_maxof)
INSTANTIATE(pow, af_pow)
INSTANTIATE(root, af_root)
INSTANTIATE(rem, af_rem)
INSTANTIATE(mod, af_mod)

INSTANTIATE(complex, af_cplx2)
INSTANTIATE(atan2, af_atan2)
INSTANTIATE(hypot, af_hypot)

#define WRAPPER(func)                                             \
    array func(const array &lhs, const double rhs) {              \
        af::dtype ty = lhs.type();                                \
        if (lhs.iscomplex()) { ty = lhs.issingle() ? f32 : f64; } \
        return func(lhs, constant(rhs, lhs.dims(), ty));          \
    }                                                             \
    array func(const double lhs, const array &rhs) {              \
        af::dtype ty = rhs.type();                                \
        if (rhs.iscomplex()) { ty = rhs.issingle() ? f32 : f64; } \
        return func(constant(lhs, rhs.dims(), ty), rhs);          \
    }

WRAPPER(min)
WRAPPER(max)
WRAPPER(pow)
WRAPPER(root)
WRAPPER(rem)
WRAPPER(mod)
WRAPPER(complex)
WRAPPER(atan2)
WRAPPER(hypot)
}  // namespace af

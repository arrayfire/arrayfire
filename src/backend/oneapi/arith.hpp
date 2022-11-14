/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <common/jit/BinaryNode.hpp>
#include <err_oneapi.hpp>
#include <optypes.hpp>
#include <af/dim4.hpp>

namespace oneapi {

template<typename T, af_op_t op>
Array<T> arithOp(const Array<T> &&lhs, const Array<T> &&rhs,
                 const af::dim4 &odims) {
    ONEAPI_NOT_SUPPORTED(__FUNCTION__);
    return common::createBinaryNode<T, T, op>(lhs, rhs, odims);
}

template<typename T, af_op_t op>
Array<T> arithOp(const Array<T> &lhs, const Array<T> &rhs,
                 const af::dim4 &odims) {
    ONEAPI_NOT_SUPPORTED(__FUNCTION__);
    return common::createBinaryNode<T, T, op>(lhs, rhs, odims);
}
}  // namespace oneapi

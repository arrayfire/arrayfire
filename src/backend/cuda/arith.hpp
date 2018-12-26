/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <binary.hpp>
#include <optypes.hpp>
#include <af/dim4.hpp>

namespace cuda {
template<typename T, af_op_t op>
Array<T> arithOp(const Array<T> &lhs, const Array<T> &rhs,
                 const af::dim4 &odims) {
    return createBinaryNode<T, T, op>(lhs, rhs, odims);
}
}  // namespace cuda

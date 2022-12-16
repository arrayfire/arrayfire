/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <tile.hpp>

#include <Array.hpp>
#include <arith.hpp>
#include <backend.hpp>
#include <optypes.hpp>
#include <unary.hpp>

#include <af/dim4.hpp>

namespace arrayfire {
namespace common {

/// duplicates the elements of an Array<T> array.
template<typename T>
detail::Array<T> tile(const detail::Array<T> &in, const af::dim4 tileDims) {
    const af::dim4 &inDims = in.dims();

    // FIXME: Always use JIT instead of checking for the condition.
    // The current limitation exists for performance reasons. it should change
    // in the future.

    bool take_jit_path = true;
    af::dim4 outDims(1, 1, 1, 1);

    // Check if JIT path can be taken. JIT path can only be taken if tiling a
    // singleton dimension.
    for (int i = 0; i < 4; i++) {
        take_jit_path &= (inDims[i] == 1 || tileDims[i] == 1);
        outDims[i] = inDims[i] * tileDims[i];
    }

    if (take_jit_path) {
        return detail::unaryOp<T, af_noop_t>(in, outDims);
    } else {
        return detail::tile<T>(in, tileDims);
    }
}

}  // namespace common
}  // namespace arrayfire

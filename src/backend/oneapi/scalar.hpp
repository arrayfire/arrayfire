/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/jit/ScalarNode.hpp>
#include <math.hpp>
#include <optypes.hpp>

namespace arrayfire {
namespace oneapi {

template<typename T>
Array<T> createScalarNode(const dim4 &size, const T val) {
    return createNodeArray<T>(size,
                              std::make_shared<common::ScalarNode<T>>(val));
}

}  // namespace oneapi
}  // namespace arrayfire

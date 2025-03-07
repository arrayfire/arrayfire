/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/jit/ShiftNodeBase.hpp>
#include <jit/BufferNode.hpp>

namespace arrayfire {
namespace oneapi {
namespace jit {

template<typename T>
using ShiftNode = common::ShiftNodeBase<BufferNode<T>>;

}  // namespace jit
}  // namespace oneapi
}  // namespace arrayfire

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include "../Param.hpp"
#include <common/jit/BufferNodeBase.hpp>

namespace cuda
{
namespace jit
{
  template<typename T>
  using BufferNode = common::BufferNodeBase<std::shared_ptr<T>, Param<T>>;
}
}

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/jit/BufferNodeBase.hpp>
#include <common/jit/Node.hpp>
#include <af/defines.h>
#include <iomanip>
#include <mutex>
#include "../kernel/KParam.hpp"

namespace opencl {
namespace jit {
using BufferNode = common::BufferNodeBase<std::shared_ptr<cl::Buffer>, KParam>;
}
}  // namespace opencl

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
#include "../kernel/KParam.hpp"

#include <memory>

namespace arrayfire {
namespace opencl {
namespace jit {
using BufferNode = common::BufferNodeBase<std::shared_ptr<cl::Buffer>, KParam>;
}  // namespace jit
}  // namespace opencl

namespace common {

template<typename DataType, typename ParamType>
bool BufferNodeBase<DataType, ParamType>::operator==(
    const BufferNodeBase<DataType, ParamType> &other) const noexcept {
    // clang-format off
    return m_data.get() == other.m_data.get() &&
           m_bytes == other.m_bytes &&
           m_param.offset == other.m_param.offset;
    // clang-format on
}

}  // namespace common
}  // namespace arrayfire

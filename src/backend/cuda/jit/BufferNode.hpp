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
#include "../Param.hpp"

namespace arrayfire {
namespace cuda {
namespace jit {
template<typename T>
using BufferNode = common::BufferNodeBase<std::shared_ptr<T>, Param<T>>;
}  // namespace jit
}  // namespace cuda

namespace common {

template<typename DataType, typename ParamType>
bool BufferNodeBase<DataType, ParamType>::operator==(
    const BufferNodeBase<DataType, ParamType> &other) const noexcept {
    // clang-format off
    return m_data.get() == other.m_data.get() &&
           m_bytes == other.m_bytes &&
           m_param.ptr == other.m_param.ptr;
    // clang-format on
}

}  // namespace common

}  // namespace arrayfire

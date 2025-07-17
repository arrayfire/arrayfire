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
           m_param.ptr == other.m_param.ptr &&
           m_linear_buffer == other.m_linear_buffer &&
           m_param.dims[0] == other.m_param.dims[0] &&
           m_param.dims[1] == other.m_param.dims[1] &&
           m_param.dims[2] == other.m_param.dims[2] &&
           m_param.dims[3] == other.m_param.dims[3] &&
           m_param.strides[0] == other.m_param.strides[0] &&
           m_param.strides[1] == other.m_param.strides[1] &&
           m_param.strides[2] == other.m_param.strides[2] &&
           m_param.strides[3] == other.m_param.strides[3];
    // clang-format on
}

}  // namespace common

}  // namespace arrayfire

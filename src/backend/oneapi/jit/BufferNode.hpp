/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <common/jit/BufferNodeBase.hpp>
#include <jit/kernel_generators.hpp>

#include <memory>

namespace arrayfire {
namespace oneapi {
namespace jit {
template<typename T>
using BufferNode = common::BufferNodeBase<std::shared_ptr<sycl::buffer<T>>,
                                          AParam<T, sycl::access_mode::read>>;
}  // namespace jit
}  // namespace oneapi

namespace common {

template<typename DataType, typename ParamType>
bool BufferNodeBase<DataType, ParamType>::operator==(
    const BufferNodeBase<DataType, ParamType> &other) const noexcept {
    // clang-format off
    return m_data.get() == other.m_data.get() &&
           m_bytes == other.m_bytes &&
           m_param.offset == other.m_param.offset &&
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

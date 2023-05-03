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
           m_param.offset == other.m_param.offset;
    // clang-format on
}

}  // namespace common
}  // namespace arrayfire

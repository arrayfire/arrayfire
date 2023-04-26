/*******************************************************
 * Copyright (c) 2022 ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <sycl/sycl.hpp>

template<typename T>
using read_accessor = sycl::accessor<T, 1, sycl::access::mode::read>;

template<typename T>
using write_accessor = sycl::accessor<T, 1, sycl::access::mode::write>;

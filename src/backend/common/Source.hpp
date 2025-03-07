/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

namespace arrayfire {
namespace common {
struct Source {
    const char* ptr;           // Pointer to the kernel source
    const std::size_t length;  // Length of the kernel source
    const std::size_t hash;    // hash value for the source *ptr;
};
}  // namespace common
}  // namespace arrayfire

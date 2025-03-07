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
namespace oneapi {
namespace kernel {

namespace creduce {
// TODO: are different values more appropriate for reduce on oneapi?
static const uint THREADS_PER_BLOCK = 256;
static const uint THREADS_X         = 32;
static const uint THREADS_Y         = THREADS_PER_BLOCK / THREADS_X;
static const uint REPEAT            = 32;

}  // namespace creduce

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire

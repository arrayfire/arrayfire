/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <mutex>

namespace common
{
typedef std::recursive_mutex mutex_t;
typedef std::lock_guard<mutex_t> lock_guard_t;
}

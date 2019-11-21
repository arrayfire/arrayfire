/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

namespace common {

// Returns the reference to the Events Enabled flag which determine if the
// events should be used by the memory manager when memory is allocated or
// freed.
//
// This is flag is necessary because events incure a small performance penilty
// which can be significant in the case where you are performing many smaller
// operations. Event based allocation and frees are required to support streams
// in ArrayFire which may be added in the future.
//
// \returns 0 if events are disabled(default). Nonzero otherwise
int& getEventsEnabledFlag();

}  // namespace common

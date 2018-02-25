/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
// Precompiled header file
#include <Param.hpp>
#include <TNJ/Node.hpp>

#include <af/array.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/MemoryManager.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <traits.hpp>
#include <types.hpp>
#include <types.hpp>
#include <ops.hpp>
#include <kernel/scan_by_key.hpp>

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

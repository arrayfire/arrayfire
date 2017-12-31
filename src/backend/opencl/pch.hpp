/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <CL/cl2.hpp>
#pragma GCC diagnostic pop



// Precompiled header file
#include <Param.hpp>
#include <JIT/Node.hpp>

#include <af/array.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/MemoryManager.hpp>
#include <GraphicsResourceManager.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <traits.hpp>
#include <types.hpp>
#include <types.hpp>
#include <ops.hpp>

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
#include <string>
#include <thread>
#include <vector>

/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <program.hpp>
#include <map>
#include <string>

namespace opencl
{
    using cl::Kernel;
    using cl::Program;

    typedef struct {
        Program* prog;
        Kernel* ker;
    } kc_entry_t;

    typedef std::map<string, kc_entry_t> kc_t;
    static kc_t kernelCaches[DeviceManager::MAX_DEVICES];
}

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <platform.hpp>
#include <af/defines.h>
//#include <kernel_headers/random.hpp>
#include <traits.hpp>
#include <sstream>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <program.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
    namespace kernel
    {
        template <typename T, af_random_type Type>
        void uniformDistribution(cl::Buffer out, size_t elements, const uintl seed, uintl &counter)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static std::map<int, Program*>  ranProgs;
                static std::map<int, Kernel*>   ranKernels;
                int device = getActiveDeviceId();

            } catch (cl::Error ex) {
                CL_TO_AF_ERROR(ex);
            }
        }
    }
}

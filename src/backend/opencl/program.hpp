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
#include <string>
#include <mutex>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

#define SHOW_DEBUG_BUILD_INFO(PROG) do {                                \
        cl_uint numDevices = PROG.getInfo<CL_PROGRAM_NUM_DEVICES>();    \
        for (unsigned int i = 0; i<numDevices; ++i) {                   \
            std::cout << PROG.getBuildInfo<CL_PROGRAM_BUILD_LOG>(       \
                PROG.getInfo<CL_PROGRAM_DEVICES>()[i]) << std::endl;    \
                                                                        \
            std::cout << PROG.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(   \
                PROG.getInfo<CL_PROGRAM_DEVICES>()[i]) << std::endl;    \
        }                                                               \
    } while(0)                                                          \


#if defined(NDEBUG)

#define SHOW_BUILD_INFO(PROG) do {                                  \
        const char *info = getenv("AF_OPENCL_SHOW_BUILD_INFO");     \
        if (info != nullptr && std::strncmp(info,"0", 1) != 0) {    \
            SHOW_DEBUG_BUILD_INFO(prog);                            \
        }                                                           \
    } while(0)

#else
#define SHOW_BUILD_INFO(PROG) SHOW_DEBUG_BUILD_INFO(PROG)
#endif

namespace opencl
{
    void buildProgram(cl::Program &prog,
                      const char *ker_str, const int ker_len, std::string options);

    void buildProgram(cl::Program &prog,
                      const int num_files,
                      const char **ker_str, const int *ker_len, std::string options);
}

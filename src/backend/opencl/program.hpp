/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cl2hpp.hpp>
#include <common/util.hpp>

#include <cstdio>
#include <string>
#include <vector>

#define SHOW_DEBUG_BUILD_INFO(PROG)                                       \
    do {                                                                  \
        cl_uint numDevices = PROG.getInfo<CL_PROGRAM_NUM_DEVICES>();      \
        for (unsigned int i = 0; i < numDevices; ++i) {                   \
            printf("%s\n", PROG.getBuildInfo<CL_PROGRAM_BUILD_LOG>(       \
                                   PROG.getInfo<CL_PROGRAM_DEVICES>()[i]) \
                               .c_str());                                 \
            printf("%s\n", PROG.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(   \
                                   PROG.getInfo<CL_PROGRAM_DEVICES>()[i]) \
                               .c_str());                                 \
        }                                                                 \
    } while (0)

#if defined(NDEBUG)

#define SHOW_BUILD_INFO(PROG)                                              \
    do {                                                                   \
        std::string info = getEnvVar("AF_OPENCL_SHOW_BUILD_INFO");         \
        if (!info.empty() && info != "0") { SHOW_DEBUG_BUILD_INFO(PROG); } \
    } while (0)

#else
#define SHOW_BUILD_INFO(PROG) SHOW_DEBUG_BUILD_INFO(PROG)
#endif

namespace opencl {

#if defined(AF_WITH_DEV_WARNINGS)
// TODO(pradeep) remove this version after porting to new cache interface
[[deprecated("use cl::Program buildProgram(vector<string>&, vector<string>&)")]]
#endif
void buildProgram(cl::Program &prog, const char *ker_str, const int ker_len,
                  const std::string &options);

#if defined(AF_WITH_DEV_WARNINGS)
// TODO(pradeep) remove this version after porting to new cache interface
[[deprecated("use cl::Program buildProgram(vector<string>&, vector<string>&)")]]
#endif
void buildProgram(cl::Program &prog, const int num_files, const char **ker_str,
                  const int *ker_len, const std::string &options);

cl::Program buildProgram(const std::vector<std::string> &kernelSources,
                         const std::vector<std::string> &options);

}  // namespace opencl

/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/compile_kernel.hpp>

#include <cl2hpp.hpp>
#include <common/Logger.hpp>
#include <common/defines.hpp>
#include <common/util.hpp>
#include <debug_opencl.hpp>
#include <err_opencl.hpp>
#include <kernel_headers/KParam.hpp>
#include <platform.hpp>
#include <traits.hpp>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using detail::Kernel;

using std::ostringstream;
using std::string;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

spdlog::logger *getLogger() {
    static std::shared_ptr<spdlog::logger> logger(common::loggerFactory("jit"));
    return logger.get();
}

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

const static std::string DEFAULT_MACROS_STR(
    "\n\
                                           #ifdef USE_DOUBLE\n\
                                           #pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\
                                           #endif\n                     \
                                           #ifdef USE_HALF\n\
                                           #pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\
                                           #else\n                     \
                                           #define half short\n          \
                                           #endif\n                      \
                                           #ifndef M_PI\n               \
                                           #define M_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164\n \
                                           #endif\n                     \
                                           ");

cl::Program buildProgram(const std::vector<std::string> &kernelSources,
                         const std::vector<std::string> &compileOpts) {
    using std::begin;
    using std::end;

    cl::Program retVal;
    try {
        static const std::string defaults =
            std::string(" -D dim_t=") +
            std::string(dtype_traits<dim_t>::getName());

        auto device = getDevice();

        const std::string cl_std =
            std::string(" -cl-std=CL") +
            device.getInfo<CL_DEVICE_OPENCL_C_VERSION>().substr(9, 3);

        cl::Program::Sources sources;
        sources.emplace_back(DEFAULT_MACROS_STR);
        sources.emplace_back(KParam_hpp, KParam_hpp_len);
        sources.insert(end(sources), begin(kernelSources), end(kernelSources));

        retVal = cl::Program(getContext(), sources);

        ostringstream options;
        for (auto &opt : compileOpts) { options << opt; }

        retVal.build({device}, (cl_std + defaults + options.str()).c_str());
    } catch (...) {
        SHOW_BUILD_INFO(retVal);
        throw;
    }
    return retVal;
}

}  // namespace opencl

namespace common {

Kernel compileKernel(const string &kernelName, const string &tInstance,
                     const vector<string> &sources,
                     const vector<string> &compileOpts, const bool isJIT) {
    using opencl::getActiveDeviceId;
    using opencl::getDevice;

    UNUSED(isJIT);
    UNUSED(tInstance);

    auto compileBegin = high_resolution_clock::now();
    auto prog         = detail::buildProgram(sources, compileOpts);
    auto prg          = new cl::Program(prog);
    auto krn =
        new cl::Kernel(*static_cast<cl::Program *>(prg), kernelName.c_str());
    auto compileEnd = high_resolution_clock::now();

    AF_TRACE("{{{:<30} : {{ compile:{:>5} ms, {{ {} }}, {} }}}}", kernelName,
             duration_cast<milliseconds>(compileEnd - compileBegin).count(),
             fmt::join(compileOpts, " "),
             getDevice(getActiveDeviceId()).getInfo<CL_DEVICE_NAME>());

    return {prg, krn};
}

Kernel loadKernel(const int device, const string &nameExpr) {
    OPENCL_NOT_SUPPORTED(
        "Disk caching OpenCL kernel binaries is not yet supported");
    return {nullptr, nullptr};
}

}  // namespace common

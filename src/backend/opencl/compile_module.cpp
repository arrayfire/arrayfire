/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/compile_module.hpp>  //compileModule & loadModuleFromDisk
#include <common/kernel_cache.hpp>    //getKernel(Module&, ...)

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

using cl::Error;
using cl::Program;
using common::loggerFactory;
using opencl::getActiveDeviceId;
using opencl::getDevice;
using opencl::Kernel;
using opencl::Module;
using spdlog::logger;

using std::begin;
using std::end;
using std::ostringstream;
using std::shared_ptr;
using std::string;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

logger *getLogger() {
    static shared_ptr<logger> logger(loggerFactory("jit"));
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
        string info = getEnvVar("AF_OPENCL_SHOW_BUILD_INFO");              \
        if (!info.empty() && info != "0") { SHOW_DEBUG_BUILD_INFO(PROG); } \
    } while (0)

#else
#define SHOW_BUILD_INFO(PROG) SHOW_DEBUG_BUILD_INFO(PROG)
#endif

namespace opencl {

const static string DEFAULT_MACROS_STR(
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

Program buildProgram(const vector<string> &kernelSources,
                     const vector<string> &compileOpts) {
    Program retVal;
    try {
        static const string defaults =
            string(" -D dim_t=") + string(dtype_traits<dim_t>::getName());

        auto device = getDevice();

        const string cl_std =
            string(" -cl-std=CL") +
            device.getInfo<CL_DEVICE_OPENCL_C_VERSION>().substr(9, 3);

        Program::Sources sources;
        sources.emplace_back(DEFAULT_MACROS_STR);
        sources.emplace_back(KParam_hpp, KParam_hpp_len);
        sources.insert(end(sources), begin(kernelSources), end(kernelSources));

        retVal = Program(getContext(), sources);

        ostringstream options;
        for (auto &opt : compileOpts) { options << opt; }

        retVal.build({device}, (cl_std + defaults + options.str()).c_str());
    } catch (Error &err) {
        if (err.err() == CL_BUILD_ERROR) { SHOW_BUILD_INFO(retVal); }
        throw;
    }
    return retVal;
}

}  // namespace opencl

namespace common {

Module compileModule(const string &moduleKey, const vector<string> &sources,
                     const vector<string> &options,
                     const vector<string> &kInstances, const bool isJIT) {
    UNUSED(kInstances);
    UNUSED(isJIT);

    auto compileBegin = high_resolution_clock::now();
    auto program      = opencl::buildProgram(sources, options);
    auto compileEnd   = high_resolution_clock::now();

    AF_TRACE("{{{:<30} : {{ compile:{:>5} ms, {{ {} }}, {} }}}}", moduleKey,
             duration_cast<milliseconds>(compileEnd - compileBegin).count(),
             fmt::join(options, " "),
             getDevice(getActiveDeviceId()).getInfo<CL_DEVICE_NAME>());

    return {program};
}

Module loadModuleFromDisk(const int device, const string &moduleKey,
                          const bool isJIT) {
    UNUSED(device);
    UNUSED(moduleKey);
    UNUSED(isJIT);
    return {};
}

Kernel getKernel(const Module &mod, const string &nameExpr,
                 const bool sourceWasJIT) {
    UNUSED(sourceWasJIT);
    return {&mod.get(), cl::Kernel(mod.get(), nameExpr.c_str())};
}

}  // namespace common

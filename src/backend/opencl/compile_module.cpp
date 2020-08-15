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

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using cl::Error;
using cl::Program;
using common::loggerFactory;
using fmt::format;
using opencl::getActiveDeviceId;
using opencl::getDevice;
using opencl::Kernel;
using opencl::Module;
using spdlog::logger;

using std::begin;
using std::end;
using std::ofstream;
using std::ostringstream;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::transform;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

logger *getLogger() {
    static shared_ptr<logger> logger(loggerFactory("jit"));
    return logger.get();
}

string getProgramBuildLog(const Program &prog) {
    string build_error("");
    try {
        build_error.reserve(4096);
        auto devices = prog.getInfo<CL_PROGRAM_DEVICES>();
        for (auto &device : prog.getInfo<CL_PROGRAM_DEVICES>()) {
            build_error +=
                format("OpenCL Device: {}\n\tOptions: {}\n\tLog:\n{}\n",
                       device.getInfo<CL_DEVICE_NAME>(),
                       prog.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device),
                       prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
        }
    } catch (const cl::Error &e) {
        build_error = format("Failed to fetch build log: {}", e.what());
    }
    return build_error;
}

#define THROW_BUILD_LOG_EXCEPTION(PROG)                              \
    do {                                                             \
        string build_error = getProgramBuildLog(PROG);               \
        string info        = getEnvVar("AF_OPENCL_SHOW_BUILD_INFO"); \
        if (!info.empty() && info != "0") puts(build_error.c_str()); \
        AF_ERROR(build_error, AF_ERR_INTERNAL);                      \
    } while (0)

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
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            THROW_BUILD_LOG_EXCEPTION(retVal);
        }
        throw;
    }
    return retVal;
}

}  // namespace opencl

string getKernelCacheFilename(const int device, const string &key) {
    auto &dev = opencl::getDevice(device);

    unsigned vendorId = dev.getInfo<CL_DEVICE_VENDOR_ID>();
    auto devName      = dev.getInfo<CL_DEVICE_NAME>();
    string infix      = to_string(vendorId) + "_" + devName;

    transform(infix.begin(), infix.end(), infix.begin(),
              [](unsigned char c) { return std::toupper(c); });
    std::replace(infix.begin(), infix.end(), ' ', '_');

    return "KER" + key + "_CL_" + infix + "_AF_" +
           to_string(AF_API_VERSION_CURRENT) + ".bin";
}

namespace common {

Module compileModule(const string &moduleKey, const vector<string> &sources,
                     const vector<string> &options,
                     const vector<string> &kInstances, const bool isJIT) {
    UNUSED(kInstances);
    UNUSED(isJIT);

    auto compileBegin = high_resolution_clock::now();
    auto program      = opencl::buildProgram(sources, options);
    auto compileEnd   = high_resolution_clock::now();

#ifdef AF_CACHE_KERNELS_TO_DISK
    const int device             = opencl::getActiveDeviceId();
    const string &cacheDirectory = getCacheDirectory();
    if (!cacheDirectory.empty()) {
        const string cacheFile = cacheDirectory + AF_PATH_SEPARATOR +
                                 getKernelCacheFilename(device, moduleKey);
        const string tempFile =
            cacheDirectory + AF_PATH_SEPARATOR + makeTempFilename();
        try {
            auto binaries = program.getInfo<CL_PROGRAM_BINARIES>();

            // TODO Handle cases where program objects are created from contexts
            // having multiple devices
            const size_t clbinSize = binaries[0].size();
            const char *clbin =
                reinterpret_cast<const char *>(binaries[0].data());
            const size_t clbinHash = deterministicHash(clbin, clbinSize);

            // write module hash and binary data to file
            ofstream out(tempFile, std::ios::binary);

            out.write(reinterpret_cast<const char *>(&clbinHash),
                      sizeof(clbinHash));
            out.write(reinterpret_cast<const char *>(&clbinSize),
                      sizeof(clbinSize));
            out.write(static_cast<const char *>(clbin), clbinSize);
            out.close();

            // try to rename temporary file into final cache file, if this fails
            // this means another thread has finished compiling this kernel
            // before the current thread.
            if (!renameFile(tempFile, cacheFile)) { removeFile(tempFile); }
        } catch (const cl::Error &e) {
            AF_TRACE("{{{:<20} : Failed to fetch opencl binary for {}, {}}}",
                     moduleKey,
                     opencl::getDevice(device).getInfo<CL_DEVICE_NAME>(),
                     e.what());
        } catch (const std::ios_base::failure &e) {
            AF_TRACE("{{{:<20} : Failed writing binary to {} for {}, {}}}",
                     moduleKey, cacheFile,
                     opencl::getDevice(device).getInfo<CL_DEVICE_NAME>(),
                     e.what());
        }
    }
#endif

    AF_TRACE("{{{:<20} : {{ compile:{:>5} ms, {{ {} }}, {} }}}}", moduleKey,
             duration_cast<milliseconds>(compileEnd - compileBegin).count(),
             fmt::join(options, " "),
             getDevice(getActiveDeviceId()).getInfo<CL_DEVICE_NAME>());

    return {program};
}

Module loadModuleFromDisk(const int device, const string &moduleKey,
                          const bool isJIT) {
    const string &cacheDirectory = getCacheDirectory();
    if (cacheDirectory.empty()) return Module{};

    auto &dev              = opencl::getDevice(device);
    const string cacheFile = cacheDirectory + AF_PATH_SEPARATOR +
                             getKernelCacheFilename(device, moduleKey);
    Program program;
    Module retVal{};
    try {
        std::ifstream in(cacheFile, std::ios::binary);
        if (!in.is_open()) {
            AF_ERROR("Unable to open binary cache file", AF_ERR_INTERNAL);
        }
        in.exceptions(std::ios::failbit | std::ios::badbit);

        // TODO Handle cases where program objects are created from contexts
        // having multiple devices
        size_t clbinHash = 0;
        in.read(reinterpret_cast<char *>(&clbinHash), sizeof(clbinHash));
        size_t clbinSize = 0;
        in.read(reinterpret_cast<char *>(&clbinSize), sizeof(clbinSize));
        vector<unsigned char> clbin(clbinSize);
        in.read(reinterpret_cast<char *>(clbin.data()), clbinSize);
        in.close();

        const size_t recomputedHash =
            deterministicHash(clbin.data(), clbinSize);
        if (recomputedHash != clbinHash) {
            AF_ERROR("Binary on disk seems to be corrupted", AF_ERR_LOAD_SYM);
        }
        program = Program(opencl::getContext(), {dev}, {clbin});
        program.build();

        AF_TRACE("{{{:<20} : loaded from {} for {} }}", moduleKey, cacheFile,
                 dev.getInfo<CL_DEVICE_NAME>());
        retVal.set(program);
    } catch (const AfError &e) {
        if (e.getError() == AF_ERR_LOAD_SYM) {
            AF_TRACE(
                "{{{:<20} : Corrupt binary({}) found on disk for {}, removed}}",
                moduleKey, cacheFile, dev.getInfo<CL_DEVICE_NAME>());
        } else {
            AF_TRACE("{{{:<20} : Unable to open {} for {}}}", moduleKey,
                     cacheFile, dev.getInfo<CL_DEVICE_NAME>());
        }
        removeFile(cacheFile);
    } catch (const std::ios_base::failure &e) {
        AF_TRACE("{{{:<20} : IO failure while loading {} for {}; {}}}",
                 moduleKey, cacheFile, dev.getInfo<CL_DEVICE_NAME>(), e.what());
        removeFile(cacheFile);
    } catch (const cl::Error &e) {
        AF_TRACE(
            "{{{:<20} : Loading OpenCL binary({}) failed for {}; {}, Build "
            "Log: {}}}",
            moduleKey, cacheFile, dev.getInfo<CL_DEVICE_NAME>(), e.what(),
            getProgramBuildLog(program));
        removeFile(cacheFile);
    }
    return retVal;
}

Kernel getKernel(const Module &mod, const string &nameExpr,
                 const bool sourceWasJIT) {
    UNUSED(sourceWasJIT);
    return {&mod.get(), cl::Kernel(mod.get(), nameExpr.c_str())};
}

}  // namespace common

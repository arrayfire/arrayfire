/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cl2hpp.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/KParam.hpp>
#include <platform.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <types.hpp>

#include <sstream>

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::NDRange;
using cl::Program;
using std::ostringstream;
using std::string;

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

// TODO(pradeep) remove this version after porting to new cache interface
void buildProgram(cl::Program& prog, const char* ker_str, const int ker_len,
                  std::string options) {
    buildProgram(prog, 1, &ker_str, &ker_len, options);
}

// TODO(pradeep) remove this version after porting to new cache interface
void buildProgram(cl::Program& prog, const int num_files, const char** ker_strs,
                  const int* ker_lens, std::string options) {
    try {
        Program::Sources setSrc;
        setSrc.emplace_back(DEFAULT_MACROS_STR.c_str(),
                            DEFAULT_MACROS_STR.length());
        setSrc.emplace_back(KParam_hpp, KParam_hpp_len);

        for (int i = 0; i < num_files; i++) {
            setSrc.emplace_back(ker_strs[i], ker_lens[i]);
        }

        const std::string defaults =
            std::string(" -D dim_t=") +
            std::string(dtype_traits<dim_t>::getName());

        prog        = cl::Program(getContext(), setSrc);
        auto device = getDevice();

        std::string cl_std =
            std::string(" -cl-std=CL") +
            device.getInfo<CL_DEVICE_OPENCL_C_VERSION>().substr(9, 3);

        // Braces needed to list initialize the vector for the first argument
        prog.build({device}, (cl_std + defaults + options).c_str());

    } catch (...) {
        SHOW_BUILD_INFO(prog);
        throw;
    }
}

cl::Program buildProgram(const std::vector<std::string>& kernelSources,
                         const std::vector<std::string>& compileOpts) {
    cl::Program retVal;
    try {
        static const std::string defaults =
            std::string(" -D dim_t=") +
            std::string(dtype_traits<dim_t>::getName());

        auto device = getDevice();

        const std::string cl_std =
            std::string(" -cl-std=CL") +
            device.getInfo<CL_DEVICE_OPENCL_C_VERSION>().substr(9, 3);

        Program::Sources sources;
        sources.emplace_back(DEFAULT_MACROS_STR);
        sources.emplace_back(KParam_hpp, KParam_hpp_len);

        for (auto ksrc : kernelSources) { sources.emplace_back(ksrc); }

        retVal = cl::Program(getContext(), sources);

        ostringstream options;
        for (auto& opt : compileOpts) { options << opt; }

        retVal.build({device}, (cl_std + defaults + options.str()).c_str());
    } catch (...) {
        SHOW_BUILD_INFO(retVal);
        throw;
    }
    return retVal;
}

}  // namespace opencl

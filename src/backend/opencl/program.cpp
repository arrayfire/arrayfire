/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <debug_opencl.hpp>
#include <kernel_headers/KParam.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <utility>

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {

void buildProgram(cl::Program &prog, const char *ker_str, const int ker_len,
                  const std::string &options) {
    buildProgram(prog, 1, &ker_str, &ker_len, options);
}

void buildProgram(cl::Program &prog, const int num_files, const char **ker_strs,
                  const int *ker_lens, const std::string &options) {
    try {
        constexpr char kernel_header[] =
            R"jit(#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#ifdef USE_HALF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#else
#define half short
#endif
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164
#endif
)jit";

        Program::Sources setSrc{
            {kernel_header, std::extent<decltype(kernel_header)>() - 1},
            {KParam_hpp, KParam_hpp_len}};

        for (int i = 0; i < num_files; i++) {
            setSrc.emplace_back(ker_strs[i], ker_lens[i]);
        }

        const std::string defaults =
            std::string(" -D dim_t=") +
            std::string(dtype_traits<dim_t>::getName());

        prog               = cl::Program(getContext(), setSrc);
        const auto &device = getDevice();

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
}  // namespace opencl

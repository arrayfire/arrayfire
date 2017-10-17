/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <program.hpp>
#include <traits.hpp>
#include <kernel_headers/KParam.hpp>
#include <debug_opencl.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
    const static std::string DEFAULT_MACROS_STR("\n\
                                           #ifdef USE_DOUBLE\n\
                                           #pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\
                                           #endif\n                     \
                                           #ifndef M_PI\n               \
                                           #define M_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164\n \
                                           #endif\n                     \
                                           ");
    void buildProgram(cl::Program &prog,
                      const char *ker_str, const int ker_len, std::string options)
    {
        buildProgram(prog, 1, &ker_str, &ker_len, options);
    }

    void buildProgram(cl::Program &prog, const int num_files,
                      const char **ker_strs, const int *ker_lens, std::string options)
    {
        try {
            Program::Sources setSrc;
            setSrc.emplace_back(DEFAULT_MACROS_STR.c_str(), DEFAULT_MACROS_STR.length());
            setSrc.emplace_back(KParam_hpp, KParam_hpp_len);

            for (int i = 0; i < num_files; i++) {
                setSrc.emplace_back(ker_strs[i], ker_lens[i]);
            }

            const std::string defaults =
                std::string(" -D dim_t=") +
                std::string(dtype_traits<dim_t>::getName());

            prog = cl::Program(getContext(), setSrc);
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
}

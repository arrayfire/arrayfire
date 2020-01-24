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
#include <common/defines.hpp>
#include <err_opencl.hpp>
#include <platform.hpp>
#include <program.hpp>

using detail::Kernel;
using std::string;
using std::vector;

namespace common {

Kernel compileKernel(const string &kernelName, const string &tInstance,
                     const vector<string> &sources,
                     const vector<string> &compileOpts, const bool isJIT) {
    UNUSED(isJIT);
    UNUSED(tInstance);

    auto prog = detail::buildProgram(sources, compileOpts);
    auto prg  = new cl::Program(prog);
    auto krn =
        new cl::Kernel(*static_cast<cl::Program *>(prg), kernelName.c_str());
    return {prg, krn};
}

Kernel loadKernel(const int device, const string &nameExpr) {
    OPENCL_NOT_SUPPORTED(
        "Disk caching OpenCL kernel binaries is not yet supported");
    return {nullptr, nullptr};
}

}  // namespace common

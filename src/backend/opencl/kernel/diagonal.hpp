/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/diag_create.hpp>
#include <kernel_headers/diag_extract.hpp>
#include <math.hpp>
#include <program.hpp>
#include "../traits.hpp"
#include "config.hpp"

using af::scalar_to_option;
using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {
namespace kernel {
template<typename T>
std::string generateOptionsString() {
    std::ostringstream options;
    options << " -D T=" << dtype_traits<T>::getName() << " -D ZERO=(T)("
            << scalar_to_option(scalar<T>(0)) << ")";
    if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
        options << " -D USE_DOUBLE";
    }
    return options.str();
}

template<typename T>
static void diagCreate(Param out, Param in, int num) {
    std::string refName = std::string("diagCreateKernel_") +
                          std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::string options    = generateOptionsString<T>();
        const char* ker_strs[] = {diag_create_cl};
        const int ker_lens[]   = {diag_create_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options);
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "diagCreateKernel");

        addKernelToCache(device, refName, entry);
    }

    NDRange local(32, 8);
    int groups_x = divup(out.info.dims[0], local[0]);
    int groups_y = divup(out.info.dims[1], local[1]);
    NDRange global(groups_x * local[0] * out.info.dims[2], groups_y * local[1]);

    auto diagCreateOp =
        KernelFunctor<Buffer, const KParam, Buffer, const KParam, int, int>(
            *entry.ker);

    diagCreateOp(EnqueueArgs(getQueue(), global, local), *(out.data), out.info,
                 *(in.data), in.info, num, groups_x);

    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
static void diagExtract(Param out, Param in, int num) {
    std::string refName = std::string("diagExtractKernel_") +
                          std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog == 0 && entry.ker == 0) {
        std::string options    = generateOptionsString<T>();
        const char* ker_strs[] = {diag_extract_cl};
        const int ker_lens[]   = {diag_extract_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options);
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "diagExtractKernel");

        addKernelToCache(device, refName, entry);
    }

    NDRange local(256, 1);
    int groups_x = divup(out.info.dims[0], local[0]);
    int groups_z = out.info.dims[2];
    NDRange global(groups_x * local[0], groups_z * local[1] * out.info.dims[3]);

    auto diagExtractOp =
        KernelFunctor<Buffer, const KParam, Buffer, const KParam, int, int>(
            *entry.ker);

    diagExtractOp(EnqueueArgs(getQueue(), global, local), *(out.data), out.info,
                  *(in.data), in.info, num, groups_z);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl

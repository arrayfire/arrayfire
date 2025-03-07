/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel/config.hpp>
#include <kernel_headers/susan.hpp>
#include <memory.hpp>
#include <traits.hpp>
#include <af/defines.h>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {
constexpr unsigned SUSAN_THREADS_X = 16;
constexpr unsigned SUSAN_THREADS_Y = 16;

template<typename T>
void susan(cl::Buffer* out, const cl::Buffer* in, const unsigned in_off,
           const unsigned idim0, const unsigned idim1, const float t,
           const float g, const unsigned edge, const unsigned radius) {
    const size_t LOCAL_MEM_SIZE =
        (SUSAN_THREADS_X + 2 * radius) * (SUSAN_THREADS_Y + 2 * radius);

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(radius),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineValue(LOCAL_MEM_SIZE),
        DefineKeyValue(BLOCK_X, SUSAN_THREADS_X),
        DefineKeyValue(BLOCK_Y, SUSAN_THREADS_Y),
        DefineKeyValue(RADIUS, radius),
        DefineKey(RESPONSE),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto susan = common::getKernel("susan_responses", {{susan_cl_src}}, targs,
                                   compileOpts);

    cl::NDRange local(SUSAN_THREADS_X, SUSAN_THREADS_Y);
    cl::NDRange global(divup(idim0 - 2 * edge, local[0]) * local[0],
                       divup(idim1 - 2 * edge, local[1]) * local[1]);

    susan(cl::EnqueueArgs(getQueue(), global, local), *out, *in, in_off, idim0,
          idim1, t, g, edge);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
unsigned nonMaximal(cl::Buffer* x_out, cl::Buffer* y_out, cl::Buffer* resp_out,
                    const unsigned idim0, const unsigned idim1,
                    const cl::Buffer* resp_in, const unsigned edge,
                    const unsigned max_corners) {
    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKey(NONMAX),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto nonMax =
        common::getKernel("non_maximal", {{susan_cl_src}}, targs, compileOpts);

    unsigned corners_found = 0;
    auto d_corners_found   = memAlloc<unsigned>(1);
    getQueue().enqueueFillBuffer(*d_corners_found, corners_found, 0,
                                 sizeof(unsigned));

    cl::NDRange local(SUSAN_THREADS_X, SUSAN_THREADS_Y);
    cl::NDRange global(divup(idim0 - 2 * edge, local[0]) * local[0],
                       divup(idim1 - 2 * edge, local[1]) * local[1]);

    nonMax(cl::EnqueueArgs(getQueue(), global, local), *x_out, *y_out,
           *resp_out, *d_corners_found, idim0, idim1, *resp_in, edge,
           max_corners);
    getQueue().enqueueReadBuffer(*d_corners_found, CL_TRUE, 0, sizeof(unsigned),
                                 &corners_found);
    return corners_found;
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire

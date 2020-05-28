/*******************************************************
 * Copyright (c) 2019, ArrayFire
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
#include <kernel_headers/flood_fill.hpp>
#include <memory.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace opencl {
namespace kernel {

constexpr int THREADS   = 256;
constexpr int TILE_DIM  = 16;
constexpr int THREADS_X = TILE_DIM;
constexpr int THREADS_Y = THREADS / TILE_DIM;
constexpr int VALID     = 2;
constexpr int INVALID   = 1;
constexpr int ZERO      = 0;

static inline std::string floodfillSrc() {
    static const std::string src(flood_fill_cl, flood_fill_cl_len);
    return src;
}

template<typename T>
void initSeeds(Param out, const Param seedsx, const Param seedsy) {
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineValue(VALID),
        DefineKey(INIT_SEEDS),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto initSeeds = common::getKernel("init_seeds", {floodfillSrc()},
                                       {TemplateTypename<T>()}, options);
    cl::NDRange local(kernel::THREADS, 1, 1);
    cl::NDRange global(divup(seedsx.info.dims[0], local[0]) * local[0], 1, 1);

    initSeeds(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *seedsx.data, seedsx.info, *seedsy.data, seedsy.info);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void finalizeOutput(Param out, const T newValue) {
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineValue(VALID),
        DefineValue(ZERO),
        DefineKey(FINALIZE_OUTPUT),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto finalizeOut = common::getKernel("finalize_output", {floodfillSrc()},
                                         {TemplateTypename<T>()}, options);
    cl::NDRange local(kernel::THREADS_X, kernel::THREADS_Y, 1);
    cl::NDRange global(divup(out.info.dims[0], local[0]) * local[0],
                       divup(out.info.dims[1], local[1]) * local[1], 1);
    finalizeOut(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
                newValue);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void floodFill(Param out, const Param image, const Param seedsx,
               const Param seedsy, const T newValue, const T lowValue,
               const T highValue, const af::connectivity nlookup) {
    constexpr int RADIUS = 1;

    UNUSED(nlookup);
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineValue(RADIUS),
        DefineValue(VALID),
        DefineValue(INVALID),
        DefineValue(ZERO),
        DefineKey(FLOOD_FILL_STEP),
        DefineKeyValue(LMEM_WIDTH, (THREADS_X + 2 * RADIUS)),
        DefineKeyValue(LMEM_HEIGHT, (THREADS_Y + 2 * RADIUS)),
        DefineKeyValue(GROUP_SIZE, (THREADS_Y * THREADS_X)),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto floodStep = common::getKernel("flood_step", {floodfillSrc()},
                                       {TemplateTypename<T>()}, options);
    cl::NDRange local(kernel::THREADS_X, kernel::THREADS_Y, 1);
    cl::NDRange global(divup(out.info.dims[0], local[0]) * local[0],
                       divup(out.info.dims[1], local[1]) * local[1], 1);

    initSeeds<T>(out, seedsx, seedsy);

    int notFinished       = 1;
    cl::Buffer* dContinue = bufferAlloc(sizeof(int));

    while (notFinished) {
        notFinished = 0;
        getQueue().enqueueWriteBuffer(*dContinue, CL_FALSE, 0, sizeof(int),
                                      &notFinished);

        floodStep(cl::EnqueueArgs(getQueue(), global, local), *out.data,
                  out.info, *image.data, image.info, lowValue, highValue,
                  *dContinue);
        CL_DEBUG_FINISH(getQueue());

        getQueue().enqueueReadBuffer(*dContinue, CL_TRUE, 0, sizeof(int),
                                     &notFinished);
    }
    bufferFree(dContinue);
    finalizeOutput<T>(out, newValue);
}

}  // namespace kernel
}  // namespace opencl

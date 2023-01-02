/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel/config.hpp>
#include <kernel_headers/random_engine_mersenne.hpp>
#include <kernel_headers/random_engine_mersenne_init.hpp>
#include <kernel_headers/random_engine_philox.hpp>
#include <kernel_headers/random_engine_threefry.hpp>
#include <kernel_headers/random_engine_write.hpp>
#include <random_engine.hpp>
#include <traits.hpp>
#include <af/defines.h>

#include <string>
#include <vector>

static const int N          = 351;
static const int TABLE_SIZE = 16;
static const int MAX_BLOCKS = 32;
static const int STATE_SIZE = (256 * 3);

namespace arrayfire {
namespace opencl {
namespace kernel {
static const uint THREADS = 256;

template<typename T>
static Kernel getRandomEngineKernel(const af_random_engine_type type,
                                    const int kerIdx,
                                    const uint elementsPerBlock) {
    std::string key;
    std::vector<common::Source> sources{random_engine_write_cl_src};
    switch (type) {
        case AF_RANDOM_ENGINE_PHILOX_4X32_10:
            key = "philoxGenerator";
            sources.emplace_back(random_engine_philox_cl_src);
            break;
        case AF_RANDOM_ENGINE_THREEFRY_2X32_16:
            key = "threefryGenerator";
            sources.emplace_back(random_engine_threefry_cl_src);
            break;
        case AF_RANDOM_ENGINE_MERSENNE_GP11213:
            key = "mersenneGenerator";
            sources.emplace_back(random_engine_mersenne_cl_src);
            break;
        default:
            AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
    }
    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(kerIdx),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineValue(THREADS),
        DefineKeyValue(RAND_DIST, kerIdx),
    };
    if (type != AF_RANDOM_ENGINE_MERSENNE_GP11213) {
        options.emplace_back(
            DefineKeyValue(ELEMENTS_PER_BLOCK, elementsPerBlock));
    }
#if defined(OS_MAC)  // Because apple is "special"
    options.emplace_back(DefineKey(IS_APPLE));
    options.emplace_back(DefineKeyValue(log10_val, std::log(10.0)));
#endif
    options.emplace_back(getTypeBuildDefinition<T>());

    return common::getKernel(key, sources, targs, options);
}

template<typename T>
static void randomDistribution(cl::Buffer out, const size_t elements,
                               const af_random_engine_type type,
                               const uintl &seed, uintl &counter, int kerIdx) {
    uint elementsPerBlock = THREADS * 4 * sizeof(uint) / sizeof(T);
    uint groups           = divup(elements, elementsPerBlock);

    uint hi  = seed >> 32;
    uint lo  = seed;
    uint hic = counter >> 32;
    uint loc = counter;

    cl::NDRange local(THREADS, 1);
    cl::NDRange global(THREADS * groups, 1);

    if ((type == AF_RANDOM_ENGINE_PHILOX_4X32_10) ||
        (type == AF_RANDOM_ENGINE_THREEFRY_2X32_16)) {
        auto randomEngineOp =
            getRandomEngineKernel<T>(type, kerIdx, elementsPerBlock);
        randomEngineOp(cl::EnqueueArgs(getQueue(), global, local), out,
                       static_cast<unsigned>(elements), hic, loc, hi, lo);
    }
    counter += elements;
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void randomDistribution(cl::Buffer out, const size_t elements, cl::Buffer state,
                        cl::Buffer pos, cl::Buffer sh1, cl::Buffer sh2,
                        const uint mask, cl::Buffer recursion_table,
                        cl::Buffer temper_table, int kerIdx) {
    int threads                = THREADS;
    int min_elements_per_block = 32 * THREADS * 4 * sizeof(uint) / sizeof(T);
    int blocks                 = divup(elements, min_elements_per_block);
    blocks                     = (blocks > MAX_BLOCKS) ? MAX_BLOCKS : blocks;
    uint elementsPerBlock      = divup(elements, blocks);

    cl::NDRange local(threads, 1);
    cl::NDRange global(threads * blocks, 1);
    auto randomEngineOp = getRandomEngineKernel<T>(
        AF_RANDOM_ENGINE_MERSENNE_GP11213, kerIdx, elementsPerBlock);
    randomEngineOp(cl::EnqueueArgs(getQueue(), global, local), out, state, pos,
                   sh1, sh2, mask, recursion_table, temper_table,
                   elementsPerBlock, static_cast<uint>(elements));
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void uniformDistributionCBRNG(cl::Buffer out, const size_t elements,
                              const af_random_engine_type type,
                              const uintl &seed, uintl &counter) {
    randomDistribution<T>(out, elements, type, seed, counter, 0);
}

template<typename T>
void normalDistributionCBRNG(cl::Buffer out, const size_t elements,
                             const af_random_engine_type type,
                             const uintl &seed, uintl &counter) {
    randomDistribution<T>(out, elements, type, seed, counter, 1);
}

template<typename T>
void uniformDistributionMT(cl::Buffer out, const size_t elements,
                           cl::Buffer state, cl::Buffer pos, cl::Buffer sh1,
                           cl::Buffer sh2, const uint mask,
                           cl::Buffer recursion_table,
                           cl::Buffer temper_table) {
    randomDistribution<T>(out, elements, state, pos, sh1, sh2, mask,
                          recursion_table, temper_table, 0);
}

template<typename T>
void normalDistributionMT(cl::Buffer out, const size_t elements,
                          cl::Buffer state, cl::Buffer pos, cl::Buffer sh1,
                          cl::Buffer sh2, const uint mask,
                          cl::Buffer recursion_table, cl::Buffer temper_table) {
    randomDistribution<T>(out, elements, state, pos, sh1, sh2, mask,
                          recursion_table, temper_table, 1);
}

void initMersenneState(cl::Buffer state, cl::Buffer table, const uintl &seed) {
    cl::NDRange local(THREADS_PER_GROUP, 1);
    cl::NDRange global(local[0] * MAX_BLOCKS, 1);

    auto initOp = common::getKernel("mersenneInitState",
                                    {{random_engine_mersenne_init_cl_src}}, {});
    initOp(cl::EnqueueArgs(getQueue(), global, local), state, table, seed);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire

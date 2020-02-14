/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cache.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <err_opencl.hpp>
#include <kernel_headers/random_engine_philox.hpp>
#include <kernel_headers/random_engine_threefry.hpp>
#include <kernel_headers/random_engine_write.hpp>
#include <platform.hpp>
#include <program.hpp>
#include <random_engine.hpp>
#include <traits.hpp>
#include <type_util.hpp>
#include <af/defines.h>
#include <sstream>
#include <string>
#include "config.hpp"

#include <kernel_headers/random_engine_mersenne.hpp>
#include <kernel_headers/random_engine_mersenne_init.hpp>

static const int N          = 351;
static const int TABLE_SIZE = 16;
static const int MAX_BLOCKS = 32;
static const int STATE_SIZE = (256 * 3);

namespace opencl {
namespace kernel {
static const uint THREADS = 256;

template<typename T>
static cl::Kernel get_random_engine_kernel(const af_random_engine_type type,
                                           const int kerIdx,
                                           const uint elementsPerBlock) {
    using std::string;
    using std::to_string;
    string engineName;
    const char *ker_strs[2];
    int ker_lens[2];
    ker_strs[0] = random_engine_write_cl;
    ker_lens[0] = random_engine_write_cl_len;
    switch (type) {
        case AF_RANDOM_ENGINE_PHILOX_4X32_10:
            engineName  = "Philox";
            ker_strs[1] = random_engine_philox_cl;
            ker_lens[1] = random_engine_philox_cl_len;
            break;
        case AF_RANDOM_ENGINE_THREEFRY_2X32_16:
            engineName  = "Threefry";
            ker_strs[1] = random_engine_threefry_cl;
            ker_lens[1] = random_engine_threefry_cl_len;
            break;
        case AF_RANDOM_ENGINE_MERSENNE_GP11213:
            engineName  = "Mersenne";
            ker_strs[1] = random_engine_mersenne_cl;
            ker_lens[1] = random_engine_mersenne_cl_len;
            break;
        default:
            AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
    }

    string ref_name = "random_engine_kernel_" + engineName + "_" +
                      string(dtype_traits<T>::getName()) + "_" +
                      to_string(kerIdx);
    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D THREADS=" << THREADS << " -D RAND_DIST=" << kerIdx;
        if (type != AF_RANDOM_ENGINE_MERSENNE_GP11213) {
            options << " -D ELEMENTS_PER_BLOCK=" << elementsPerBlock;
        }
        if (std::is_same<T, double>::value) { options << " -D USE_DOUBLE"; }
        if (std::is_same<T, common::half>::value) { options << " -D USE_HALF"; }
#if defined(OS_MAC)  // Because apple is "special"
        options << " -D IS_APPLE"
                << " -D log10_val=" << std::log(10.0);
#endif
        cl::Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());
        entry.prog = new cl::Program(prog);
        entry.ker  = new cl::Kernel(*entry.prog, "generate");

        addKernelToCache(device, ref_name, entry);
    }

    return *entry.ker;
}

static cl::Kernel get_mersenne_init_kernel(void) {
    using std::string;
    using std::to_string;
    string engineName;
    const char *ker_str = random_engine_mersenne_init_cl;
    int ker_len         = random_engine_mersenne_init_cl_len;
    string ref_name     = "mersenne_init";
    int device          = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::string emptyOptionString;
        cl::Program prog;
        buildProgram(prog, 1, &ker_str, &ker_len, emptyOptionString);
        entry.prog = new cl::Program(prog);
        entry.ker  = new cl::Kernel(*entry.prog, "initState");

        addKernelToCache(device, ref_name, entry);
    }

    return *entry.ker;
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
        cl::Kernel ker =
            get_random_engine_kernel<T>(type, kerIdx, elementsPerBlock);
        auto randomEngineOp =
            cl::KernelFunctor<cl::Buffer, uint, uint, uint, uint, uint>(ker);
        randomEngineOp(cl::EnqueueArgs(getQueue(), global, local), out,
                       elements, hic, loc, hi, lo);
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
    int elementsPerBlock       = divup(elements, blocks);

    cl::NDRange local(threads, 1);
    cl::NDRange global(threads * blocks, 1);
    cl::Kernel ker = get_random_engine_kernel<T>(
        AF_RANDOM_ENGINE_MERSENNE_GP11213, kerIdx, elementsPerBlock);
    auto randomEngineOp =
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                          cl::Buffer, uint, cl::Buffer, cl::Buffer, uint, uint>(
            ker);
    randomEngineOp(cl::EnqueueArgs(getQueue(), global, local), out, state, pos,
                   sh1, sh2, mask, recursion_table, temper_table,
                   elementsPerBlock, elements);
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

    cl::Kernel ker = get_mersenne_init_kernel();
    auto initOp    = cl::KernelFunctor<cl::Buffer, cl::Buffer, uintl>(ker);
    initOp(cl::EnqueueArgs(getQueue(), global, local), state, table, seed);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl

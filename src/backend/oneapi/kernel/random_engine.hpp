/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <debug_oneapi.hpp>
#include <err_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <kernel/random_engine_mersenne.hpp>
#include <kernel/random_engine_philox.hpp>
#include <kernel/random_engine_threefry.hpp>
#include <kernel/random_engine_write.hpp>
#include <random_engine.hpp>
#include <traits.hpp>
#include <types.hpp>
#include <af/defines.h>

#include <functional>
#include <string>
#include <vector>

static const int N          = 351;
static const int TABLE_SIZE = 16;
static const int MAX_BLOCKS = 32;
static const int STATE_SIZE = (256 * 3);

namespace arrayfire {
namespace oneapi {
namespace kernel {

static const uint THREADS           = 256;
static const uint THREADS_PER_GROUP = 256;
static const uint THREADS_X         = 32;
static const uint THREADS_Y         = THREADS_PER_GROUP / THREADS_X;
static const uint REPEAT            = 32;

template<typename T>
void uniformDistributionCBRNG(Param<T> out, const size_t elements,
                              const af_random_engine_type type,
                              const uintl &seed, uintl &counter) {
    int threads          = THREADS;
    int elementsPerBlock = threads * 4 * sizeof(uint) / sizeof(T);
    int blocks           = divup(elements, elementsPerBlock);
    uint hi              = seed >> 32;
    uint lo              = seed;
    uint hic             = counter >> 32;
    uint loc             = counter;
    sycl::nd_range<1> ndrange(sycl::range<1>(blocks * threads),
                              sycl::range<1>(threads));
    switch (type) {
        case AF_RANDOM_ENGINE_PHILOX_4X32_10:
            getQueue().submit([&](sycl::handler &h) {
                write_accessor<T> out_acc{*out.data, h};

                h.parallel_for(ndrange,
                               uniformPhilox<T>(out_acc, hi, lo, hic, loc,
                                                elementsPerBlock, elements));
            });
            ONEAPI_DEBUG_FINISH(getQueue());
            break;
        case AF_RANDOM_ENGINE_THREEFRY_2X32_16:
            getQueue().submit([&](sycl::handler &h) {
                write_accessor<T> out_acc{*out.data, h};

                h.parallel_for(ndrange,
                               uniformThreefry<T>(out_acc, hi, lo, hic, loc,
                                                  elementsPerBlock, elements));
            });
            ONEAPI_DEBUG_FINISH(getQueue());
            break;
        default:
            AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
    }
    counter += elements;
}

template<typename T>
void normalDistributionCBRNG(Param<T> out, const size_t elements,
                             const af_random_engine_type type,
                             const uintl &seed, uintl &counter) {
    int threads          = THREADS;
    int elementsPerBlock = threads * 4 * sizeof(uint) / sizeof(T);
    int blocks           = divup(elements, elementsPerBlock);
    uint hi              = seed >> 32;
    uint lo              = seed;
    uint hic             = counter >> 32;
    uint loc             = counter;
    sycl::nd_range<1> ndrange(sycl::range<1>(blocks * threads),
                              sycl::range<1>(threads));
    switch (type) {
        case AF_RANDOM_ENGINE_PHILOX_4X32_10:
            getQueue().submit([&](sycl::handler &h) {
                write_accessor<T> out_acc{*out.data, h};

                h.parallel_for(ndrange,
                               normalPhilox<T>(out_acc, hi, lo, hic, loc,
                                               elementsPerBlock, elements));
            });
            break;
        case AF_RANDOM_ENGINE_THREEFRY_2X32_16:
            getQueue().submit([&](sycl::handler &h) {
                write_accessor<T> out_acc{*out.data, h};

                h.parallel_for(ndrange,
                               normalThreefry<T>(out_acc, hi, lo, hic, loc,
                                                 elementsPerBlock, elements));
            });
            break;
        default:
            AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
    }
    counter += elements;
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T>
void uniformDistributionMT(Param<T> out, const size_t elements,
                           Param<uint> state, Param<uint> pos, Param<uint> sh1,
                           Param<uint> sh2, const uint mask,
                           Param<uint> recursion_table,
                           Param<uint> temper_table) {
    int threads                = THREADS;
    int min_elements_per_block = 32 * threads * 4 * sizeof(uint) / sizeof(T);
    int blocks                 = divup(elements, min_elements_per_block);
    blocks                     = (blocks > BLOCKS) ? BLOCKS : blocks;
    uint elementsPerBlock      = divup(elements, blocks);

    sycl::nd_range<1> ndrange(sycl::range<1>(blocks * threads),
                              sycl::range<1>(threads));
    getQueue().submit([&](sycl::handler &h) {
        write_accessor<T> out_acc{*out.data, h};
        auto state_acc     = state.data->get_access(h);
        auto pos_acc       = pos.data->get_access(h);
        auto sh1_acc       = sh1.data->get_access(h);
        auto sh2_acc       = sh2.data->get_access(h);
        auto recursion_acc = sh2.data->get_access(h);
        auto temper_acc    = sh2.data->get_access(h);

        auto lstate_acc     = sycl::local_accessor<uint, 1>(STATE_SIZE, h);
        auto lrecursion_acc = sycl::local_accessor<uint, 1>(TABLE_SIZE, h);
        auto ltemper_acc    = sycl::local_accessor<uint, 1>(TABLE_SIZE, h);

        h.parallel_for(
            ndrange, uniformMersenne<T>(
                         out_acc, state_acc, pos_acc, sh1_acc, sh2_acc, mask,
                         recursion_acc, temper_acc, lstate_acc, lrecursion_acc,
                         ltemper_acc, elementsPerBlock, elements));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T>
void normalDistributionMT(Param<T> out, const size_t elements,
                          Param<uint> state, Param<uint> pos, Param<uint> sh1,
                          Param<uint> sh2, const uint mask,
                          Param<uint> recursion_table,
                          Param<uint> temper_table) {
    int threads                = THREADS;
    int min_elements_per_block = 32 * threads * 4 * sizeof(uint) / sizeof(T);
    int blocks                 = divup(elements, min_elements_per_block);
    blocks                     = (blocks > BLOCKS) ? BLOCKS : blocks;
    uint elementsPerBlock      = divup(elements, blocks);

    sycl::nd_range<1> ndrange(sycl::range<1>(blocks * threads),
                              sycl::range<1>(threads));
    getQueue().submit([&](sycl::handler &h) {
        write_accessor<T> out_acc{*out.data, h};
        auto state_acc     = state.data->get_access(h);
        auto pos_acc       = pos.data->get_access(h);
        auto sh1_acc       = sh1.data->get_access(h);
        auto sh2_acc       = sh2.data->get_access(h);
        auto recursion_acc = sh2.data->get_access(h);
        auto temper_acc    = sh2.data->get_access(h);

        auto lstate_acc     = sycl::local_accessor<uint, 1>(STATE_SIZE, h);
        auto lrecursion_acc = sycl::local_accessor<uint, 1>(TABLE_SIZE, h);
        auto ltemper_acc    = sycl::local_accessor<uint, 1>(TABLE_SIZE, h);

        h.parallel_for(
            ndrange, normalMersenne<T>(out_acc, state_acc, pos_acc, sh1_acc,
                                       sh2_acc, mask, recursion_acc, temper_acc,
                                       lstate_acc, lrecursion_acc, ltemper_acc,
                                       elementsPerBlock, elements));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire

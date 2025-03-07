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
#include <backend.hpp>
#include <common/Binary.hpp>
#include <common/Transform.hpp>
#include <common/dispatch.hpp>
#include <debug_oneapi.hpp>
#include <err_oneapi.hpp>
#include <kernel/reduce_all.hpp>
#include <kernel/reduce_config.hpp>
#include <kernel/reduce_dim.hpp>
#include <kernel/reduce_first.hpp>
#include <math.hpp>
#include <memory.hpp>

#include <algorithm>
#include <climits>
#include <complex>
#include <iostream>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename Ti, typename To, af_op_t op>
void reduce_default_dispatch(Param<To> out, Param<Ti> in, int dim,
                             bool change_nan, double nanval) {
    switch (dim) {
        case 0:
            return reduce_first_default<Ti, To, op>(out, in, change_nan,
                                                    nanval);
        case 1:
            return reduce_dim_default<Ti, To, op, 1>(out, in, change_nan,
                                                     nanval);
        case 2:
            return reduce_dim_default<Ti, To, op, 2>(out, in, change_nan,
                                                     nanval);
        case 3:
            return reduce_dim_default<Ti, To, op, 3>(out, in, change_nan,
                                                     nanval);
    }
}

template<typename Ti, typename To, af_op_t op>
void reduce_cpu_dispatch(Param<To> out, Param<Ti> in, int dim, bool change_nan,
                         double nanval) {
    // TODO: use kernels optimized for SIMD-based subgroup sizes
    reduce_default_dispatch<Ti, To, op>(out, in, dim, change_nan, nanval);
}

template<typename Ti, typename To, af_op_t op>
void reduce_gpu_dispatch(Param<To> out, Param<Ti> in, int dim, bool change_nan,
                         double nanval) {
    // TODO: use kernels optimized for gpu subgroup sizes
    reduce_default_dispatch<Ti, To, op>(out, in, dim, change_nan, nanval);
}

template<typename Ti, typename To, af_op_t op>
void reduce(Param<To> out, Param<Ti> in, int dim, bool change_nan,
            double nanval) {
    // TODO: logic to dispatch to different kernels depending on device type
    if (getQueue().get_device().is_cpu()) {
        reduce_cpu_dispatch<Ti, To, op>(out, in, dim, change_nan, nanval);
    } else if (getQueue().get_device().is_gpu()) {
        reduce_gpu_dispatch<Ti, To, op>(out, in, dim, change_nan, nanval);
    } else {
        reduce_default_dispatch<Ti, To, op>(out, in, dim, change_nan, nanval);
    }
}

template<typename Ti, typename To, af_op_t op>
void reduce_all(Param<To> out, Param<Ti> in, bool change_nan, double nanval) {
    int in_elements =
        in.info.dims[0] * in.info.dims[1] * in.info.dims[2] * in.info.dims[3];
    bool is_linear = (in.info.strides[0] == 1);
    for (int k = 1; k < 4; k++) {
        is_linear &= (in.info.strides[k] ==
                      (in.info.strides[k - 1] * in.info.dims[k - 1]));
    }

    if (is_linear) {
        in.info.dims[0] = in_elements;
        for (int k = 1; k < 4; k++) {
            in.info.dims[k]    = 1;
            in.info.strides[k] = in_elements;
        }
    }

    uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
    threads_x      = std::min(threads_x, creduce::THREADS_PER_BLOCK);
    uint threads_y = creduce::THREADS_PER_BLOCK / threads_x;

    // TODO: perf REPEAT, consider removing or runtime eval
    // max problem size < SM resident threads, don't use REPEAT
    uint blocks_x = divup(in.info.dims[0], threads_x * creduce::REPEAT);
    uint blocks_y = divup(in.info.dims[1], threads_y);

    reduce_all_launcher_default<Ti, To, op>(out, in, blocks_x, blocks_y,
                                            threads_x, change_nan, nanval);
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire

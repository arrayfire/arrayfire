/*******************************************************
 * Copyright (c) 2014, ArrayFire
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
#include <kernel_headers/laset.hpp>
#include <magma_types.h>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

template<int num>
const char *laset_name() {
    return "laset_none";
}
template<>
const char *laset_name<0>() {
    return "laset_full";
}
template<>
const char *laset_name<1>() {
    return "laset_lower";
}
template<>
const char *laset_name<2>() {
    return "laset_upper";
}

template<typename T, int uplo>
void laset(int m, int n, T offdiag, T diag, cl_mem dA, size_t dA_offset,
           magma_int_t ldda, cl_command_queue queue) {
    constexpr int BLK_X = 64;
    constexpr int BLK_Y = 32;

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateArg(uplo),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()), DefineValue(BLK_X),
        DefineValue(BLK_Y),
        DefineKeyValue(IS_CPLX, static_cast<int>(iscplx<T>())),
        getTypeBuildDefinition<T>()};

    auto lasetOp =
        common::getKernel(laset_name<uplo>(), {{laset_cl_src}}, targs, options);

    int groups_x = (m - 1) / BLK_X + 1;
    int groups_y = (n - 1) / BLK_Y + 1;

    cl::NDRange local(BLK_X, 1);
    cl::NDRange global(groups_x * local[0], groups_y * local[1]);

    // retain the cl_mem object during cl::Buffer creation
    cl::Buffer dAObj(dA, true);

    cl::CommandQueue q(queue);
    lasetOp(cl::EnqueueArgs(q, global, local), m, n, offdiag, diag, dAObj,
            dA_offset, ldda);
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire

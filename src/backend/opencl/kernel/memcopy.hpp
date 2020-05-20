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
#include <common/traits.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/copy.hpp>
#include <kernel_headers/memcopy.hpp>
#include <traits.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace opencl {
namespace kernel {
typedef struct {
    dim_t dim[4];
} dims_t;

constexpr uint DIM0 = 32;
constexpr uint DIM1 = 8;

template<typename T>
void memcopy(cl::Buffer out, const dim_t *ostrides, const cl::Buffer in,
             const dim_t *idims, const dim_t *istrides, int offset,
             uint ndims) {
    static const std::string source(memcopy_cl, memcopy_cl_len);

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    options.emplace_back(getTypeBuildDefinition<T>());

    auto memCopy = common::getKernel("memCopy", {source}, targs, options);

    dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
    dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};
    dims_t _idims    = {{idims[0], idims[1], idims[2], idims[3]}};

    size_t local_size[2] = {DIM0, DIM1};
    if (ndims == 1) {
        local_size[0] *= local_size[1];
        local_size[1] = 1;
    }

    int groups_0 = divup(idims[0], local_size[0]);
    int groups_1 = divup(idims[1], local_size[1]);

    cl::NDRange local(local_size[0], local_size[1]);
    cl::NDRange global(groups_0 * idims[2] * local_size[0],
                       groups_1 * idims[3] * local_size[1]);

    memCopy(cl::EnqueueArgs(getQueue(), global, local), out, _ostrides, in,
            _idims, _istrides, offset, groups_0, groups_1);
    CL_DEBUG_FINISH(getQueue());
}

template<typename inType, typename outType>
void copy(Param dst, const Param src, const int ndims,
          const outType default_value, const double factor,
          const bool same_dims) {
    using std::string;

    static const string source(copy_cl, copy_cl_len);

    std::vector<TemplateArg> targs = {
        TemplateTypename<inType>(),
        TemplateTypename<outType>(),
        TemplateArg(same_dims),
    };
    std::vector<string> options = {
        DefineKeyValue(inType, dtype_traits<inType>::getName()),
        DefineKeyValue(outType, dtype_traits<outType>::getName()),
        string(" -D inType_" + string(dtype_traits<inType>::getName())),
        string(" -D outType_" + string(dtype_traits<outType>::getName())),
        DefineKeyValue(SAME_DIMS, static_cast<int>(same_dims)),
    };
    options.emplace_back(getTypeBuildDefinition<inType, outType>());

    auto copy = common::getKernel("reshapeCopy", {source}, targs, options);

    cl::NDRange local(DIM0, DIM1);
    size_t local_size[] = {DIM0, DIM1};

    local_size[0] *= local_size[1];
    if (ndims == 1) { local_size[1] = 1; }

    int blk_x = divup(dst.info.dims[0], local_size[0]);
    int blk_y = divup(dst.info.dims[1], local_size[1]);

    cl::NDRange global(blk_x * dst.info.dims[2] * DIM0,
                       blk_y * dst.info.dims[3] * DIM1);

    dims_t trgt_dims;
    if (same_dims) {
        trgt_dims = {{dst.info.dims[0], dst.info.dims[1], dst.info.dims[2],
                      dst.info.dims[3]}};
    } else {
        dim_t trgt_l = std::min(dst.info.dims[3], src.info.dims[3]);
        dim_t trgt_k = std::min(dst.info.dims[2], src.info.dims[2]);
        dim_t trgt_j = std::min(dst.info.dims[1], src.info.dims[1]);
        dim_t trgt_i = std::min(dst.info.dims[0], src.info.dims[0]);
        trgt_dims    = {{trgt_i, trgt_j, trgt_k, trgt_l}};
    }

    copy(cl::EnqueueArgs(getQueue(), global, local), *dst.data, dst.info,
         *src.data, src.info, default_value, (float)factor, trgt_dims, blk_x,
         blk_y);
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl

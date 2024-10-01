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
#include <kernel/config.hpp>
#include <kernel/reduce.hpp>
#include <kernel/scan_dim.hpp>
#include <kernel/scan_first.hpp>
#include <kernel/sort_by_key.hpp>
#include <kernel_headers/coo2dense.hpp>
#include <kernel_headers/csr2coo.hpp>
#include <kernel_headers/csr2dense.hpp>
#include <kernel_headers/dense2csr.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {
template<typename T>
void coo2dense(Param out, const Param values, const Param rowIdx,
               const Param colIdx) {
    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
        TemplateArg(REPEAT),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(reps, REPEAT),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto coo2dense = common::getKernel("coo2Dense", {{coo2dense_cl_src}},
                                       tmpltArgs, compileOpts);

    cl::NDRange local(THREADS_PER_GROUP, 1, 1);

    cl::NDRange global(
        divup(values.info.dims[0], local[0] * REPEAT) * THREADS_PER_GROUP, 1,
        1);

    coo2dense(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
              *values.data, values.info, *rowIdx.data, rowIdx.info,
              *colIdx.data, colIdx.info);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void csr2dense(Param output, const Param values, const Param rowIdx,
               const Param colIdx) {
    constexpr int MAX_GROUPS = 4096;
    // FIXME: This needs to be based non nonzeros per row
    constexpr int threads = 64;

    const int M = rowIdx.info.dims[0] - 1;

    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
        TemplateArg(threads),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(THREADS, threads),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto csr2dense = common::getKernel("csr2Dense", {{csr2dense_cl_src}},
                                       tmpltArgs, compileOpts);

    cl::NDRange local(threads, 1);
    int groups_x = std::min((int)(divup(M, local[0])), MAX_GROUPS);
    cl::NDRange global(local[0] * groups_x, 1);

    csr2dense(cl::EnqueueArgs(getQueue(), global, local), *output.data,
              *values.data, *rowIdx.data, *colIdx.data, M);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void dense2csr(Param values, Param rowIdx, Param colIdx, const Param dense) {
    constexpr bool IsComplex =
        std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(IS_CPLX, (IsComplex ? 1 : 0)),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto dense2Csr = common::getKernel("dense2Csr", {{dense2csr_cl_src}},
                                       tmpltArgs, compileOpts);

    int num_rows = dense.info.dims[0];
    int num_cols = dense.info.dims[1];

    // sd1 contains output of scan along dim 1 of dense
    Array<int> sd1 = createEmptyArray<int>(dim4(num_rows, num_cols));
    // rd1 contains output of nonzero count along dim 1 along dense
    Array<int> rd1 = createEmptyArray<int>(num_rows);

    scanDim<T, int, af_notzero_t>(sd1, dense, 1, true);
    reduceDim<T, int, af_notzero_t>(rd1, dense, 0, 0, 1);
    scanFirst<int, int, af_add_t>(rowIdx, rd1, false);

    int nnz = values.info.dims[0];
    getQueue().enqueueFillBuffer(
        *rowIdx.data, nnz,
        rowIdx.info.offset + (rowIdx.info.dims[0] - 1) * sizeof(int),
        sizeof(int));

    cl::NDRange local(THREADS_X, THREADS_Y);
    int groups_x = divup(dense.info.dims[0], local[0]);
    int groups_y = divup(dense.info.dims[1], local[1]);
    cl::NDRange global(groups_x * local[0], groups_y * local[1]);

    const Param sdParam = sd1;

    dense2Csr(cl::EnqueueArgs(getQueue(), global, local), *values.data,
              *colIdx.data, *dense.data, dense.info, *sdParam.data,
              sdParam.info, *rowIdx.data);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void swapIndex(Param ovalues, Param oindex, const Param ivalues,
               const cl::Buffer *iindex, const Param swapIdx) {
    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto swapIndex = common::getKernel("swapIndex", {{csr2coo_cl_src}},
                                       tmpltArgs, compileOpts);

    cl::NDRange global(ovalues.info.dims[0], 1, 1);

    swapIndex(cl::EnqueueArgs(getQueue(), global), *ovalues.data, *oindex.data,
              *ivalues.data, *iindex, *swapIdx.data,
              static_cast<int>(ovalues.info.dims[0]));
    CL_DEBUG_FINISH(getQueue());
}

template<typename T>
void csr2coo(Param ovalues, Param orowIdx, Param ocolIdx, const Param ivalues,
             const Param irowIdx, const Param icolIdx, Param index) {
    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto csr2coo = common::getKernel("csr2Coo", {{csr2coo_cl_src}}, tmpltArgs,
                                     compileOpts);

    const int MAX_GROUPS = 4096;
    int M                = irowIdx.info.dims[0] - 1;
    // FIXME: This needs to be based non nonzeros per row
    int threads = 64;

    cl::Buffer *scratch = bufferAlloc(orowIdx.info.dims[0] * sizeof(int));

    cl::NDRange local(threads, 1);
    int groups_x = std::min((int)(divup(M, local[0])), MAX_GROUPS);
    cl::NDRange global(local[0] * groups_x, 1);

    csr2coo(cl::EnqueueArgs(getQueue(), global, local), *scratch, *ocolIdx.data,
            *irowIdx.data, *icolIdx.data, M);

    // Now we need to sort this into column major
    kernel::sort0ByKeyIterative<int, int>(ocolIdx, index, true);

    // Now use index to sort values and rows
    kernel::swapIndex<T>(ovalues, orowIdx, ivalues, scratch, index);

    CL_DEBUG_FINISH(getQueue());

    bufferFree(scratch);
}

template<typename T>
void coo2csr(Param ovalues, Param orowIdx, Param ocolIdx, const Param ivalues,
             const Param irowIdx, const Param icolIdx, Param index,
             Param rowCopy, const int M) {
    std::vector<TemplateArg> tmpltArgs = {
        TemplateTypename<T>(),
    };
    std::vector<std::string> compileOpts = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    compileOpts.emplace_back(getTypeBuildDefinition<T>());

    auto csrReduce = common::getKernel("csrReduce", {{csr2coo_cl_src}},
                                       tmpltArgs, compileOpts);

    // Now we need to sort this into column major
    kernel::sort0ByKeyIterative<int, int>(rowCopy, index, true);

    // Now use index to sort values and rows
    kernel::swapIndex<T>(ovalues, ocolIdx, ivalues, icolIdx.data, index);

    CL_DEBUG_FINISH(getQueue());

    cl::NDRange global(irowIdx.info.dims[0], 1, 1);

    csrReduce(cl::EnqueueArgs(getQueue(), global), *orowIdx.data, *rowCopy.data,
              M, static_cast<int>(ovalues.info.dims[0]));
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire

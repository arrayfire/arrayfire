/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_oneapi.hpp>
// #include <kernel/config.hpp>
#include <kernel/reduce.hpp>
#include <kernel/scan_dim.hpp>
#include <kernel/scan_first.hpp>
#include <kernel/sort_by_key.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
using read_accessor = sycl::accessor<T, 1, sycl::access::mode::read>;
template<typename T>
using write_accessor = sycl::accessor<T, 1, sycl::access::mode::write>;


// template<typename T>
// void coo2dense(Param out, const Param values, const Param rowIdx,
//                const Param colIdx) {
//     std::vector<TemplateArg> tmpltArgs = {
//         TemplateTypename<T>(),
//         TemplateArg(REPEAT),
//     };
//     std::vector<std::string> compileOpts = {
//         DefineKeyValue(T, dtype_traits<T>::getName()),
//         DefineKeyValue(resp, REPEAT),
//     };
//     compileOpts.emplace_back(getTypeBuildDefinition<T>());

//     auto coo2dense = common::getKernel("coo2Dense", {{coo2dense_cl_src}},
//                                        tmpltArgs, compileOpts);

//     cl::NDRange local(THREADS_PER_GROUP, 1, 1);

//     cl::NDRange global(
//         divup(out.info.dims[0], local[0] * REPEAT) * THREADS_PER_GROUP, 1, 1);

//     coo2dense(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
//               *values.data, values.info, *rowIdx.data, rowIdx.info,
//               *colIdx.data, colIdx.info);
//     CL_DEBUG_FINISH(getQueue());
// }

// template<typename T>
// void csr2dense(Param output, const Param values, const Param rowIdx,
//                const Param colIdx) {
//     constexpr int MAX_GROUPS = 4096;
//     // FIXME: This needs to be based non nonzeros per row
//     constexpr int threads = 64;

//     const int M = rowIdx.info.dims[0] - 1;

//     std::vector<TemplateArg> tmpltArgs = {
//         TemplateTypename<T>(),
//         TemplateArg(threads),
//     };
//     std::vector<std::string> compileOpts = {
//         DefineKeyValue(T, dtype_traits<T>::getName()),
//         DefineKeyValue(THREADS, threads),
//     };
//     compileOpts.emplace_back(getTypeBuildDefinition<T>());

//     auto csr2dense = common::getKernel("csr2Dense", {{csr2dense_cl_src}},
//                                        tmpltArgs, compileOpts);

//     cl::NDRange local(threads, 1);
//     int groups_x = std::min((int)(divup(M, local[0])), MAX_GROUPS);
//     cl::NDRange global(local[0] * groups_x, 1);

//     csr2dense(cl::EnqueueArgs(getQueue(), global, local), *output.data,
//               *values.data, *rowIdx.data, *colIdx.data, M);
//     CL_DEBUG_FINISH(getQueue());
// }

template<typename T>
class dense2csrCreateKernel {
public:
  dense2csrCreateKernel(write_accessor<T> svalptr,
                        write_accessor<int> scolptr,
                        read_accessor<T> dvalptr,
                        const KParam valinfo,
                        read_accessor<int> dcolptr,
                        const KParam colinfo,
                        read_accessor<int> rowptr)
    : svalptr_(svalptr), scolptr_(scolptr), dvalptr_(dvalptr), valinfo_(valinfo), dcolptr_(dcolptr), colinfo_(colinfo), rowptr_(rowptr) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

    // int gidx = it.get_global_id(0);
    // int gidy = it.get_global_id(1);

    // if (gidx >= valinfo_.dims[0]) return;
    // if (gidy >= valinfo_.dims[1]) return;

    // int rowoff = rowptr_[gidx];
    // svalptr_ += rowoff;
    // scolptr_ += rowoff;

    // dvalptr_ += valinfo_.offset;
    // dcolptr_ += colinfo_.offset;

    // int idx = gidx + gidy * valinfo_.strides[1];
    // T val   = dvalptr_[gidx + gidy * valinfo_.strides[1]];
    // if (IS_ZERO(val)) return;

    // int oloc          = dcolptr_[gidx + gidy * colinfo_.strides[1]];
    // svalptr_[oloc - 1] = val;
    // scolptr_[oloc - 1] = gidy;
}

private:
write_accessor<T> svalptr_;
write_accessor<int> scolptr_;
read_accessor<T> dvalptr_;
const KParam valinfo_;
read_accessor<int> dcolptr_;
const KParam colinfo_;
read_accessor<int> rowptr_;
};

template<typename T>
void dense2csr(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
               const Param<T> dense) {
    constexpr bool IsComplex =
      std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

    int num_rows = dense.info.dims[0];
    int num_cols = dense.info.dims[1];

    // sd1 contains output of scan along dim 1 of dense
    Array<int> sd1 = createEmptyArray<int>(dim4(num_rows, num_cols));
    // rd1 contains output of nonzero count along dim 1 along dense
    Array<int> rd1 = createEmptyArray<int>(num_rows);

    // scanDim<T, int, af_notzero_t>(sd1, dense, 1, true);
    printf("!!! %d\n", __LINE__);
    scan_dim<T, int, af_notzero_t, 1>(sd1, dense, true);
    // reduceDim<T, int, af_notzero_t>(rd1, dense, 0, 0, 1);
    printf("!!! %d\n", __LINE__);
    reduce_dim_default<T, int, af_notzero_t, 1>(rd1, dense, 0, 0);
    // scanFirst<int, int, af_add_t>(rowIdx, rd1, false);
    printf("!!! %d\n", __LINE__);
    scan_first<int, int, af_add_t>(rowIdx, rd1, false);

    printf("!!! %d\n", __LINE__);
    const int nnz = values.info.dims[0];
    const sycl::range<1> fill_range(rowIdx.info.offset + (rowIdx.info.dims[0] - 1) * sizeof(int));
    getQueue().submit([&](auto &h) {
        sycl::accessor d_rowIdx{*rowIdx.data, h, fill_range, sycl::write_only, sycl::no_init};
        h.fill(d_rowIdx, nnz);
    });

    auto local   = sycl::range(THREADS_X, THREADS_Y);
    int groups_x = divup(dense.info.dims[0], local[0]);
    int groups_y = divup(dense.info.dims[1], local[1]);
    auto global  = sycl::range(groups_x * local[0], groups_y * local[1]);

    const Param<int> sdParam = sd1;

    getQueue().submit([&](auto &h) {
        sycl::accessor d_dense{*dense.data, h, sycl::read_only};
        sycl::accessor d_sdParam{*sdParam.data, h, sycl::read_only};
        sycl::accessor d_rowIdx{*rowIdx.data, h, sycl::read_only};
        sycl::accessor d_values{*values.data, h, sycl::write_only, sycl::no_init};
        sycl::accessor d_colIdx{*colIdx.data, h, sycl::write_only, sycl::no_init};
        h.parallel_for(sycl::nd_range{global, local},
                       dense2csrCreateKernel<T>(
        d_values,
        d_colIdx,
        d_dense,
        dense.info,
        d_sdParam,
        sdParam.info,
        d_rowIdx));
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}

// template<typename T>
// void swapIndex(Param ovalues, Param oindex, const Param ivalues,
//                const cl::Buffer *iindex, const Param swapIdx) {
//     std::vector<TemplateArg> tmpltArgs = {
//         TemplateTypename<T>(),
//     };
//     std::vector<std::string> compileOpts = {
//         DefineKeyValue(T, dtype_traits<T>::getName()),
//     };
//     compileOpts.emplace_back(getTypeBuildDefinition<T>());

//     auto swapIndex = common::getKernel("swapIndex", {{csr2coo_cl_src}},
//                                        tmpltArgs, compileOpts);

//     cl::NDRange global(ovalues.info.dims[0], 1, 1);

//     swapIndex(cl::EnqueueArgs(getQueue(), global), *ovalues.data, *oindex.data,
//               *ivalues.data, *iindex, *swapIdx.data,
//               static_cast<int>(ovalues.info.dims[0]));
//     CL_DEBUG_FINISH(getQueue());
// }

// template<typename T>
// void csr2coo(Param ovalues, Param orowIdx, Param ocolIdx, const Param ivalues,
//              const Param irowIdx, const Param icolIdx, Param index) {
//     std::vector<TemplateArg> tmpltArgs = {
//         TemplateTypename<T>(),
//     };
//     std::vector<std::string> compileOpts = {
//         DefineKeyValue(T, dtype_traits<T>::getName()),
//     };
//     compileOpts.emplace_back(getTypeBuildDefinition<T>());

//     auto csr2coo = common::getKernel("csr2Coo", {{csr2coo_cl_src}}, tmpltArgs,
//                                      compileOpts);

//     const int MAX_GROUPS = 4096;
//     int M                = irowIdx.info.dims[0] - 1;
//     // FIXME: This needs to be based non nonzeros per row
//     int threads = 64;

//     cl::Buffer *scratch = bufferAlloc(orowIdx.info.dims[0] * sizeof(int));

//     cl::NDRange local(threads, 1);
//     int groups_x = std::min((int)(divup(M, local[0])), MAX_GROUPS);
//     cl::NDRange global(local[0] * groups_x, 1);

//     csr2coo(cl::EnqueueArgs(getQueue(), global, local), *scratch, *ocolIdx.data,
//             *irowIdx.data, *icolIdx.data, M);

//     // Now we need to sort this into column major
//     kernel::sort0ByKeyIterative<int, int>(ocolIdx, index, true);

//     // Now use index to sort values and rows
//     kernel::swapIndex<T>(ovalues, orowIdx, ivalues, scratch, index);

//     CL_DEBUG_FINISH(getQueue());

//     bufferFree(scratch);
// }

// template<typename T>
// void coo2csr(Param ovalues, Param orowIdx, Param ocolIdx, const Param ivalues,
//              const Param irowIdx, const Param icolIdx, Param index,
//              Param rowCopy, const int M) {
//     std::vector<TemplateArg> tmpltArgs = {
//         TemplateTypename<T>(),
//     };
//     std::vector<std::string> compileOpts = {
//         DefineKeyValue(T, dtype_traits<T>::getName()),
//     };
//     compileOpts.emplace_back(getTypeBuildDefinition<T>());

//     auto csrReduce = common::getKernel("csrReduce", {{csr2coo_cl_src}},
//                                        tmpltArgs, compileOpts);

//     // Now we need to sort this into column major
//     kernel::sort0ByKeyIterative<int, int>(rowCopy, index, true);

//     // Now use index to sort values and rows
//     kernel::swapIndex<T>(ovalues, ocolIdx, ivalues, icolIdx.data, index);

//     CL_DEBUG_FINISH(getQueue());

//     cl::NDRange global(irowIdx.info.dims[0], 1, 1);

//     csrReduce(cl::EnqueueArgs(getQueue(), global), *orowIdx.data, *rowCopy.data,
//               M, static_cast<int>(ovalues.info.dims[0]));
//     CL_DEBUG_FINISH(getQueue());
// }
}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire

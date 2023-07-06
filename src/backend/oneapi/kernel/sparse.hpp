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

template<typename T>
class coo2DenseCreateKernel {
public:
    coo2DenseCreateKernel(write_accessor<T> oPtr, const KParam output, write_accessor<T> vPtr, const KParam values, read_accessor<int> rPtr, const KParam rowIdx, read_accessor<int> cPtr, const KParam colIdx) : oPtr_(oPtr), output_(output), vPtr_(vPtr), values_(values), rPtr_(rPtr), rowIdx_(rowIdx), cPtr_(cPtr), colIdx_(colIdx) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

    const int id = g.get_group_id(0) * g.get_local_range(0) * REPEAT + it.get_local_id(0);

    if (id >= values_.dims[0]) return;

    const int dimSize = g.get_local_range(0);

    for (int i = it.get_local_id(0); i < REPEAT * dimSize; i += dimSize) {
        if (i >= values_.dims[0]) return;

        T v   = vPtr_[i];
        int r = rPtr_[i];
        int c = cPtr_[i];

        int offset = r + c * output_.strides[1];

        oPtr_[offset] = v;
    }
}

private:
write_accessor<T> oPtr_;
const KParam output_;
write_accessor<T> vPtr_;
const KParam values_;
read_accessor<int> rPtr_;
const KParam rowIdx_;
read_accessor<int> cPtr_;
const KParam colIdx_;
};

template<typename T>
void coo2dense(Param<T> out, const Param<T> values, const Param<int> rowIdx,
               const Param<int> colIdx) {

    auto local = sycl::range(THREADS_PER_BLOCK, 1);
    auto global = sycl::range(divup(out.info.dims[0], local[0] * REPEAT) * THREADS_PER_BLOCK, 1);

getQueue().submit([&](auto &h) {
sycl::accessor d_rowIdx{*rowIdx.data, h, sycl::read_only};
sycl::accessor d_colIdx{*colIdx.data, h, sycl::read_only};
sycl::accessor d_out{*out.data, h, sycl::write_only, sycl::no_init};
sycl::accessor d_values{*values.data, h, sycl::write_only, sycl::no_init};
h.parallel_for(
  sycl::nd_range{global, local},
  coo2DenseCreateKernel<T>(d_out, out.info, d_values, values.info, d_rowIdx, rowIdx.info, d_colIdx, colIdx.info));
});

    ONEAPI_DEBUG_FINISH(getQueue());
}


template<typename T, int THREADS>
class csr2DenseCreateKernel {
public:
    csr2DenseCreateKernel(write_accessor<T> output, read_accessor<T> values, read_accessor<int> rowidx, read_accessor<int> colidx, const int M) : output_(output), values_(values), rowidx_(rowidx), colidx_(colidx), M_(M) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

    int lid = it.get_local_id(0);
    for (int rowId = g.get_group_id(0); rowId < M_; rowId += it.get_group_range(0)) {
        int colStart = rowidx_[rowId];
        int colEnd   = rowidx_[rowId + 1];
        for (int colId = colStart + lid; colId < colEnd; colId += THREADS) {
            output_[rowId + colidx_[colId] * M_] = values_[colId];
        }
    }
}

private:
write_accessor<T> output_;
read_accessor<T> values_;
read_accessor<int> rowidx_;
read_accessor<int> colidx_;
const int M_;
};


  template<typename T>
void csr2dense(Param<T> output, const Param<T> values, const Param<int> rowIdx,
               const Param<int> colIdx) {
    constexpr int MAX_GROUPS = 4096;
    // FIXME: This needs to be based non nonzeros per row
    constexpr int threads = 64;

    const int M = rowIdx.info.dims[0] - 1;







    auto local = sycl::range(threads, 1);
    int groups_x = std::min((int)(divup(M, local[0])), MAX_GROUPS);
    auto global = sycl::range(local[0] * groups_x, 1);

getQueue().submit([&](auto &h) {
sycl::accessor d_values{*values.data, h, sycl::read_only};
sycl::accessor d_rowIdx{*rowIdx.data, h, sycl::read_only};
sycl::accessor d_colIdx{*colIdx.data, h, sycl::read_only};
sycl::accessor d_output{*output.data, h, sycl::write_only, sycl::no_init};
h.parallel_for(
  sycl::nd_range{global, local},
  csr2DenseCreateKernel<T, threads>(d_output, d_values, d_rowIdx, d_colIdx, M));
});

    ONEAPI_DEBUG_FINISH(getQueue());
}

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
        // sycl::group g = it.get_group();

        int gidx = it.get_global_id(0);
        int gidy = it.get_global_id(1);

        if (gidx >= (unsigned)valinfo_.dims[0]) return;
        if (gidy >= (unsigned)valinfo_.dims[1]) return;

        int rowoff       = rowptr_[gidx];
        T *svalptr_ptr   = svalptr_.get_pointer();
        int *scolptr_ptr = scolptr_.get_pointer();
        svalptr_ptr += rowoff;
        scolptr_ptr += rowoff;

        T *dvalptr_ptr   = dvalptr_.get_pointer();
        int *dcolptr_ptr = dcolptr_.get_pointer();
        dvalptr_ptr += valinfo_.offset;
        dcolptr_ptr += colinfo_.offset;

        T val   = dvalptr_ptr[gidx + gidy * (unsigned)valinfo_.strides[1]];

        if constexpr (std::is_same_v<decltype(val), std::complex<float>> ||
                      std::is_same_v<decltype(val), std::complex<double>>) {
            if (val.real() == 0 && val.imag() == 0) return;
        } else {
            if (val == 0) return;
        }

        int oloc              = dcolptr_ptr[gidx + gidy * colinfo_.strides[1]];
        svalptr_ptr[oloc - 1] = val;
        scolptr_ptr[oloc - 1] = gidy;
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

template <typename T>
void printArray(const char *name, const unsigned N, const Param<T> &thing) {
    auto blah = sycl::host_accessor<T>(*thing.data);
    printf("%s:\n", name);
    for (int i = 0; i < N; i++)
      printf("%f, ", blah[i]);
    printf("\n\n");
}

template <>
void printArray(const char *name, const unsigned N, const Param<int> &thing) {
    auto blah = sycl::host_accessor<int>(*thing.data);
    printf("%s:\n", name);
    for (int i = 0; i < N; i++)
      printf("%d, ", blah[i]);
    printf("\n\n");
}

template<typename T>
void dense2csr(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
               const Param<T> dense) {
    // constexpr bool IsComplex =
    //   std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

    int num_rows = dense.info.dims[0];
    int num_cols = dense.info.dims[1];

    // sd1 contains output of scan along dim 1 of dense
    Array<int> sd1 = createEmptyArray<int>(dim4(num_rows, num_cols));
    // rd1 contains output of nonzero count along dim 1 along dense
    Array<int> rd1 = createEmptyArray<int>(num_rows);

    // scanDim<T, int, af_notzero_t>(sd1, dense, 1, true);
    scan_dim<T, int, af_notzero_t, 1>(sd1, dense, true);
    // reduceDim<T, int, af_notzero_t>(rd1, dense, 0, 0, 1);
    reduce_dim_default<T, int, af_notzero_t, 1>(rd1, dense, 0, 0);
    // scanFirst<int, int, af_add_t>(rowIdx, rd1, false);
    scan_first<int, int, af_add_t>(rowIdx, rd1, false);

    printArray("rowIdx", rowIdx.info.dims[0], rowIdx);

    const int nnz = values.info.dims[0];

    const sycl::id<1> fillOffset(rowIdx.info.offset + (rowIdx.info.dims[0] - 1));
    const sycl::range<1> fillRange(rowIdx.info.dims[0] - fillOffset[0]);
    getQueue().submit([&](auto &h) {
        sycl::accessor d_rowIdx{*rowIdx.data, h, fillRange, fillOffset};
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
                       dense2csrCreateKernel<T>(d_values,
                                                d_colIdx,
                                                d_dense,
                                                dense.info,
                                                d_sdParam,
                                                sdParam.info,
                                                d_rowIdx));
    });

    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T>
class swapIndexCreateKernel {
public:
    swapIndexCreateKernel(write_accessor<T> ovalues, write_accessor<int> oindex, read_accessor<T> ivalues, read_accessor<int> iindex, read_accessor<int> swapIdx, const int nNZ) : ovalues_(ovalues), oindex_(oindex), ivalues_(ivalues), iindex_(iindex), swapIdx_(swapIdx), nNZ_(nNZ) {}
    void operator()(sycl::id<1> it) const {

    // int id = it.get_global_id(0);
    // if (id >= nNZ_) return;

    // int idx = swapIdx_[id];

    // ovalues_[id] = ivalues_[idx];
    // oindex_[id]  = iindex_[idx];
    }

private:
write_accessor<T> ovalues_;
write_accessor<int> oindex_;
read_accessor<T> ivalues_;
read_accessor<int> iindex_;
read_accessor<int> swapIdx_;
const int nNZ_;
};

template<typename T>
void swapIndex(Param<T> ovalues, Param<int> oindex, const Param<T> ivalues,
               sycl::buffer<int> iindex, const Param<int> swapIdx) {
auto global = sycl::range(ovalues.info.dims[0]);

getQueue().submit([&](auto &h) {
    sycl::accessor d_ivalues{*ivalues.data, h, sycl::read_only};
    sycl::accessor d_iindex{iindex, h, sycl::read_only};
    sycl::accessor d_swapIdx{*swapIdx.data, h, sycl::read_only};
    sycl::accessor d_ovalues{*ovalues.data, h, sycl::write_only, sycl::no_init};
    sycl::accessor d_oindex{*oindex.data, h, sycl::write_only, sycl::no_init};
    h.parallel_for(global,
                   swapIndexCreateKernel<T>(
                       d_ovalues, d_oindex, d_ivalues, d_iindex, d_swapIdx,
                       static_cast<int>(ovalues.info.dims[0])));
});

ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T>
class csr2CooCreateKernel {
public:
    csr2CooCreateKernel(write_accessor<int> orowidx, write_accessor<int> ocolidx, read_accessor<int> irowidx, read_accessor<int> icolidx, const int M) : orowidx_(orowidx), ocolidx_(ocolidx), irowidx_(irowidx), icolidx_(icolidx), M_(M) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

    int lid = it.get_local_id(0);
    for (int rowId = g.get_group_id(0); rowId < M_; rowId += it.get_group_range(0)) {
        int colStart = irowidx_[rowId];
        int colEnd   = irowidx_[rowId + 1];
        for (int colId = colStart + lid; colId < colEnd;
             colId += g.get_local_range(0)) {
            orowidx_[colId] = rowId;
            ocolidx_[colId] = icolidx_[colId];
        }
    }
}

private:
write_accessor<int> orowidx_;
write_accessor<int> ocolidx_;
read_accessor<int> irowidx_;
read_accessor<int> icolidx_;
const int M_;
};

template<typename T>
void csr2coo(Param<T> ovalues, Param<int> orowIdx, Param<int> ocolIdx, const Param<T> ivalues,
             const Param<int> irowIdx, const Param<int> icolIdx, Param<int> index) {


    const int MAX_GROUPS = 4096;
    int M                = irowIdx.info.dims[0] - 1;
    // FIXME: This needs to be based non nonzeros per row
    int threads = 64;

    // cl::Buffer *scratch = bufferAlloc(orowIdx.info.dims[0] * sizeof(int));
    auto scratch = memAlloc<int>(orowIdx.info.dims[0]);

    auto local = sycl::range(threads, 1);
    int groups_x = std::min((int)(divup(M, local[0])), MAX_GROUPS);
    auto global = sycl::range(local[0] * groups_x, 1);

    getQueue().submit([&](auto &h) {
        sycl::accessor d_irowIdx{*irowIdx.data, h, sycl::read_only};
        sycl::accessor d_icolIdx{*icolIdx.data, h, sycl::read_only};
        sycl::accessor d_scratch{*scratch, h, sycl::write_only, sycl::no_init};
        sycl::accessor d_ocolIdx{*ocolIdx.data, h, sycl::write_only,
                                 sycl::no_init};
        h.parallel_for(sycl::nd_range{global, local},
                       csr2CooCreateKernel<T>(d_scratch, d_ocolIdx, d_irowIdx,
                                              d_icolIdx, M));
    });

    // Now we need to sort this into column major
    kernel::sort0ByKeyIterative<int, int>(ocolIdx, index, true);

    // Now use index to sort values and rows
    kernel::swapIndex<T>(ovalues, orowIdx, ivalues, *scratch, index);

    ONEAPI_DEBUG_FINISH(getQueue());
}

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

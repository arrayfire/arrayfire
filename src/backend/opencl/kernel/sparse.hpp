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
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/coo2dense.hpp>
#include <kernel_headers/csr2coo.hpp>
#include <kernel_headers/csr2dense.hpp>
#include <kernel_headers/dense2csr.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <type_util.hpp>
#include <map>
#include <mutex>
#include <string>
#include "config.hpp"
#include "reduce.hpp"
#include "scan_dim.hpp"
#include "scan_first.hpp"
#include "sort_by_key.hpp"

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {
namespace kernel {
template <typename T>
void coo2dense(Param out, const Param values, const Param rowIdx,
               const Param colIdx) {
    std::string ref_name = std::string("coo2dense_") +
                           std::string(dtype_traits<T>::getName()) +
                           std::string("_") + std::to_string(REPEAT);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D reps=" << REPEAT;

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        Program prog;
        buildProgram(prog, coo2dense_cl, coo2dense_cl_len, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "coo2dense_kernel");

        addKernelToCache(device, ref_name, entry);
    };

    auto coo2denseOp =
        KernelFunctor<Buffer, const KParam, const Buffer, const KParam,
                      const Buffer, const KParam, const Buffer, const KParam>(
            *entry.ker);

    NDRange local(THREADS_PER_GROUP, 1, 1);

    NDRange global(
        divup(out.info.dims[0], local[0] * REPEAT) * THREADS_PER_GROUP, 1, 1);

    coo2denseOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
                *values.data, values.info, *rowIdx.data, rowIdx.info,
                *colIdx.data, colIdx.info);

    CL_DEBUG_FINISH(getQueue());
}

template <typename T>
void csr2dense(Param output, const Param values, const Param rowIdx,
               const Param colIdx) {
    const int MAX_GROUPS = 4096;
    int M                = rowIdx.info.dims[0] - 1;
    // FIXME: This needs to be based non nonzeros per row
    int threads = 64;

    std::string ref_name = std::string("csr2dense_") +
                           std::string(dtype_traits<T>::getName()) +
                           std::string("_") + std::to_string(threads);

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        options << " -D THREADS=" << threads;

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        const char *ker_strs[] = {csr2dense_cl};
        const int ker_lens[]   = {csr2dense_cl_len};

        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "csr2dense");

        addKernelToCache(device, ref_name, entry);
    }

    NDRange local(threads, 1);
    int groups_x = std::min((int)(divup(M, local[0])), MAX_GROUPS);
    NDRange global(local[0] * groups_x, 1);
    auto csr2dense_kernel = *entry.ker;
    auto csr2dense_func =
        KernelFunctor<Buffer, Buffer, Buffer, Buffer, int>(csr2dense_kernel);

    csr2dense_func(EnqueueArgs(getQueue(), global, local), *output.data,
                   *values.data, *rowIdx.data, *colIdx.data, M);

    CL_DEBUG_FINISH(getQueue());
}

template <typename T>
void dense2csr(Param values, Param rowIdx, Param colIdx, const Param dense) {
    int num_rows = dense.info.dims[0];
    int num_cols = dense.info.dims[1];

    // sd1 contains output of scan along dim 1 of dense
    Array<int> sd1 = createEmptyArray<int>(dim4(num_rows, num_cols));
    // rd1 contains output of nonzero count along dim 1 along dense
    Array<int> rd1 = createEmptyArray<int>(num_rows);

    scan_dim<T, int, af_notzero_t, true>(sd1, dense, 1);
    reduce_dim<T, int, af_notzero_t>(rd1, dense, 0, 0, 1);
    scan_first<int, int, af_add_t, false>(rowIdx, rd1);

    int nnz = values.info.dims[0];
    getQueue().enqueueWriteBuffer(
        *rowIdx.data, CL_TRUE,
        rowIdx.info.offset + (rowIdx.info.dims[0] - 1) * sizeof(int),
        sizeof(int), (void *)&nnz);

    std::string ref_name =
        std::string("dense2csr_") + std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }
        if (std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value) {
            options << " -D IS_CPLX=1";
        } else {
            options << " -D IS_CPLX=0";
        }

        const char *ker_strs[] = {dense2csr_cl};
        const int ker_lens[]   = {dense2csr_cl_len};

        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "dense2csr_split_kernel");

        addKernelToCache(device, ref_name, entry);
    }

    NDRange local(THREADS_X, THREADS_Y);
    int groups_x = divup(dense.info.dims[0], local[0]);
    int groups_y = divup(dense.info.dims[1], local[1]);
    NDRange global(groups_x * local[0], groups_y * local[1]);
    auto dense2csr_split =
        KernelFunctor<Buffer, Buffer, Buffer, KParam, Buffer, KParam, Buffer>(
            *entry.ker);

    dense2csr_split(EnqueueArgs(getQueue(), global, local), *values.data,
                    *colIdx.data, *dense.data, dense.info, *sd1.get(), sd1,
                    *rowIdx.data);

    CL_DEBUG_FINISH(getQueue());
}

template <typename T>
void swapIndex(Param ovalues, Param oindex, const Param ivalues,
               const cl::Buffer *iindex, const Param swapIdx) {
    std::string ref_name = std::string("swapIndex_kernel_") +
                           std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        Program prog;
        buildProgram(prog, csr2coo_cl, csr2coo_cl_len, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "swapIndex_kernel");

        addKernelToCache(device, ref_name, entry);
    };

    auto swapIndexOp = KernelFunctor<Buffer, Buffer, const Buffer, const Buffer,
                                     const Buffer, const int>(*entry.ker);

    NDRange global(ovalues.info.dims[0], 1, 1);

    swapIndexOp(EnqueueArgs(getQueue(), global), *ovalues.data, *oindex.data,
                *ivalues.data, *iindex, *swapIdx.data, ovalues.info.dims[0]);

    CL_DEBUG_FINISH(getQueue());
}

template <typename T>
void csr2coo(Param ovalues, Param orowIdx, Param ocolIdx, const Param ivalues,
             const Param irowIdx, const Param icolIdx, Param index) {
    const int MAX_GROUPS = 4096;
    int M                = irowIdx.info.dims[0] - 1;
    // FIXME: This needs to be based non nonzeros per row
    int threads = 64;

    std::string ref_name =
        std::string("csr2coo_") + std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        const char *ker_strs[] = {csr2coo_cl};
        const int ker_lens[]   = {csr2coo_cl_len};

        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "csr2coo");

        addKernelToCache(device, ref_name, entry);
    }

    cl::Buffer *scratch = bufferAlloc(orowIdx.info.dims[0] * sizeof(int));

    NDRange local(threads, 1);
    int groups_x = std::min((int)(divup(M, local[0])), MAX_GROUPS);
    NDRange global(local[0] * groups_x, 1);
    auto csr2coo_kernel = *entry.ker;
    auto csr2coo_func =
        KernelFunctor<Buffer, Buffer, const Buffer, const Buffer, int>(
            csr2coo_kernel);

    csr2coo_func(EnqueueArgs(getQueue(), global, local), *scratch,
                 *ocolIdx.data, *irowIdx.data, *icolIdx.data, M);

    // Now we need to sort this into column major
    kernel::sort0ByKeyIterative<int, int>(ocolIdx, index, true);

    // Now use index to sort values and rows
    kernel::swapIndex<T>(ovalues, orowIdx, ivalues, scratch, index);

    CL_DEBUG_FINISH(getQueue());

    bufferFree(scratch);
}

template <typename T>
void coo2csr(Param ovalues, Param orowIdx, Param ocolIdx, const Param ivalues,
             const Param irowIdx, const Param icolIdx, Param index,
             Param rowCopy, const int M) {
    // Now we need to sort this into column major
    kernel::sort0ByKeyIterative<int, int>(rowCopy, index, true);

    // Now use index to sort values and rows
    kernel::swapIndex<T>(ovalues, ocolIdx, ivalues, icolIdx.data, index);

    CL_DEBUG_FINISH(getQueue());

    std::string ref_name = std::string("csrReduce_kernel_") +
                           std::string(dtype_traits<T>::getName());

    int device       = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName();

        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        Program prog;
        buildProgram(prog, csr2coo_cl, csr2coo_cl_len, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "csrReduce_kernel");

        addKernelToCache(device, ref_name, entry);
    };

    auto csrReduceOp =
        KernelFunctor<Buffer, const Buffer, const int, const int>(*entry.ker);

    NDRange global(irowIdx.info.dims[0], 1, 1);

    csrReduceOp(EnqueueArgs(getQueue(), global), *orowIdx.data, *rowCopy.data,
                M, ovalues.info.dims[0]);

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl

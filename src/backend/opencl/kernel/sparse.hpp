/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/coo2dense.hpp>
#include <kernel_headers/dense2csr.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <cache.hpp>
#include <type_util.hpp>
#include "scan_dim.hpp"
#include "reduce.hpp"
#include "scan_first.hpp"
#include "config.hpp"

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
    namespace kernel
    {
        template<typename T>
        void coo2dense(Param out, const Param values, const Param rowIdx, const Param colIdx)
        {
            try {

                std::string ref_name =
                    std::string("coo2dense_") +
                    std::string(dtype_traits<T>::getName()) +
                    std::string("_") +
                    std::to_string(REPEAT);

                int device = getActiveDeviceId();
                auto idx = kernelCaches[device].find(ref_name);
                kc_entry_t entry;

                if (idx == kernelCaches[device].end()) {
                    std::ostringstream options;
                    options << " -D T="        << dtype_traits<T>::getName()
                            << " -D reps="     << REPEAT
                            ;

                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    Program prog;
                    buildProgram(prog, coo2dense_cl, coo2dense_cl_len, options.str());
                    entry.prog   = new Program(prog);
                    entry.ker = new Kernel(*entry.prog, "coo2dense_kernel");
                } else {
                    entry = idx->second;
                };

                auto coo2denseOp = KernelFunctor<Buffer, const KParam,
                                           const Buffer, const KParam,
                                           const Buffer, const KParam,
                                           const Buffer, const KParam>
                                          (*entry.ker);

                NDRange local(THREADS_PER_GROUP, 1, 1);

                NDRange global(divup(out.info.dims[0], local[0] * REPEAT) * THREADS_PER_GROUP, 1, 1);

                coo2denseOp(EnqueueArgs(getQueue(), global, local),
                       *out.data, out.info,
                       *values.data, values.info,
                       *rowIdx.data, rowIdx.info,
                       *colIdx.data, colIdx.info);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
            }
        }

        template<typename T>
        void dense2csr(Param values, Param rowIdx, Param colIdx, const Param dense)
        {
            try {
                int num_rows = dense.info.dims[0];
                int num_cols = dense.info.dims[1];
                int dense_elements = num_rows * num_cols;
                Param sd1, rd1, sd0;
                // sd1 contains output of scan along dim 1 of dense
                sd1.data = bufferAlloc(dense_elements * sizeof(int));
                // rd1 contains output of nonzero count along dim 1 along dense
                rd1.data = bufferAlloc(num_rows * sizeof(int));
                // sd0 contains output of exclusive scan rd1
                sd0 = rowIdx;

                sd1.info.offset = 0;
                rd1.info.offset = 0;

                sd1.info.dims[0] = num_rows;
                rd1.info.dims[0] = num_rows;

                sd1.info.dims[1] = num_cols;
                rd1.info.dims[1] = 1;

                sd1.info.dims[2] = 1;
                rd1.info.dims[2] = 1;

                sd1.info.dims[3] = 1;
                rd1.info.dims[3] = 1;

                sd1.info.strides[0] = 1;
                rd1.info.strides[0] = 1;
                for (int i = 1; i < 4; i++) {
                    sd1.info.strides[i] = sd1.info.dims[i - 1] * sd1.info.strides[i - 1];
                    rd1.info.strides[i] = rd1.info.dims[i - 1] * rd1.info.strides[i - 1];
                }

                scan_dim<T, int, af_notzero_t, true>(sd1, dense, 1);
                reduce_dim<T, int, af_notzero_t>(rd1, dense, 0, 0, 1);
                scan_first<int, int, af_add_t, false>(sd0, rd1);

                int nnz = values.info.dims[0];
                getQueue().enqueueWriteBuffer(*sd0.data, CL_TRUE,
                                              sd0.info.offset + (rowIdx.info.dims[0] - 1) * sizeof(int),
                                              sizeof(int),
                                              (void *)&nnz);

                std::string ref_name =
                    std::string("dense2csr_") +
                    std::string(dtype_traits<T>::getName());

                int device = getActiveDeviceId();
                auto idx = kernelCaches[device].find(ref_name);
                kc_entry_t entry;

                if (idx == kernelCaches[device].end()) {

                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName();
                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }
                    if (std::is_same<T, cfloat>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D IS_CPLX=1";
                    } else {
                        options << " -D IS_CPLX=0";
                    }

                    const char *ker_strs[] = {dense2csr_cl};
                    const int   ker_lens[] = {dense2csr_cl_len};

                    Program prog;
                    buildProgram(prog, 1, ker_strs, ker_lens, options.str());
                    entry.prog = new Program(prog);
                    entry.ker  = new Kernel(*entry.prog, "dense2csr_split_kernel");

                    kernelCaches[device][ref_name] = entry;
                } else {
                    entry = idx->second;
                }

                NDRange local(THREADS_X, THREADS_Y);
                int groups_x = divup(dense.info.dims[0], local[0]);
                int groups_y = divup(dense.info.dims[1], local[1]);
                NDRange global(groups_x * local[0], groups_y * local[1]);
                auto dense2csr_split = KernelFunctor<Buffer, Buffer,
                                                     Buffer, KParam,
                                                     Buffer, KParam,
                                                     Buffer>(*entry.ker);

                dense2csr_split(EnqueueArgs(getQueue(), global, local),
                                *values.data, *colIdx.data,
                                *dense.data, dense.info,
                                *sd1.data, sd1.info,
                                *sd0.data);

                CL_DEBUG_FINISH(getQueue());

                bufferFree(rd1.data);
                bufferFree(sd1.data);
            } catch (cl::Error &err) {
                CL_TO_AF_ERROR(err);
            }
        }
    }
}

/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#pragma once
#include <kernel_headers/cscmm.hpp>
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
#include <af/opencl.h>

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
        void cscmm_nn(Param out,
                      const Param &values, const Param &colIdx, const Param &rowIdx,
                      const Param &rhs, const T alpha, const T beta, bool is_conj)
        {
            bool use_alpha = (alpha != scalar<T>(1.0));
            bool use_beta = (beta != scalar<T>(0.0));

            int threads = 256;
            // TODO: Find a better way to tune these parameters
            int rows_per_group = 8;
            int cols_per_group = 8;

            std::string ref_name =
                std::string("cscmm_nn_") +
                std::string(dtype_traits<T>::getName()) +
                std::string("_") +
                std::to_string(use_alpha) +
                std::string("_") +
                std::to_string(use_beta) +
                std::string("_") +
                std::to_string(is_conj) +
                std::string("_") +
                std::to_string(rows_per_group) +
                std::string("_") +
                std::to_string(cols_per_group) +
                std::string("_") +
                std::to_string(threads);

            int device = getActiveDeviceId();

            kc_entry_t entry = kernelCache(device, ref_name);

            if (entry.prog==0 && entry.ker==0) {

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName();
                options << " -D USE_ALPHA=" << use_alpha;
                options << " -D USE_BETA=" << use_beta;
                options << " -D IS_CONJ=" << is_conj;
                options << " -D THREADS=" << threads;
                options << " -D ROWS_PER_GROUP=" << rows_per_group;
                options << " -D COLS_PER_GROUP=" << cols_per_group;

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

                const char *ker_strs[] = {cscmm_cl};
                const int   ker_lens[] = {cscmm_cl_len};

                Program prog;
                buildProgram(prog, 1, ker_strs, ker_lens, options.str());
                entry.prog = new Program(prog);
                entry.ker  = new Kernel(*entry.prog, "cscmm_nn");

                addKernelToCache(device, ref_name, entry);
            }

            auto cscmm_kernel = *entry.ker;
            auto cscmm_func = KernelFunctor<Buffer,
                                            Buffer, Buffer, Buffer,
                                            int, int, int,
                                            Buffer, KParam, T, T>(cscmm_kernel);

            NDRange local(threads, 1);
            int M = out.info.dims[0];
            int N = out.info.dims[1];
            int K = colIdx.info.dims[0] - 1;

            int groups_x = divup(M, rows_per_group);
            int groups_y = divup(N, cols_per_group);
            NDRange global(local[0] * groups_x, local[1] * groups_y);

            cscmm_func(EnqueueArgs(getQueue(), global, local),
                        *out.data, *values.data, *colIdx.data, *rowIdx.data,
                        M, K, N, *rhs.data, rhs.info, alpha, beta);

            CL_DEBUG_FINISH(getQueue());
        }
    }
}

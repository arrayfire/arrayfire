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
#include <kernel_headers/cscmv.hpp>
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
        void cscmv(Param out,
                   const Param &values, const Param &colIdx, const Param &rowIdx,
                   const Param &rhs, const T alpha, const T beta, bool is_conj)
        {
            bool use_alpha = (alpha != scalar<T>(1.0));
            bool use_beta = (beta != scalar<T>(0.0));

            int threads = 256;
            //TODO: rows_per_group limited by register pressure. Find better way to handle this.
            int rows_per_group = 64;

            std::string ref_name =
                std::string("cscmv_") +
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

                const char *ker_strs[] = {cscmv_cl};
                const int   ker_lens[] = {cscmv_cl_len};

                Program prog;
                buildProgram(prog, 1, ker_strs, ker_lens, options.str());
                entry.prog = new Program(prog);
                entry.ker  = new Kernel(*entry.prog, "cscmv_block");

                addKernelToCache(device, ref_name, entry);
            }

            auto cscmv_kernel = *entry.ker;
            auto cscmv_func = KernelFunctor<Buffer,
                                            Buffer, Buffer, Buffer,
                                            int, int,
                                            Buffer, KParam, T, T>(cscmv_kernel);

            NDRange local(threads);
            int K = colIdx.info.dims[0] - 1;
            int M = out.info.dims[0];
            int groups_x = divup(M, rows_per_group);
            NDRange global(local[0] * groups_x, 1);

            cscmv_func(EnqueueArgs(getQueue(), global, local),
                        *out.data, *values.data, *colIdx.data, *rowIdx.data,
                        M, K, *rhs.data, rhs.info, alpha, beta);

            CL_DEBUG_FINISH(getQueue());
        }
    }
}

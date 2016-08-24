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
#include <kernel_headers/csrmv.hpp>
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
        const int MAX_GROUPS_X = 4096 * 4;
        template<typename T>
        void csrmv(Param out,
                   const Param &values, const Param &rowIdx, const Param &colIdx,
                   const Param &rhs, const T alpha, const T beta)
        {
            try {
                bool use_alpha = (alpha != scalar<T>(1.0));
                bool use_beta = (beta != scalar<T>(0.0));
                std::string ref_name =
                    std::string("csrmv_") +
                    std::string(dtype_traits<T>::getName()) +
                    std::string("_") +
                    std::to_string(use_alpha) +
                    std::string("_") +
                    std::to_string(use_beta);

                int device = getActiveDeviceId();
                auto idx = kernelCaches[device].find(ref_name);
                kc_entry_t entry;

                if (idx == kernelCaches[device].end()) {

                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName();
                    options << " -D USE_ALPHA=" << use_alpha;
                    options << " -D USE_BETA=" << use_beta;
                    options << " -D THREADS_PER_GROUP=" << THREADS_PER_GROUP;

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

                    const char *ker_strs[] = {csrmv_cl};
                    const int   ker_lens[] = {csrmv_cl_len};

                    Program prog;
                    buildProgram(prog, 1, ker_strs, ker_lens, options.str());
                    entry.prog = new Program(prog);
                    entry.ker  = new Kernel[2];
                    entry.ker[0] = Kernel(*entry.prog, "csrmv_thread");
                    entry.ker[1] = Kernel(*entry.prog, "csrmv_block");
                } else {
                    entry = idx->second;
                }

                // TODO: Figure out the proper way to choose either csrmv_thread or csrmv_block
                bool is_csrmv_block = true;
                auto csrmv_kernel = is_csrmv_block ? entry.ker[1] : entry.ker[0];
                auto csrmv_func = KernelFunctor<Buffer,
                                                Buffer, Buffer, Buffer,
                                                int,
                                                Buffer, KParam, T, T>(csrmv_kernel);

                NDRange local(THREADS_PER_GROUP, 1);
                int M = rowIdx.info.dims[0] - 1;
                int num_groups = is_csrmv_block ? M : divup(M, local[0]);
                int groups_y = divup(num_groups, MAX_GROUPS_X);
                int groups_x = divup(num_groups, groups_y);
                NDRange global(local[0] * groups_x, local[1] * groups_y);

                csrmv_func(EnqueueArgs(getQueue(), global, local),
                           *out.data, *values.data, *rowIdx.data, *colIdx.data,
                           M, *rhs.data, rhs.info, alpha, beta);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error &ex) {
                CL_TO_AF_ERROR(ex);
            }
        }
    }
}

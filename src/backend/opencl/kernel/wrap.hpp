/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/wrap.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <map>
#include <mutex>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <type_util.hpp>
#include <math.hpp>
#include "config.hpp"
#include <cache.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
    namespace kernel
    {
        template<typename T>
        void wrap(Param out, const Param in,
                  const dim_t wx, const dim_t wy,
                  const dim_t sx, const dim_t sy,
                  const dim_t px, const dim_t py,
                  const bool is_column)
        {
            try {

                std::string ref_name =
                    std::string("wrap_") +
                    std::string(dtype_traits<T>::getName()) +
                    std::string("_") +
                    std::to_string(is_column);

                int device = getActiveDeviceId();
                kc_t::iterator idx = kernelCaches[device].find(ref_name);

                kc_entry_t entry;
                if (idx == kernelCaches[device].end()) {

                    ToNum<T> toNum;
                    std::ostringstream options;
                    options << " -D is_column=" << is_column
                            << " -D ZERO=" << toNum(scalar<T>(0))
                            << " -D T="    << dtype_traits<T>::getName();

                    if (std::is_same<T, double>::value ||
                        std::is_same<T, cdouble>::value) {
                        options << " -D USE_DOUBLE";
                    }

                    Program prog;
                    buildProgram(prog, wrap_cl, wrap_cl_len, options.str());

                    entry.prog = new Program(prog);
                    entry.ker = new Kernel(*entry.prog, "wrap_kernel");

                    kernelCaches[device][ref_name] = entry;
                } else {
                    entry = idx->second;
                }

                dim_t nx = (out.info.dims[0] + 2 * px - wx) / sx + 1;
                dim_t ny = (out.info.dims[1] + 2 * py - wy) / sy + 1;

                NDRange local(THREADS_X, THREADS_Y);

                dim_t groups_x = divup(out.info.dims[0], local[0]);
                dim_t groups_y = divup(out.info.dims[1], local[1]);

                NDRange global(local[0] * groups_x * out.info.dims[2],
                               local[1] * groups_y * out.info.dims[3]);


                auto wrapOp = make_kernel<Buffer, const KParam,
                                          const Buffer, const KParam,
                                          const dim_t, const dim_t,
                                          const dim_t, const dim_t,
                                          const dim_t, const dim_t,
                                          const dim_t, const dim_t,
                                          const dim_t, const dim_t> (*entry.ker);

                wrapOp(EnqueueArgs(getQueue(), global, local),
                       *out.data, out.info, *in.data, in.info,
                       wx, wy, sx, sy, px, py, nx, ny, groups_x, groups_y);

                CL_DEBUG_FINISH(getQueue());

            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}

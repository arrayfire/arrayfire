/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/transpose.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <types.hpp>

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

static const int TILE_DIM  = 32;
static const int THREADS_X = TILE_DIM;
static const int THREADS_Y = 256 / TILE_DIM;

template<typename T, bool conjugate, bool IS32MULTIPLE>
void transpose(Param out, const Param in)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>  trsProgs;
        static std::map<int, Kernel*> trsKernels;

        int device = getActiveDeviceId();

        std::call_once(compileFlags[device], [device] () {

                std::ostringstream options;
                options << " -D TILE_DIM=" << TILE_DIM
                        << " -D THREADS_Y=" << THREADS_Y
                        << " -D IS32MULTIPLE=" << IS32MULTIPLE
                        << " -D DOCONJUGATE=" << (conjugate && af::iscplx<T>())
                        << " -D T=" << dtype_traits<T>::getName();

                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }

                cl::Program prog;
                buildProgram(prog, transpose_cl, transpose_cl_len, options.str());
                trsProgs[device] = new Program(prog);

                trsKernels[device] = new Kernel(*trsProgs[device], "transpose");
            });


        NDRange local(THREADS_X, THREADS_Y);

        int blk_x = divup(in.info.dims[0], TILE_DIM);
        int blk_y = divup(in.info.dims[1], TILE_DIM);

        // launch batch * blk_x blocks along x dimension
        NDRange global(blk_x * local[0] * in.info.dims[2],
                       blk_y * local[1] * in.info.dims[3]);

        auto transposeOp = make_kernel<Buffer, const KParam,
                                       const Buffer, const KParam,
                                       const int, const int> (*trsKernels[device]);

        transposeOp(EnqueueArgs(getQueue(), global, local),
                    *out.data, out.info, *in.data, in.info, blk_x, blk_y);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}

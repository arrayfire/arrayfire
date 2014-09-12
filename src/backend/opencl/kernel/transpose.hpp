#pragma once
#include <kernel_headers/transpose.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

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

static const dim_type TILE_DIM  = 32;
static const dim_type THREADS_X = TILE_DIM;
static const dim_type THREADS_Y = (256 / TILE_DIM);

template<typename T>
void transpose(Param out, const Param in)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static Program            trsProgs[DeviceManager::MAX_DEVICES];
        static Kernel           trsKernels[DeviceManager::MAX_DEVICES];

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D TILE_DIM=" << TILE_DIM;

                buildProgram(trsProgs[device],
                             transpose_cl,
                             transpose_cl_len,
                             options.str());

                trsKernels[device] = Kernel(trsProgs[device], "transpose");
            });


        NDRange local(THREADS_X, THREADS_Y);

        dim_type blk_x = divup(in.info.dims[0], TILE_DIM);
        dim_type blk_y = divup(in.info.dims[1], TILE_DIM);

        // launch batch * blk_x blocks along x dimension
        NDRange global(blk_x * TILE_DIM * in.info.dims[2], blk_y * TILE_DIM);

        auto transposeOp = make_kernel<Buffer, KParam,
                                       Buffer, KParam,
                                       dim_type> (trsKernels[device]);


        transposeOp(EnqueueArgs(getQueue(), global, local),
                    out.data, out.info, in.data, in.info, blk_x);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        SHOW_CL_ERROR(err);
        throw;
    }
}

}

}

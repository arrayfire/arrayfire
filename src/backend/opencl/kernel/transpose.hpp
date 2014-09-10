#pragma once
#include <kernel_headers/transpose.hpp>
#include <cl.hpp>
#include <platform.hpp>
#include <traits.hpp>
#include <sstream>
#include <string>
#include <mutex>
#include <dispatch.hpp>

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
static const dim_type THREADS_Y = TILE_DIM/4;

template<typename T>
void transpose( Buffer &out, const Buffer &in, const dim_type ndims, const dim_type * const dims,
                const dim_type * const strides, const dim_type offset)
{
    static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
    static Program            trsProgs[DeviceManager::MAX_DEVICES];
    static Kernel           trsKernels[DeviceManager::MAX_DEVICES];

    int device = getActiveDeviceId();

    std::call_once( compileFlags[device], [device] () {
                Program::Sources setSrc;
                setSrc.emplace_back(transpose_cl, transpose_cl_len);

                trsProgs[device] = Program(getContext(), setSrc);

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                    << " -D dim_type=" << dtype_traits<dim_type>::getName()
                    << " -D TILE_DIM=" << TILE_DIM;
                trsProgs[device].build(options.str().c_str());

                trsKernels[device] = Kernel(trsProgs[device], "transpose");
            });

    auto transposeOp = make_kernel< Buffer, Buffer,
                                    dim_type, dim_type,
                                    dim_type, dim_type,
                                    dim_type, dim_type > (trsKernels[device]);

    NDRange local(THREADS_X,THREADS_Y);

    dim_type blk_x = divup(dims[0],TILE_DIM);
    dim_type blk_y = divup(dims[1],TILE_DIM);
    // launch batch * blk_x blocks along x dimension
    NDRange global( blk_x*TILE_DIM*dims[2],
            blk_y*TILE_DIM);

    transposeOp(EnqueueArgs(getQueue(), global, local),
                out, in, dims[0], dims[1],
                strides[1], strides[2], offset, blk_x);
}

}
}

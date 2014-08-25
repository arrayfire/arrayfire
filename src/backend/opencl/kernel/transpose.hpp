#pragma once

#include <kernel_headers/transpose.hpp>
#define __CL_ENABLE_EXCEPTIONS
#include <cl.hpp>
#include <ctx.hpp>
#include <traits.hpp>
#include <sstream>
#include <string>

using cl::Buffer;
using cl::Program;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

#define divup(a, b) ((a)+(b)-1)/(b)

namespace opencl
{
namespace kernel
{

static const dim_type TILE_DIM  = 32;
static const dim_type THREADS_X = TILE_DIM;
static const dim_type THREADS_Y = TILE_DIM/4;

template<typename T>
void transpose( Buffer &out, const Buffer &in, const dim_type ndims, const dim_type * const dims,
                const dim_type * const offsets, const dim_type * const strides)
{
    Program::Sources setSrc;
    setSrc.emplace_back(transpose_cl, transpose_cl_len);
    Program prog(getCtx(0), setSrc);

    std::ostringstream options;
    options << " -D T=" << dtype_traits<T>::getName()
        << " -D dim_type=" << dtype_traits<dim_type>::getName()
        << " -D TILE_DIM=" << TILE_DIM;
    prog.build(options.str().c_str());

    auto transposeOp = make_kernel< Buffer, Buffer,
         dim_type, dim_type,
         dim_type, dim_type,
         dim_type, dim_type,
         dim_type >
             (prog, "transpose");

    NDRange local(THREADS_X,THREADS_Y);

    dim_type blk_x = divup(dims[0],TILE_DIM);
    dim_type blk_y = divup(dims[1],TILE_DIM);
    // launch batch * blk_x blocks along x dimension
    NDRange global( blk_x*TILE_DIM*dims[2],
            blk_y*TILE_DIM);

    transposeOp(    EnqueueArgs(getQueue(0), global, local),
            out, in, dims[0], dims[1],
            offsets[0], offsets[1],
            strides[0], strides[1], blk_x);
}

}
}

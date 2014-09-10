#include <kernel_headers/morph.hpp>
#include <cl.hpp>
#include <platform.hpp>
#include <traits.hpp>
#include <sstream>
#include <string>
#include <dispatch.hpp>

using cl::Buffer;
using cl::Program;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::LocalSpaceArg;
using cl::NDRange;
using std::string;

namespace opencl
{

namespace kernel
{

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

static const dim_type CUBE_X    =  8;
static const dim_type CUBE_Y    =  8;
static const dim_type CUBE_Z    =  8;

// FIXME: This struct declaration should stay in
//        sync with the struct defined inside morph.cl
typedef struct Params {
    cl_long      offset;
    cl_long     dims[4];
    cl_long istrides[4];
    cl_long ostrides[4];
} MorphParams;

template<typename T, bool isDilation, dim_type windLen>
void morph(Buffer         &out,
        const Buffer      &in,
        const Buffer      &mask,
        const MorphParams &params)
{
    Program::Sources setSrc;
    setSrc.emplace_back(morph_cl, morph_cl_len);
    Program prog(getContext(), setSrc);

    std::ostringstream options;
    options << " -D T=" << dtype_traits<T>::getName()
        << " -D dim_type=" << dtype_traits<dim_type>::getName()
        << " -D isDilation="<< isDilation
        << " -D windLen=" << windLen;
    prog.build(options.str().c_str());

    auto morphOp = make_kernel< Buffer, Buffer,
                                Buffer, cl::LocalSpaceArg,
                                Buffer, dim_type
                              > (prog, "morph");

    NDRange local(THREADS_X, THREADS_Y);

    dim_type blk_x = divup(params.dims[0], THREADS_X);
    dim_type blk_y = divup(params.dims[1], THREADS_Y);
    // launch batch * blk_x blocks along x dimension
    NDRange global(blk_x * THREADS_X * params.dims[2],
                   blk_y * THREADS_Y);

    // copy mask/filter to constant memory
    cl_int se_size   = sizeof(T)*windLen*windLen;
    cl::Buffer mBuff = cl::Buffer(getContext(), CL_MEM_READ_ONLY, se_size);
    getQueue().enqueueCopyBuffer(mask, mBuff, 0, 0, se_size);

    // copy params struct to opencl buffer
    cl::Buffer pBuff = cl::Buffer(getContext(), CL_MEM_READ_ONLY, sizeof(kernel::MorphParams));
    getQueue().enqueueWriteBuffer(pBuff, CL_TRUE, 0, sizeof(kernel::MorphParams), &params);

    // calculate shared memory size
    const int halo    = windLen/2;
    const int padding = 2*halo;
    const int locLen  = THREADS_X + padding + 1;
    const int locSize = locLen * (THREADS_Y+padding);

    morphOp(EnqueueArgs(getQueue(), global, local),
            out, in, mBuff,
            cl::Local(locSize*sizeof(T)),
            pBuff, blk_x);
}

template<typename T, bool isDilation, dim_type windLen>
void morph3d(Buffer       &out,
        const Buffer      &in,
        const Buffer      &mask,
        const MorphParams &params)
{
    Program::Sources setSrc;
    setSrc.emplace_back(morph_cl, morph_cl_len);
    Program prog(getContext(), setSrc);

    std::ostringstream options;
    options << " -D T=" << dtype_traits<T>::getName()
        << " -D dim_type=" << dtype_traits<dim_type>::getName()
        << " -D isDilation="<< isDilation
        << " -D windLen=" << windLen;
    prog.build(options.str().c_str());

    auto morphOp = make_kernel< Buffer, Buffer,
                                Buffer, cl::LocalSpaceArg,
                                Buffer
                              > (prog, "morph3d");

    NDRange local(CUBE_X, CUBE_Y, CUBE_Z);

    dim_type blk_x = divup(params.dims[0], CUBE_X);
    dim_type blk_y = divup(params.dims[1], CUBE_Y);
    dim_type blk_z = divup(params.dims[2], CUBE_Z);
    // launch batch * blk_x blocks along x dimension
    NDRange global(blk_x * CUBE_X,
                   blk_y * CUBE_Y,
                   blk_z * CUBE_Z);

    // copy mask/filter to constant memory
    cl_int se_size   = sizeof(T)*windLen*windLen*windLen;
    cl::Buffer mBuff = cl::Buffer(getContext(), CL_MEM_READ_ONLY, se_size);
    getQueue().enqueueCopyBuffer(mask, mBuff, 0, 0, se_size);

    // copy params struct to opencl buffer
    cl::Buffer pBuff = cl::Buffer(getContext(), CL_MEM_READ_ONLY, sizeof(kernel::MorphParams));
    getQueue().enqueueWriteBuffer(pBuff, CL_TRUE, 0, sizeof(kernel::MorphParams), &params);

    // calculate shared memory size
    const int halo    = windLen/2;
    const int padding = 2*halo;
    const int locLen  = CUBE_X+padding+1;
    const int locArea = locLen *(CUBE_Y+padding);
    const int locSize = locArea*(CUBE_Z+padding);

    morphOp(EnqueueArgs(getQueue(), global, local),
            out, in, mBuff, cl::Local(locSize*sizeof(T)), pBuff);
}

}

}

#include <kernel_headers/bilateral.hpp>
#include <cl.hpp>
#include <ctx.hpp>
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

// FIXME: This struct declaration should stay in
//        sync with the struct defined inside morph.cl
struct KernelParams {
    cl_long offset;
    cl_long idims[4];
    cl_long istrides[4];
    cl_long ostrides[4];
};

template<typename T, bool isColor>
void bilateral(Buffer &out, const Buffer &in,
        const KernelParams &params, float s_sigma, float c_sigma)
{
    Program::Sources setSrc;
    setSrc.emplace_back(bilateral_cl, bilateral_cl_len);
    Program prog(getCtx(0), setSrc);

    std::ostringstream options;
    options << " -D T=" << dtype_traits<T>::getName()
        << " -D dim_type=" << dtype_traits<dim_type>::getName();
    prog.build(options.str().c_str());

    auto bilateralOp = make_kernel< Buffer, Buffer,
                                cl::LocalSpaceArg,
                                cl::LocalSpaceArg,
                                Buffer, float, float,
                                dim_type, dim_type
                              > (prog, "bilateral");

    NDRange local(THREADS_X, THREADS_Y);

    dim_type blk_x = divup(params.idims[0], THREADS_X);
    dim_type blk_y = divup(params.idims[1], THREADS_Y);

    dim_type bCount= blk_x * params.idims[2];
    if (isColor)
        bCount *= params.idims[3];

    NDRange global(bCount*THREADS_X, blk_y*THREADS_Y);

    // copy params struct to opencl buffer
    cl::Buffer pBuff = cl::Buffer(getCtx(0), CL_MEM_READ_ONLY, sizeof(kernel::KernelParams));
    getQueue(0).enqueueWriteBuffer(pBuff, CL_TRUE, 0, sizeof(kernel::KernelParams), &params);

    // calculate local memory size
    dim_type radius = (dim_type)std::max(s_sigma * 1.5f, 1.f);
    dim_type num_shrd_elems    = (THREADS_X + 2 * radius) * (THREADS_Y + 2 * radius);
    dim_type num_gauss_elems   = (2*radius+1)*(2*radius+1);

    bilateralOp(EnqueueArgs(getQueue(0), global, local),
            out, in, cl::Local(num_shrd_elems*sizeof(T)),
            cl::Local(num_gauss_elems*sizeof(T)),
            pBuff, s_sigma, c_sigma, num_shrd_elems, blk_x);
}

}

}

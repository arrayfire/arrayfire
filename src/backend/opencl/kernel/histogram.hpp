#include <kernel_headers/histogram.hpp>
#include <cl.hpp>
#include <ctx.hpp>
#include <traits.hpp>
#include <sstream>
#include <string>

#define divup(a, b) ((a)+(b)-1)/(b)

namespace opencl
{

namespace kernel
{

static const unsigned MAX_BINS  = 4000;
static const dim_type THREADS_X =  256;
static const dim_type THRD_LOAD =   16;

// FIXME: This struct declaration should stay in
//        sync with the struct defined inside morph.cl
typedef struct Params {
    cl_long      offset;
    cl_long    idims[4];
    cl_long istrides[4];
    cl_long ostrides[4];
} HistParams;

template<typename inType, typename outType>
void histogram(Buffer &out, const Buffer &in, const Buffer &minmax,
              const HistParams &params, dim_type nbins)
{
    Program::Sources setSrc;
    setSrc.emplace_back(histogram_cl, histogram_cl_len);
    Program prog(getCtx(0), setSrc);

    std::ostringstream options;
    options << " -D inType=" << dtype_traits<inType>::getName()
        << " -D outType=" << dtype_traits<outType>::getName()
        << " -D dim_type=" << dtype_traits<dim_type>::getName()
        << " -D THRD_LOAD=" << THRD_LOAD;
    prog.build(options.str().c_str());

    auto histogramOp = make_kernel< Buffer, Buffer,
                                Buffer, Buffer,
                                cl::LocalSpaceArg,
                                dim_type, dim_type, dim_type
                              > (prog, "histogram");

    NDRange local(THREADS_X, 1);

    dim_type numElements = params.idims[0]*params.idims[1];

    dim_type blk_x       = divup(numElements, THRD_LOAD*THREADS_X);

    dim_type batchCount  = params.idims[2];

    NDRange global(blk_x*THREADS_X, batchCount);

    // copy params struct to opencl buffer
    cl::Buffer pBuff = cl::Buffer(getCtx(0), CL_MEM_READ_ONLY, sizeof(kernel::HistParams));
    getQueue(0).enqueueWriteBuffer(pBuff, CL_TRUE, 0, sizeof(kernel::HistParams), &params);

    dim_type locSize = nbins * sizeof(outType);

    histogramOp(EnqueueArgs(getQueue(0), global, local),
            out, in, minmax, pBuff, cl::Local(locSize),
            numElements, nbins, blk_x);
}

}

}

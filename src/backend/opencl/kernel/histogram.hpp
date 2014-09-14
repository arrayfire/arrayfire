#pragma once
#include <kernel_headers/histogram.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

using cl::Kernel;

namespace opencl
{

namespace kernel
{

static const unsigned MAX_BINS  = 4000;
static const dim_type THREADS_X =  256;
static const dim_type THRD_LOAD =   16;

template<typename inType, typename outType>
void histogram(Param out, const Param in, const Param minmax, dim_type nbins)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static Program            histProgs[DeviceManager::MAX_DEVICES];
        static Kernel           histKernels[DeviceManager::MAX_DEVICES];

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D inType=" << dtype_traits<inType>::getName()
                            << " -D outType=" << dtype_traits<outType>::getName()
                            << " -D THRD_LOAD=" << THRD_LOAD;

                    buildProgram(histProgs[device], histogram_cl, histogram_cl_len, options.str());

                    histKernels[device] = Kernel(histProgs[device], "histogram");
                });

        auto histogramOp = make_kernel<Buffer, KParam, Buffer, KParam,
                                       Buffer, cl::LocalSpaceArg,
                                       dim_type, dim_type, dim_type
                                      >(histKernels[device]);

        NDRange local(THREADS_X, 1);

        dim_type numElements = in.info.dims[0]*in.info.dims[1];

        dim_type blk_x       = divup(numElements, THRD_LOAD*THREADS_X);

        dim_type batchCount  = in.info.dims[2];

        NDRange global(blk_x*THREADS_X, batchCount);

        dim_type locSize = nbins * sizeof(outType);

        histogramOp(EnqueueArgs(getQueue(), global, local),
                out.data, out.info, in.data, in.info, minmax.data,
                cl::Local(locSize), numElements, nbins, blk_x);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}

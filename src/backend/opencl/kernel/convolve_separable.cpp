/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel_headers/convolve_separable.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <map>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <memory.hpp>
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

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename T, typename accType, int conv_dim, bool expand>
void convSep(Param out, const Param signal, const Param filter)
{
    try {

        const int fLen = filter.info.dims[0] * filter.info.dims[1];

        std::string ref_name =
            std::string("convsep_") +
            std::to_string(conv_dim) +
            std::string("_") +
            std::string(dtype_traits<T>::getName()) +
            std::string("_") +
            std::string(dtype_traits<accType>::getName()) +
            std::string("_") +
            std::to_string(expand) +
            std::string("_") +
            std::to_string(fLen);

        int device = getActiveDeviceId();
        kc_t::iterator idx = kernelCaches[device].find(ref_name);

        kc_entry_t entry;
        if (idx == kernelCaches[device].end()) {
            const size_t C0_SIZE  = (THREADS_X+2*(fLen-1))* THREADS_Y;
            const size_t C1_SIZE  = (THREADS_Y+2*(fLen-1))* THREADS_X;

            size_t locSize = (conv_dim==0 ? C0_SIZE : C1_SIZE);

            std::ostringstream options;
            options << " -D T=" << dtype_traits<T>::getName()
                    << " -D accType="<< dtype_traits<accType>::getName()
                    << " -D CONV_DIM="<< conv_dim
                    << " -D EXPAND="<< expand
                    << " -D FLEN="<< fLen
                    << " -D LOCAL_MEM_SIZE="<<locSize;
            if (std::is_same<T, double>::value ||
                std::is_same<T, cdouble>::value) {
                options << " -D USE_DOUBLE";
            }
            Program prog;
            buildProgram(prog, convolve_separable_cl, convolve_separable_cl_len, options.str());

            entry.prog   = new Program(prog);
            entry.ker  = new Kernel(*entry.prog, "convolve");
            kernelCaches[device][ref_name] = entry;
        } else {
            entry = idx->second;
        }

        auto convOp = make_kernel<Buffer, KParam, Buffer, KParam, Buffer,
                                  int, int>(*entry.ker);

        NDRange local(THREADS_X, THREADS_Y);

        int blk_x = divup(out.info.dims[0], THREADS_X);
        int blk_y = divup(out.info.dims[1], THREADS_Y);

        NDRange global(blk_x*signal.info.dims[2]*THREADS_X,
                       blk_y*signal.info.dims[3]*THREADS_Y);

        cl::Buffer *mBuff = bufferAlloc(fLen*sizeof(accType));
        // FIX ME: if the filter array is strided, direct might cause issues
        getQueue().enqueueCopyBuffer(*filter.data, *mBuff, 0, 0, fLen*sizeof(accType));

        convOp(EnqueueArgs(getQueue(), global, local),
               *out.data, out.info, *signal.data, signal.info, *mBuff, blk_x, blk_y);

        bufferFree(mBuff);
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

#define INSTANTIATE(T, accT)  \
    template void convSep<T, accT, 0, true >(Param out, const Param sig, const Param filt); \
    template void convSep<T, accT, 1, true >(Param out, const Param sig, const Param filt); \
    template void convSep<T, accT, 0, false>(Param out, const Param sig, const Param filt); \
    template void convSep<T, accT, 1, false>(Param out, const Param sig, const Param filt);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)

}

}

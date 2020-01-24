/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel_headers/convolve_separable.hpp>
#include <kernel_headers/ops.hpp>

#include <Param.hpp>
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <debug_opencl.hpp>
#include <kernel/names.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <map>
#include <mutex>
#include <string>

using cl::Buffer;
using cl::EnqueueArgs;
using cl::Kernel;
using cl::KernelFunctor;
using cl::NDRange;
using cl::Program;
using std::string;

namespace opencl {

namespace kernel {

static const int THREADS_X = 16;
static const int THREADS_Y = 16;

template<typename T, typename accType, int conv_dim, bool expand>
void convSep(Param out, const Param signal, const Param filter) {
    const int fLen = filter.info.dims[0] * filter.info.dims[1];

    std::string ref_name =
        std::string("convsep_") + std::to_string(conv_dim) + std::string("_") +
        std::string(dtype_traits<T>::getName()) + std::string("_") +
        std::string(dtype_traits<accType>::getName()) + std::string("_") +
        std::to_string(expand) + std::string("_") + std::to_string(fLen);

    int device = getActiveDeviceId();

    kc_entry_t entry = kernelCache(device, ref_name);

    if (entry.prog == 0 && entry.ker == 0) {
        const size_t C0_SIZE = (THREADS_X + 2 * (fLen - 1)) * THREADS_Y;
        const size_t C1_SIZE = (THREADS_Y + 2 * (fLen - 1)) * THREADS_X;

        size_t locSize = (conv_dim == 0 ? C0_SIZE : C1_SIZE);

        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D Ti=" << dtype_traits<T>::getName()
                << " -D To=" << dtype_traits<accType>::getName()
                << " -D accType=" << dtype_traits<accType>::getName()
                << " -D CONV_DIM=" << conv_dim << " -D EXPAND=" << expand
                << " -D FLEN=" << fLen << " -D LOCAL_MEM_SIZE=" << locSize
                << " -D " << binOpName<af_mul_t>();

        if ((af_dtype)dtype_traits<T>::af_type == c32 ||
            (af_dtype)dtype_traits<T>::af_type == c64) {
            options << " -D CPLX=1";
        } else {
            options << " -D CPLX=0";
        }
        if (std::is_same<T, double>::value || std::is_same<T, cdouble>::value) {
            options << " -D USE_DOUBLE";
        }

        const char *ker_strs[] = {ops_cl, convolve_separable_cl};
        const int ker_lens[]   = {ops_cl_len, convolve_separable_cl_len};
        Program prog;
        buildProgram(prog, 2, ker_strs, ker_lens, options.str());

        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "convolve");

        addKernelToCache(device, ref_name, entry);
    }

    auto convOp =
        KernelFunctor<Buffer, KParam, Buffer, KParam, Buffer, int, int>(
            *entry.ker);

    NDRange local(THREADS_X, THREADS_Y);

    int blk_x = divup(out.info.dims[0], THREADS_X);
    int blk_y = divup(out.info.dims[1], THREADS_Y);

    NDRange global(blk_x * signal.info.dims[2] * THREADS_X,
                   blk_y * signal.info.dims[3] * THREADS_Y);

    cl::Buffer *mBuff = bufferAlloc(fLen * sizeof(accType));
    // FIX ME: if the filter array is strided, direct might cause issues
    getQueue().enqueueCopyBuffer(*filter.data, *mBuff, 0, 0,
                                 fLen * sizeof(accType));

    convOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
           *signal.data, signal.info, *mBuff, blk_x, blk_y);

    bufferFree(mBuff);
}

#define INSTANTIATE(T, accT)                                             \
    template void convSep<T, accT, 0, true>(Param out, const Param sig,  \
                                            const Param filt);           \
    template void convSep<T, accT, 1, true>(Param out, const Param sig,  \
                                            const Param filt);           \
    template void convSep<T, accT, 0, false>(Param out, const Param sig, \
                                             const Param filt);          \
    template void convSep<T, accT, 1, false>(Param out, const Param sig, \
                                             const Param filt);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat, cfloat)
INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(uint, float)
INSTANTIATE(int, float)
INSTANTIATE(uchar, float)
INSTANTIATE(char, float)
INSTANTIATE(ushort, float)
INSTANTIATE(short, float)
INSTANTIATE(uintl, float)
INSTANTIATE(intl, float)

}  // namespace kernel

}  // namespace opencl

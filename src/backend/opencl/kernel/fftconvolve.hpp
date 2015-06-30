/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <backend.hpp>
#include <program.hpp>
#include <dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/fftconvolve_pack.hpp>
#include <kernel_headers/fftconvolve_multiply.hpp>
#include <kernel_headers/fftconvolve_reorder.hpp>
#include <memory.hpp>
#include <map>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::EnqueueArgs;
using cl::LocalSpaceArg;
using cl::NDRange;

namespace opencl
{

namespace kernel
{

static const int THREADS = 256;

void calcParamSizes(Param& sig_tmp,
                    Param& filter_tmp,
                    Param& packed,
                    Param& sig,
                    Param& filter,
                    const int baseDim,
                    ConvolveBatchKind kind)
{
    sig_tmp.info.dims[0] = filter_tmp.info.dims[0] = packed.info.dims[0];
    sig_tmp.info.strides[0] = filter_tmp.info.strides[0] = 1;

    for (int k = 1; k < 4; k++) {
        if (k < baseDim) {
            sig_tmp.info.dims[k]    = packed.info.dims[k];
            filter_tmp.info.dims[k] = packed.info.dims[k];
        }
        else {
            sig_tmp.info.dims[k]    = sig.info.dims[k];
            filter_tmp.info.dims[k] = filter.info.dims[k];
        }

        sig_tmp.info.strides[k]    = sig_tmp.info.strides[k - 1] * sig_tmp.info.dims[k - 1];
        filter_tmp.info.strides[k] = filter_tmp.info.strides[k - 1] * filter_tmp.info.dims[k - 1];
    }

    // Calculate memory offsets for packed signal and filter
    sig_tmp.data = packed.data;
    filter_tmp.data = packed.data;

    if (kind == CONVOLVE_BATCH_KERNEL) {
        filter_tmp.info.offset = 0;
        sig_tmp.info.offset = filter_tmp.info.strides[3] * filter_tmp.info.dims[3] * 2;
    }
    else {
        sig_tmp.info.offset = 0;
        filter_tmp.info.offset = sig_tmp.info.strides[3] * sig_tmp.info.dims[3] * 2;
    }
}

template<typename convT, typename T, bool isDouble, typename printT>
void packDataHelper(Param packed,
                    Param sig,
                    Param filter,
                    const int baseDim,
                    ConvolveBatchKind kind)
{
    try {
        static std::once_flag     compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*>  fftconvolveProgs;
        static std::map<int, Kernel*>   pdKernel;
        static std::map<int, Kernel*>   paKernel;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName();

                if ((af_dtype) dtype_traits<convT>::af_type == c32) {
                    options << " -D CONVT=float";
                }
                else if ((af_dtype) dtype_traits<convT>::af_type == c64 && isDouble) {
                    options << " -D CONVT=double"
                            << " -D USE_DOUBLE";
                }

                cl::Program prog;
                buildProgram(prog, fftconvolve_pack_cl, fftconvolve_pack_cl_len, options.str());
                fftconvolveProgs[device] = new Program(prog);

                pdKernel[device] = new Kernel(*fftconvolveProgs[device], "pack_data");
                paKernel[device] = new Kernel(*fftconvolveProgs[device], "pad_array");
            });

        Param sig_tmp, filter_tmp;
        calcParamSizes(sig_tmp, filter_tmp, packed, sig, filter, baseDim, kind);

        int sig_packed_elem = sig_tmp.info.strides[3] * sig_tmp.info.dims[3];
        int filter_packed_elem = filter_tmp.info.strides[3] * filter_tmp.info.dims[3];

        // Number of packed complex elements in dimension 0
        int sig_half_d0 = divup(sig.info.dims[0], 2);
        int sig_half_d0_odd = sig.info.dims[0] % 2;

        int blocks = divup(sig_packed_elem, THREADS);

        // Locate features kernel sizes
        NDRange local(THREADS);
        NDRange global(blocks * THREADS);

        // Pack signal in a complex matrix where first dimension is half the input
        // (allows faster FFT computation) and pad array to a power of 2 with 0s
        auto pdOp = make_kernel<Buffer, KParam,
                                Buffer, KParam,
                                const int, const int> (*pdKernel[device]);

        pdOp(EnqueueArgs(getQueue(), global, local),
             *sig_tmp.data, sig_tmp.info, *sig.data, sig.info,
             sig_half_d0, sig_half_d0_odd);
        CL_DEBUG_FINISH(getQueue());

        blocks = divup(filter_packed_elem, THREADS);
        global = NDRange(blocks * THREADS);

        // Pad filter array with 0s
        auto paOp = make_kernel<Buffer, KParam,
                                Buffer, KParam> (*paKernel[device]);

        paOp(EnqueueArgs(getQueue(), global, local),
             *filter_tmp.data, filter_tmp.info,
             *filter.data, filter.info);
        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

template<typename convT, typename T, bool isDouble, typename printT>
void complexMultiplyHelper(Param packed,
                           Param sig,
                           Param filter,
                           const int baseDim,
                           ConvolveBatchKind kind)
{
    try {
        static std::once_flag     compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> fftconvolveProgs;
        static std::map<int, Kernel*>  cmKernel;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D CONVOLVE_BATCH_NONE=" << (int)CONVOLVE_BATCH_NONE
                        << " -D CONVOLVE_BATCH_SIGNAL=" << (int)CONVOLVE_BATCH_SIGNAL
                        << " -D CONVOLVE_BATCH_KERNEL=" << (int)CONVOLVE_BATCH_KERNEL
                        << " -D CONVOLVE_BATCH_SAME=" << (int)CONVOLVE_BATCH_SAME;

                if ((af_dtype) dtype_traits<convT>::af_type == c32) {
                    options << " -D CONVT=float";
                }
                else if ((af_dtype) dtype_traits<convT>::af_type == c64 && isDouble) {
                    options << " -D CONVT=double"
                            << " -D USE_DOUBLE";
                }

                cl::Program prog;
                buildProgram(prog,
                             fftconvolve_multiply_cl,
                             fftconvolve_multiply_cl_len,
                             options.str());
                fftconvolveProgs[device] = new Program(prog);

                cmKernel[device] = new Kernel(*fftconvolveProgs[device], "complex_multiply");
            });

        Param sig_tmp, filter_tmp;
        calcParamSizes(sig_tmp, filter_tmp, packed, sig, filter, baseDim, kind);

        int sig_packed_elem = sig_tmp.info.strides[3] * sig_tmp.info.dims[3];
        int filter_packed_elem = filter_tmp.info.strides[3] * filter_tmp.info.dims[3];
        int mul_elem = (sig_packed_elem < filter_packed_elem) ?
                            filter_packed_elem : sig_packed_elem;

        int blocks = divup(mul_elem, THREADS);

        NDRange local(THREADS);
        NDRange global(blocks * THREADS);

        // Multiply filter and signal FFT arrays
        auto cmOp = make_kernel<Buffer, KParam,
                                Buffer, KParam,
                                Buffer, KParam,
                                const int, const int> (*cmKernel[device]);

        cmOp(EnqueueArgs(getQueue(), global, local),
             *packed.data, packed.info,
             *sig_tmp.data, sig_tmp.info,
             *filter_tmp.data, filter_tmp.info,
             mul_elem, (int)kind);
        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

template<typename T, typename convT, bool isDouble, bool roundOut, bool expand, typename printT>
void reorderOutputHelper(Param out,
                         Param packed,
                         Param sig,
                         Param filter,
                         const int baseDim,
                         ConvolveBatchKind kind)
{
    try {
        static std::once_flag     compileFlags[DeviceManager::MAX_DEVICES];
        static std::map<int, Program*> fftconvolveProgs;
        static std::map<int, Kernel*>  roKernel;

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D ROUND_OUT=" << (int)roundOut
                        << " -D EXPAND=" << (int)expand;

                if ((af_dtype) dtype_traits<convT>::af_type == c32) {
                    options << " -D CONVT=float";
                }
                else if ((af_dtype) dtype_traits<convT>::af_type == c64 && isDouble) {
                    options << " -D CONVT=double"
                            << " -D USE_DOUBLE";
                }

                cl::Program prog;
                buildProgram(prog,
                             fftconvolve_reorder_cl,
                             fftconvolve_reorder_cl_len,
                             options.str());
                fftconvolveProgs[device] = new Program(prog);

                roKernel[device] = new Kernel(*fftconvolveProgs[device], "reorder_output");
            });

        Param sig_tmp, filter_tmp;
        calcParamSizes(sig_tmp, filter_tmp, packed, sig, filter, baseDim, kind);

        // Number of packed complex elements in dimension 0
        int sig_half_d0 = divup(sig.info.dims[0], 2);

        int blocks = divup(out.info.strides[3] * out.info.dims[3], THREADS);

        NDRange local(THREADS);
        NDRange global(blocks * THREADS);

        auto roOp = make_kernel<Buffer, KParam,
                                Buffer, KParam,
                                KParam, const int,
                                const int> (*roKernel[device]);

        if (kind == CONVOLVE_BATCH_KERNEL) {
            roOp(EnqueueArgs(getQueue(), global, local),
                 *out.data, out.info,
                 *filter_tmp.data, filter_tmp.info,
                 filter.info, sig_half_d0, baseDim);
        }
        else {
            roOp(EnqueueArgs(getQueue(), global, local),
                 *out.data, out.info,
                 *sig_tmp.data, sig_tmp.info,
                 filter.info, sig_half_d0, baseDim);
        }
        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

} // namespace kernel

} // namespace opencl

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
#include <common/dispatch.hpp>
#include <cache.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/fftconvolve_pack.hpp>
#include <kernel_headers/fftconvolve_multiply.hpp>
#include <kernel_headers/fftconvolve_reorder.hpp>
#include <memory.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
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
                    AF_BATCH_KIND kind)
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

    if (kind == AF_BATCH_RHS) {
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
                    AF_BATCH_KIND kind)
{
    std::string refName =
        std::string("pack_data_") +
        std::string(dtype_traits<convT>::getName()) +
        std::string(dtype_traits<T>::getName()) +
        std::to_string(isDouble);

    int device = getActiveDeviceId();
    kc_entry_t pdkEntry = kernelCache(device, refName);

    if (pdkEntry.prog==0 && pdkEntry.ker==0) {
        std::ostringstream options;

        options << " -D T=" << dtype_traits<T>::getName();

        if ((af_dtype) dtype_traits<convT>::af_type == c32) {
            options << " -D CONVT=float";
        }
        else if ((af_dtype) dtype_traits<convT>::af_type == c64 && isDouble) {
            options << " -D CONVT=double"
                << " -D USE_DOUBLE";
        }

        const char* ker_strs[] = {fftconvolve_pack_cl};
        const int   ker_lens[] = {fftconvolve_pack_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        pdkEntry.prog = new Program(prog);
        pdkEntry.ker  = new Kernel(*pdkEntry.prog, "pack_data");

        addKernelToCache(device, refName, pdkEntry);
    }

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
    auto pdOp = KernelFunctor< Buffer, KParam, Buffer, KParam, const int, const int > (*pdkEntry.ker);

    pdOp(EnqueueArgs(getQueue(), global, local),
         *sig_tmp.data, sig_tmp.info, *sig.data, sig.info, sig_half_d0, sig_half_d0_odd);

    CL_DEBUG_FINISH(getQueue());

    refName =
        std::string("pack_array_") +
        std::string(dtype_traits<convT>::getName()) +
        std::string(dtype_traits<T>::getName()) +
        std::to_string(isDouble);

    kc_entry_t pakEntry = kernelCache(device, refName);

    if (pakEntry.prog==0 && pakEntry.ker==0) {
        std::ostringstream options;

        options << " -D T=" << dtype_traits<T>::getName();

        if ((af_dtype) dtype_traits<convT>::af_type == c32) {
            options << " -D CONVT=float";
        }
        else if ((af_dtype) dtype_traits<convT>::af_type == c64 && isDouble) {
            options << " -D CONVT=double"
                << " -D USE_DOUBLE";
        }

        const char* ker_strs[] = {fftconvolve_pack_cl};
        const int   ker_lens[] = {fftconvolve_pack_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        pakEntry.prog = new Program(prog);
        pakEntry.ker  = new Kernel(*pakEntry.prog, "pad_array");

        addKernelToCache(device, refName, pakEntry);
    }

    blocks = divup(filter_packed_elem, THREADS);
    global = NDRange(blocks * THREADS);

    // Pad filter array with 0s
    auto paOp = KernelFunctor< Buffer, KParam, Buffer, KParam > (*pakEntry.ker);

    paOp(EnqueueArgs(getQueue(), global, local),
         *filter_tmp.data, filter_tmp.info, *filter.data, filter.info);

    CL_DEBUG_FINISH(getQueue());
}

template<typename convT, typename T, bool isDouble, typename printT>
void complexMultiplyHelper(Param packed,
                           Param sig,
                           Param filter,
                           const int baseDim,
                           AF_BATCH_KIND kind)
{
    std::string refName =
        std::string("complex_multiply_") +
        std::string(dtype_traits<convT>::getName()) +
        std::string(dtype_traits<T>::getName()) +
        std::to_string(isDouble);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;

        options << " -D T=" << dtype_traits<T>::getName()
            << " -D AF_BATCH_NONE=" << (int)AF_BATCH_NONE
            << " -D AF_BATCH_LHS="  << (int)AF_BATCH_LHS
            << " -D AF_BATCH_RHS="  << (int)AF_BATCH_RHS
            << " -D AF_BATCH_SAME=" << (int)AF_BATCH_SAME;

        if ((af_dtype) dtype_traits<convT>::af_type == c32) {
            options << " -D CONVT=float";
        } else if ((af_dtype) dtype_traits<convT>::af_type == c64 && isDouble) {
            options << " -D CONVT=double"
                    << " -D USE_DOUBLE";
        }

        const char* ker_strs[] = {fftconvolve_multiply_cl};
        const int   ker_lens[] = {fftconvolve_multiply_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "complex_multiply");

        addKernelToCache(device, refName, entry);
    }

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
    auto cmOp = KernelFunctor< Buffer, KParam, Buffer, KParam, Buffer, KParam,
                               const int, const int > (*entry.ker);

    cmOp(EnqueueArgs(getQueue(), global, local),
         *packed.data, packed.info, *sig_tmp.data, sig_tmp.info,
         *filter_tmp.data, filter_tmp.info, mul_elem, (int)kind);

    CL_DEBUG_FINISH(getQueue());
}

template<typename T, typename convT, bool isDouble, bool roundOut, bool expand, typename printT>
void reorderOutputHelper(Param out,
                         Param packed,
                         Param sig,
                         Param filter,
                         const int baseDim,
                         AF_BATCH_KIND kind)
{
    std::string refName =
        std::string("reorder_output_") +
        std::string(dtype_traits<T>::getName()) +
        std::string(dtype_traits<convT>::getName()) +
        std::to_string(isDouble) +
        std::to_string(roundOut) +
        std::to_string(expand);

    int device = getActiveDeviceId();
    kc_entry_t entry = kernelCache(device, refName);

    if (entry.prog==0 && entry.ker==0) {
        std::ostringstream options;

        options << " -D T=" << dtype_traits<T>::getName()
            << " -D ROUND_OUT=" << (int)roundOut
            << " -D EXPAND=" << (int)expand;

        if ((af_dtype) dtype_traits<convT>::af_type == c32) {
            options << " -D CONVT=float";
        } else if ((af_dtype) dtype_traits<convT>::af_type == c64 && isDouble) {
            options << " -D CONVT=double"
                << " -D USE_DOUBLE";
        }

        const char* ker_strs[] = {fftconvolve_reorder_cl};
        const int   ker_lens[] = {fftconvolve_reorder_cl_len};
        Program prog;
        buildProgram(prog, 1, ker_strs, ker_lens, options.str());
        entry.prog = new Program(prog);
        entry.ker  = new Kernel(*entry.prog, "reorder_output");

        addKernelToCache(device, refName, entry);
    }

    int fftScale = 1;

    // Calculate the scale by which to divide clFFT results
    for (int k = 0; k < baseDim; k++)
        fftScale *= packed.info.dims[k];

    Param sig_tmp, filter_tmp;
    calcParamSizes(sig_tmp, filter_tmp, packed, sig, filter, baseDim, kind);

    // Number of packed complex elements in dimension 0
    int sig_half_d0 = divup(sig.info.dims[0], 2);

    int blocks = divup(out.info.strides[3] * out.info.dims[3], THREADS);

    NDRange local(THREADS);
    NDRange global(blocks * THREADS);

    auto roOp = KernelFunctor< Buffer, KParam, Buffer, KParam, KParam, const int,
                               const int, const int > (*entry.ker);

    if (kind == AF_BATCH_RHS) {
        roOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *filter_tmp.data, filter_tmp.info, filter.info, sig_half_d0, baseDim, fftScale);
    } else {
        roOp(EnqueueArgs(getQueue(), global, local), *out.data, out.info,
             *sig_tmp.data, sig_tmp.info, filter.info, sig_half_d0, baseDim, fftScale);
    }

    CL_DEBUG_FINISH(getQueue());
}
} // namespace kernel
} // namespace opencl

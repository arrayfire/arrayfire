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
#include <err_clfft.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/fftconvolve.hpp>
#include <memory.hpp>

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

static const dim_type THREADS = 256;

template<typename T, typename convT, bool isDouble, bool roundOut, dim_type baseDim, bool expand>
void fftconvolve(Param out,
                 Param sig,
                 Param filter,
                 ConvolveBatchKind kind)
{
    try {
        static std::once_flag     compileFlags[DeviceManager::MAX_DEVICES];
        static Program        fftconvolveProgs[DeviceManager::MAX_DEVICES];
        static Kernel                 pdKernel[DeviceManager::MAX_DEVICES];
        static Kernel                 paKernel[DeviceManager::MAX_DEVICES];
        static Kernel                 cmKernel[DeviceManager::MAX_DEVICES];
        static Kernel                 roKernel[DeviceManager::MAX_DEVICES];

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {

                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D CONVT=" << dtype_traits<convT>::getName()
                        << " -D ROUND_OUT=" << (int)roundOut
                        << " -D EXPAND=" << (int)expand
                        << " -D ONE2ONE=" << (int)ONE2ONE
                        << " -D MANY2ONE=" << (int)MANY2ONE
                        << " -D ONE2MANY=" << (int)ONE2MANY
                        << " -D MANY2MANY=" << (int)MANY2MANY;

                if (std::is_same<T, double>::value) {
                    options << " -D USE_DOUBLE";
                }

                buildProgram(fftconvolveProgs[device],
                             fftconvolve_cl,
                             fftconvolve_cl_len,
                             options.str());

                pdKernel[device] = Kernel(fftconvolveProgs[device], "pack_data");
                paKernel[device] = Kernel(fftconvolveProgs[device], "pad_array");
                cmKernel[device] = Kernel(fftconvolveProgs[device], "complex_multiply");
                roKernel[device] = Kernel(fftconvolveProgs[device], "reorder_output");
            });

        dim_type *sd = sig.info.dims;
        dim_type *fd = filter.info.dims;
        dim_type fftScale = 1;

        Param packed;
        size_t fft_dims[4];

        // Pack both signal and filter on same memory array, this will ensure
        // better use of batched clFFT capabilities
        for (dim_type k = 0; k < 4; k++) {
            if (k < baseDim)
                packed.info.dims[k] = 1 << (unsigned)ceil(log2(sd[k] + fd[k] - 1));
            else if (k == baseDim)
                packed.info.dims[k] = sd[k] + fd[k];
            else
                packed.info.dims[k] = 1;

            packed.info.strides[k] = (k == 0) ? 1 : packed.info.strides[k - 1] * packed.info.dims[k - 1];

            fft_dims[k] = (k == 0) ? packed.info.dims[k] / 2 : packed.info.dims[k];
            if (k < baseDim) {
                fftScale *= fft_dims[baseDim-k-1];
            }
        }

        dim_type packed_elem = packed.info.strides[3] * packed.info.dims[3];

        // Create clFFT plan
        clfftPlanHandle plan;
        CLFFT_CHECK(clfftCreateDefaultPlan(&plan, getContext()(), (clfftDim)baseDim, fft_dims));

        size_t fft_strides[4];
        fft_strides[0] = 1;
        fft_strides[1] = fft_strides[0] * fft_dims[0];
        for (dim_type k = 2; k < 4; k++) {
            fft_strides[k] = fft_strides[k - 1] * fft_dims[k - 1];
        }

        // Configure clFFT plan
        CLFFT_CHECK(clfftSetLayout(plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED));
        CLFFT_CHECK(clfftSetPlanBatchSize(plan, fft_dims[baseDim]));
        CLFFT_CHECK(clfftSetPlanDistance(plan, fft_strides[baseDim], fft_strides[baseDim]));
        CLFFT_CHECK(clfftSetPlanInStride(plan, (clfftDim)baseDim, fft_strides));
        CLFFT_CHECK(clfftSetPlanOutStride(plan, (clfftDim)baseDim, fft_strides));
        if (isDouble)
            CLFFT_CHECK(clfftSetPlanPrecision(plan, CLFFT_DOUBLE));
        else
            CLFFT_CHECK(clfftSetPlanPrecision(plan, CLFFT_SINGLE));
        CLFFT_CHECK(clfftSetResultLocation(plan, CLFFT_INPLACE));

        // clfftBakePlan is commented out as it fails for some of the unit tests,
        // the best solution is possibly to unify FFT calls with fft(), so both
        // will benefit of plan caches
        //CLFFT_CHECK(clfftBakePlan(plan, 1, &(getQueue()()), NULL, NULL));

        packed.data = bufferAlloc(packed_elem * sizeof(convT));

        Param sig_tmp, filter_tmp;
        sig_tmp.info.dims[0] = filter_tmp.info.dims[0] = packed.info.dims[0];
        sig_tmp.info.strides[0] = filter_tmp.info.strides[0] = 1;

        for (dim_type k = 1; k < 4; k++) {
            if (k < baseDim) {
                sig_tmp.info.dims[k]    = packed.info.dims[k];
                filter_tmp.info.dims[k] = packed.info.dims[k];
            }
            else {
                sig_tmp.info.dims[k]    = sd[k];
                filter_tmp.info.dims[k] = fd[k];
            }

            sig_tmp.info.strides[k]    = sig_tmp.info.strides[k - 1] * sig_tmp.info.dims[k - 1];
            filter_tmp.info.strides[k] = filter_tmp.info.strides[k - 1] * filter_tmp.info.dims[k - 1];
        }

        // Calculate memory offsets for packed signal and filter
        sig_tmp.data = packed.data;
        filter_tmp.data = packed.data;

        const dim_type sig_tmp_off = 0;
        const dim_type filter_tmp_off = sig_tmp.info.strides[3] * sig_tmp.info.dims[3];

        dim_type sig_packed_elem = sig_tmp.info.strides[3] * sig_tmp.info.dims[3];
        dim_type filter_packed_elem = filter_tmp.info.strides[3] * filter_tmp.info.dims[3];

        // Number of packed complex elements in dimension 0
        dim_type sig_half_d0 = divup(sd[0], 2);
        int sig_half_d0_odd = sd[0] % 2;

        dim_type blocks = divup(sig_packed_elem / 2, THREADS);

        // Locate features kernel sizes
        NDRange local(THREADS);
        NDRange global(blocks * THREADS);

        // Pack signal in a complex matrix where first dimension is half the input
        // (allows faster FFT computation) and pad array to a power of 2 with 0s
        auto pdOp = make_kernel<Buffer, KParam,
                                Buffer, KParam,
                                const dim_type, const int> (pdKernel[device]);

        pdOp(EnqueueArgs(getQueue(), global, local),
             *sig_tmp.data, sig_tmp.info, *sig.data, sig.info,
             sig_half_d0, sig_half_d0_odd);
        CL_DEBUG_FINISH(getQueue());

        blocks = divup(filter_packed_elem, THREADS);
        global = NDRange(blocks * THREADS);

        // Pad filter array with 0s
        auto paOp = make_kernel<Buffer, KParam, const dim_type,
                                Buffer, KParam> (paKernel[device]);

        paOp(EnqueueArgs(getQueue(), global, local),
             *sig_tmp.data, filter_tmp.info, filter_tmp_off,
             *filter.data, filter.info);
        CL_DEBUG_FINISH(getQueue());

        // Compute forward FFT
        CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &(getQueue()()),
                                          0, NULL, NULL, &((*packed.data)()),
                                          NULL, NULL));

        dim_type mul_elem = (sig_packed_elem < filter_packed_elem) ?
                            filter_packed_elem / 2 : sig_packed_elem / 2;
        blocks = divup(mul_elem, THREADS);
        global = NDRange(blocks * THREADS);

        // Multiply filter and signal FFT arrays
        auto cmOp = make_kernel<Buffer, KParam, const dim_type,
                                Buffer, KParam, const dim_type,
                                Buffer, KParam, const dim_type,
                                const dim_type, const int> (cmKernel[device]);

        if (kind == ONE2MANY) {
            cmOp(EnqueueArgs(getQueue(), global, local),
                 *filter_tmp.data, filter_tmp.info, filter_tmp_off,
                 *sig_tmp.data, sig_tmp.info, sig_tmp_off,
                 *filter_tmp.data, filter_tmp.info, filter_tmp_off,
                 mul_elem, (int)kind);
        }
        else {
            cmOp(EnqueueArgs(getQueue(), global, local),
                 *sig_tmp.data, sig_tmp.info, sig_tmp_off,
                 *sig_tmp.data, sig_tmp.info, sig_tmp_off,
                 *filter_tmp.data, filter_tmp.info, filter_tmp_off,
                 mul_elem, (int)kind);
        }
        CL_DEBUG_FINISH(getQueue());

        // Compute inverse FFT
        CLFFT_CHECK(clfftEnqueueTransform(plan, CLFFT_BACKWARD, 1, &(getQueue()()),
                                          0, NULL, NULL, &((*packed.data)()),
                                          NULL, NULL));

        CLFFT_CHECK(clfftDestroyPlan(&plan));

        blocks = divup(out.info.strides[3] * out.info.dims[3], THREADS);
        global = NDRange(blocks * THREADS);

        auto roOp = make_kernel<Buffer, KParam,
                                Buffer, KParam, const dim_type,
                                KParam, const dim_type,
                                const int, const int> (roKernel[device]);

        if (kind == ONE2MANY) {
            roOp(EnqueueArgs(getQueue(), global, local),
                 *out.data, out.info,
                 *filter_tmp.data, filter_tmp.info, filter_tmp_off,
                 filter.info, sig_half_d0, baseDim, fftScale);
        }
        else {
            roOp(EnqueueArgs(getQueue(), global, local),
                 *out.data, out.info,
                 *sig_tmp.data, sig_tmp.info, sig_tmp_off,
                 filter.info, sig_half_d0, baseDim, fftScale);
        }

        bufferFree(packed.data);
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

} // namespace kernel

} // namespace cuda

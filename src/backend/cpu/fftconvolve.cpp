/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <dispatch.hpp>
#include <fft.hpp>
#include <err_cpu.hpp>
#include <fftw3.h>
#include <copy.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/fftconvolve.hpp>

namespace cpu
{

template<typename T, typename convT, typename cT, bool isDouble, bool roundOut, dim_t baseDim>
Array<T> fftconvolve(Array<T> const& signal, Array<T> const& filter,
                     const bool expand, AF_BATCH_KIND kind)
{
    signal.eval();
    filter.eval();

    const af::dim4 sd = signal.dims();
    const af::dim4 fd = filter.dims();

    dim_t fftScale = 1;

    af::dim4 packed_dims(1, 1, 1, 1);
    int fft_dims[baseDim];
    af::dim4 sig_tmp_dims, sig_tmp_strides;
    af::dim4 filter_tmp_dims, filter_tmp_strides;

    // Pack both signal and filter on same memory array, this will ensure
    // better use of batched FFT capabilities
    fft_dims[baseDim - 1] = nextpow2((unsigned)((int)ceil(sd[0] / 2.f) + fd[0] - 1));
    packed_dims[0] = 2 * fft_dims[baseDim - 1];
    fftScale *= fft_dims[baseDim - 1];

    for (dim_t k = 1; k < baseDim; k++) {
        packed_dims[k] = nextpow2((unsigned)(sd[k] + fd[k] - 1));
        fft_dims[baseDim - k - 1] = packed_dims[k];
        fftScale *= fft_dims[baseDim - k - 1];
    }

    dim_t sbatch = 1, fbatch = 1;
    for (int k = baseDim; k < 4; k++) {
        sbatch *= sd[k];
        fbatch *= fd[k];
    }
    packed_dims[baseDim] = (sbatch + fbatch);

    Array<convT> packed = createEmptyArray<convT>(packed_dims);

    sig_tmp_dims[0]    = filter_tmp_dims[0] = packed_dims[0];
    sig_tmp_strides[0] = filter_tmp_strides[0] = 1;

    for (dim_t k = 1; k < 4; k++) {
        if (k < baseDim) {
            sig_tmp_dims[k]    = packed_dims[k];
            filter_tmp_dims[k] = packed_dims[k];
        }
        else {
            sig_tmp_dims[k]    = sd[k];
            filter_tmp_dims[k] = fd[k];
        }

        sig_tmp_strides[k]    = sig_tmp_strides[k - 1] * sig_tmp_dims[k - 1];
        filter_tmp_strides[k] = filter_tmp_strides[k - 1] * filter_tmp_dims[k - 1];
    }

    // Number of packed complex elements in dimension 0
    dim_t sig_half_d0 = divup(sd[0], 2);

    // Pack signal in a complex matrix where first dimension is half the input
    // (allows faster FFT computation) and pad array to a power of 2 with 0s
    getQueue().enqueue(kernel::packData<convT, T>, packed, sig_tmp_dims, sig_tmp_strides, signal);

    // Pad filter array with 0s
    const dim_t offset = sig_tmp_strides[3]*sig_tmp_dims[3];
    getQueue().enqueue(kernel::padArray<convT, T>, packed, filter_tmp_dims, filter_tmp_strides,
                       filter, offset);

    dim4 fftDims(1, 1, 1, 1);
    for (int i=0; i<baseDim; ++i)
        fftDims[i] = fft_dims[i];

    auto upstream_dft = [=] (Param<convT> packed, const dim4 fftDims) {
        int fft_dims[baseDim];
        for (int i=0; i<baseDim; ++i)
            fft_dims[i] = fftDims[i];
        const dim4 packed_dims = packed.dims();
        const af::dim4 packed_strides = packed.strides();
        // Compute forward FFT
        if (isDouble) {
            fftw_plan plan = fftw_plan_many_dft(baseDim,
                                                fft_dims,
                                                packed_dims[baseDim],
                                                (fftw_complex*)packed.get(),
                                                NULL,
                                                packed_strides[0],
                                                packed_strides[baseDim] / 2,
                                                (fftw_complex*)packed.get(),
                                                NULL,
                                                packed_strides[0],
                                                packed_strides[baseDim] / 2,
                                                FFTW_FORWARD,
                                                FFTW_ESTIMATE);

            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            fftwf_plan plan = fftwf_plan_many_dft(baseDim,
                                                  fft_dims,
                                                  packed_dims[baseDim],
                                                  (fftwf_complex*)packed.get(),
                                                  NULL,
                                                  packed_strides[0],
                                                  packed_strides[baseDim] / 2,
                                                  (fftwf_complex*)packed.get(),
                                                  NULL,
                                                  packed_strides[0],
                                                  packed_strides[baseDim] / 2,
                                                  FFTW_FORWARD,
                                                  FFTW_ESTIMATE);

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
    };
    getQueue().enqueue(upstream_dft, packed, fftDims);

    // Multiply filter and signal FFT arrays
    getQueue().enqueue(kernel::complexMultiply<convT>, packed,
                       sig_tmp_dims, sig_tmp_strides,
                       filter_tmp_dims, filter_tmp_strides,
                       kind, offset);

    auto upstream_idft = [=] (Param<convT> packed, const dim4 fftDims) {
        int fft_dims[baseDim];
        for (int i=0; i<baseDim; ++i)
            fft_dims[i] = fftDims[i];
        const dim4 packed_dims = packed.dims();
        const af::dim4 packed_strides = packed.strides();
        // Compute inverse FFT
        if (isDouble) {
            fftw_plan plan = fftw_plan_many_dft(baseDim,
                                                fft_dims,
                                                packed_dims[baseDim],
                                                (fftw_complex*)packed.get(),
                                                NULL,
                                                packed_strides[0],
                                                packed_strides[baseDim] / 2,
                                                (fftw_complex*)packed.get(),
                                                NULL,
                                                packed_strides[0],
                                                packed_strides[baseDim] / 2,
                                                FFTW_BACKWARD,
                                                FFTW_ESTIMATE);

            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            fftwf_plan plan = fftwf_plan_many_dft(baseDim,
                                                  fft_dims,
                                                  packed_dims[baseDim],
                                                  (fftwf_complex*)packed.get(),
                                                  NULL,
                                                  packed_strides[0],
                                                  packed_strides[baseDim] / 2,
                                                  (fftwf_complex*)packed.get(),
                                                  NULL,
                                                  packed_strides[0],
                                                  packed_strides[baseDim] / 2,
                                                  FFTW_BACKWARD,
                                                  FFTW_ESTIMATE);

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
    };
    getQueue().enqueue(upstream_idft, packed, fftDims);

    // Compute output dimensions
    dim4 oDims(1);
    if (expand) {
        for(dim_t d=0; d<4; ++d) {
            if (kind==AF_BATCH_NONE || kind==AF_BATCH_RHS) {
                oDims[d] = sd[d]+fd[d]-1;
            } else {
                oDims[d] = (d<baseDim ? sd[d]+fd[d]-1 : sd[d]);
            }
        }
    } else {
        oDims = sd;
        if (kind==AF_BATCH_RHS) {
            for (dim_t i=baseDim; i<4; ++i)
                oDims[i] = fd[i];
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);

    getQueue().enqueue(kernel::reorder<T, convT, roundOut, baseDim>, out, packed, filter,
                       sig_half_d0, fftScale, sig_tmp_dims, sig_tmp_strides, filter_tmp_dims,
                       filter_tmp_strides, expand, kind);

    return out;
}

#define INSTANTIATE(T, convT, cT, isDouble, roundOut)                                                   \
    template Array<T> fftconvolve <T, convT, cT, isDouble, roundOut, 1>                                 \
        (Array<T> const& signal, Array<T> const& filter, const bool expand, AF_BATCH_KIND kind);        \
    template Array<T> fftconvolve <T, convT, cT, isDouble, roundOut, 2>                                 \
        (Array<T> const& signal, Array<T> const& filter, const bool expand, AF_BATCH_KIND kind);        \
    template Array<T> fftconvolve <T, convT, cT, isDouble, roundOut, 3>                                 \
        (Array<T> const& signal, Array<T> const& filter, const bool expand, AF_BATCH_KIND kind);

INSTANTIATE(double, double, cdouble, true , false)
INSTANTIATE(float , float,  cfloat,  false, false)
INSTANTIATE(uint  , float,  cfloat,  false, true)
INSTANTIATE(int   , float,  cfloat,  false, true)
INSTANTIATE(uchar , float,  cfloat,  false, true)
INSTANTIATE(char  , float,  cfloat,  false, true)
INSTANTIATE(uintl , float,  cfloat,  false, true)
INSTANTIATE(intl  , float,  cfloat,  false, true)
INSTANTIATE(ushort, float,  cfloat,  false, true)
INSTANTIATE(short , float,  cfloat,  false, true)

} // namespace cpu

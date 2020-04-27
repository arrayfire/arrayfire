/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fftconvolve.hpp>

#include <Array.hpp>
#include <common/dispatch.hpp>
#include <fftw3.h>
#include <kernel/fftconvolve.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

#include <array>
#include <cmath>

using af::dim4;
using std::array;
using std::ceil;

namespace cpu {

template<typename T, typename convT, typename cT, bool isDouble, bool roundOut,
         dim_t baseDim>
Array<T> fftconvolve(Array<T> const& signal, Array<T> const& filter,
                     const bool expand, AF_BATCH_KIND kind) {
    const dim4& sd = signal.dims();
    const dim4& fd = filter.dims();
    dim_t fftScale = 1;

    dim4 packedDims(1, 1, 1, 1);
    array<int, baseDim> fftDims;

    // Pack both signal and filter on same memory array, this will ensure
    // better use of batched FFT capabilities
    fftDims[baseDim - 1] = nextpow2(
        static_cast<unsigned>(static_cast<int>(ceil(sd[0] / 2.f)) + fd[0] - 1));
    packedDims[0] = 2 * fftDims[baseDim - 1];
    fftScale *= fftDims[baseDim - 1];

    for (dim_t k = 1; k < baseDim; k++) {
        packedDims[k] = nextpow2(static_cast<unsigned>(sd[k] + fd[k] - 1));
        fftDims[baseDim - k - 1] = packedDims[k];
        fftScale *= fftDims[baseDim - k - 1];
    }

    dim_t sbatch = 1, fbatch = 1;
    for (int k = baseDim; k < AF_MAX_DIMS; k++) {
        sbatch *= sd[k];
        fbatch *= fd[k];
    }
    packedDims[baseDim] = (sbatch + fbatch);

    Array<convT> packed = createEmptyArray<convT>(packedDims);

    dim4 paddedSigDims(packedDims[0], (1 < baseDim ? packedDims[1] : sd[1]),
                       (2 < baseDim ? packedDims[2] : sd[2]),
                       (3 < baseDim ? packedDims[3] : sd[3]));
    dim4 paddedFilDims(packedDims[0], (1 < baseDim ? packedDims[1] : fd[1]),
                       (2 < baseDim ? packedDims[2] : fd[2]),
                       (3 < baseDim ? packedDims[3] : fd[3]));
    dim4 paddedSigStrides = calcStrides(paddedSigDims);
    dim4 paddedFilStrides = calcStrides(paddedFilDims);

    // Number of packed complex elements in dimension 0
    dim_t sig_half_d0 = divup(sd[0], 2);

    // Pack signal in a complex matrix where first dimension is half the input
    // (allows faster FFT computation) and pad array to a power of 2 with 0s
    getQueue().enqueue(kernel::packData<convT, T>, packed, paddedSigDims,
                       paddedSigStrides, signal);

    // Pad filter array with 0s
    const dim_t offset = paddedSigStrides[3] * paddedSigDims[3];
    getQueue().enqueue(kernel::padArray<convT, T>, packed, paddedFilDims,
                       paddedFilStrides, filter, offset);

    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    auto upstream_dft = [=](Param<convT> packed,
                            const array<int, baseDim> fftDims) {
        const dim4 packedDims     = packed.dims();
        const dim4 packed_strides = packed.strides();
        // Compute forward FFT
        if (isDouble) {
            fftw_plan plan = fftw_plan_many_dft(
                baseDim, fftDims.data(), packedDims[baseDim],
                reinterpret_cast<fftw_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[baseDim] / 2,
                reinterpret_cast<fftw_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[baseDim] / 2, FFTW_FORWARD,
                FFTW_ESTIMATE);  // NOLINT(hicpp-signed-bitwise)

            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            fftwf_plan plan = fftwf_plan_many_dft(
                baseDim, fftDims.data(), packedDims[baseDim],
                reinterpret_cast<fftwf_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[baseDim] / 2,
                reinterpret_cast<fftwf_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[baseDim] / 2, FFTW_FORWARD,
                FFTW_ESTIMATE);  // NOLINT(hicpp-signed-bitwise)

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
    };
    getQueue().enqueue(upstream_dft, packed, fftDims);

    // Multiply filter and signal FFT arrays
    getQueue().enqueue(kernel::complexMultiply<convT>, packed, paddedSigDims,
                       paddedSigStrides, paddedFilDims, paddedFilStrides, kind,
                       offset);

    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    auto upstream_idft = [=](Param<convT> packed,
                             const array<int, baseDim> fftDims) {
        const dim4 packedDims     = packed.dims();
        const dim4 packed_strides = packed.strides();
        // Compute inverse FFT
        if (isDouble) {
            fftw_plan plan = fftw_plan_many_dft(
                baseDim, fftDims.data(), packedDims[baseDim],
                reinterpret_cast<fftw_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[baseDim] / 2,
                reinterpret_cast<fftw_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[baseDim] / 2, FFTW_BACKWARD,
                FFTW_ESTIMATE);  // NOLINT(hicpp-signed-bitwise)

            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            fftwf_plan plan = fftwf_plan_many_dft(
                baseDim, fftDims.data(), packedDims[baseDim],
                reinterpret_cast<fftwf_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[baseDim] / 2,
                reinterpret_cast<fftwf_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[baseDim] / 2, FFTW_BACKWARD,
                FFTW_ESTIMATE);  // NOLINT(hicpp-signed-bitwise)

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
    };
    getQueue().enqueue(upstream_idft, packed, fftDims);

    // Compute output dimensions
    dim4 oDims(1);
    if (expand) {
        for (dim_t d = 0; d < 4; ++d) {
            if (kind == AF_BATCH_NONE || kind == AF_BATCH_RHS) {
                oDims[d] = sd[d] + fd[d] - 1;
            } else {
                oDims[d] = (d < baseDim ? sd[d] + fd[d] - 1 : sd[d]);
            }
        }
    } else {
        oDims = sd;
        if (kind == AF_BATCH_RHS) {
            for (dim_t i = baseDim; i < 4; ++i) { oDims[i] = fd[i]; }
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);

    getQueue().enqueue(kernel::reorder<T, convT, roundOut, baseDim>, out,
                       packed, filter, sig_half_d0, fftScale, paddedSigDims,
                       paddedSigStrides, paddedFilDims, paddedFilStrides,
                       expand, kind);

    return out;
}

#define INSTANTIATE(T, convT, cT, isDouble, roundOut)                      \
    template Array<T> fftconvolve<T, convT, cT, isDouble, roundOut, 1>(    \
        Array<T> const& signal, Array<T> const& filter, const bool expand, \
        AF_BATCH_KIND kind);                                               \
    template Array<T> fftconvolve<T, convT, cT, isDouble, roundOut, 2>(    \
        Array<T> const& signal, Array<T> const& filter, const bool expand, \
        AF_BATCH_KIND kind);                                               \
    template Array<T> fftconvolve<T, convT, cT, isDouble, roundOut, 3>(    \
        Array<T> const& signal, Array<T> const& filter, const bool expand, \
        AF_BATCH_KIND kind);

INSTANTIATE(double, double, cdouble, true, false)
INSTANTIATE(float, float, cfloat, false, false)
INSTANTIATE(uint, float, cfloat, false, true)
INSTANTIATE(int, float, cfloat, false, true)
INSTANTIATE(uchar, float, cfloat, false, true)
INSTANTIATE(char, float, cfloat, false, true)
INSTANTIATE(uintl, float, cfloat, false, true)
INSTANTIATE(intl, float, cfloat, false, true)
INSTANTIATE(ushort, float, cfloat, false, true)
INSTANTIATE(short, float, cfloat, false, true)

}  // namespace cpu

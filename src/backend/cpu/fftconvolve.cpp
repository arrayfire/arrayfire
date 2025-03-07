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
#include <functional>
#include <type_traits>

using af::dim4;
using std::array;
using std::ceil;

namespace arrayfire {
namespace cpu {

template<typename T, typename convT>
using reorderFunc = std::function<void(
    Param<T> out, Param<convT> packed, CParam<T> filter,
    const dim_t sig_half_d0, const dim_t fftScale, const dim4 sig_tmp_dims,
    const dim4 sig_tmp_strides, const dim4 filter_tmp_dims,
    const dim4 filter_tmp_strides, AF_BATCH_KIND kind)>;

template<typename T>
Array<T> fftconvolve(Array<T> const& signal, Array<T> const& filter,
                     const bool expand, AF_BATCH_KIND kind, const int rank) {
    using convT = typename std::conditional<std::is_integral<T>::value ||
                                                std::is_same<T, float>::value,
                                            float, double>::type;

    constexpr bool IsTypeDouble = std::is_same<T, double>::value;

    const dim4& sd = signal.dims();
    const dim4& fd = filter.dims();
    dim_t fftScale = 1;

    dim4 packedDims(1, 1, 1, 1);
    array<int, AF_MAX_DIMS> fftDims{};  // AF_MAX_DIMS(4) > rank

    // Pack both signal and filter on same memory array, this will ensure
    // better use of batched FFT capabilities
    fftDims[rank - 1] = nextpow2(
        static_cast<unsigned>(static_cast<int>(ceil(sd[0] / 2.f)) + fd[0] - 1));
    packedDims[0] = 2 * fftDims[rank - 1];
    fftScale *= fftDims[rank - 1];

    for (int k = 1; k < rank; k++) {
        packedDims[k] = nextpow2(static_cast<unsigned>(sd[k] + fd[k] - 1));
        fftDims[rank - k - 1] = packedDims[k];
        fftScale *= fftDims[rank - k - 1];
    }

    dim_t sbatch = 1, fbatch = 1;
    for (int k = rank; k < AF_MAX_DIMS; k++) {
        sbatch *= sd[k];
        fbatch *= fd[k];
    }
    packedDims[rank] = (sbatch + fbatch);

    Array<convT> packed = createEmptyArray<convT>(packedDims);

    dim4 paddedSigDims(packedDims[0], (1 < rank ? packedDims[1] : sd[1]),
                       (2 < rank ? packedDims[2] : sd[2]),
                       (3 < rank ? packedDims[3] : sd[3]));
    dim4 paddedFilDims(packedDims[0], (1 < rank ? packedDims[1] : fd[1]),
                       (2 < rank ? packedDims[2] : fd[2]),
                       (3 < rank ? packedDims[3] : fd[3]));
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
                            const array<int, AF_MAX_DIMS> fftDims) {
        const dim4 packedDims     = packed.dims();
        const dim4 packed_strides = packed.strides();
        // Compute forward FFT
        if (IsTypeDouble) {
            fftw_plan plan = fftw_plan_many_dft(
                rank, fftDims.data(), packedDims[rank],
                reinterpret_cast<fftw_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[rank] / 2,
                reinterpret_cast<fftw_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[rank] / 2, FFTW_FORWARD,
                FFTW_ESTIMATE);  // NOLINT(hicpp-signed-bitwise)

            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            fftwf_plan plan = fftwf_plan_many_dft(
                rank, fftDims.data(), packedDims[rank],
                reinterpret_cast<fftwf_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[rank] / 2,
                reinterpret_cast<fftwf_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[rank] / 2, FFTW_FORWARD,
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
                             const array<int, AF_MAX_DIMS> fftDims) {
        const dim4 packedDims     = packed.dims();
        const dim4 packed_strides = packed.strides();
        // Compute inverse FFT
        if (IsTypeDouble) {
            fftw_plan plan = fftw_plan_many_dft(
                rank, fftDims.data(), packedDims[rank],
                reinterpret_cast<fftw_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[rank] / 2,
                reinterpret_cast<fftw_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[rank] / 2, FFTW_BACKWARD,
                FFTW_ESTIMATE);  // NOLINT(hicpp-signed-bitwise)

            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            fftwf_plan plan = fftwf_plan_many_dft(
                rank, fftDims.data(), packedDims[rank],
                reinterpret_cast<fftwf_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[rank] / 2,
                reinterpret_cast<fftwf_complex*>(packed.get()), nullptr,
                packed_strides[0], packed_strides[rank] / 2, FFTW_BACKWARD,
                FFTW_ESTIMATE);  // NOLINT(hicpp-signed-bitwise)

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
    };
    getQueue().enqueue(upstream_idft, packed, fftDims);

    // Compute output dimensions
    dim4 oDims(1);
    if (expand) {
        for (int d = 0; d < AF_MAX_DIMS; ++d) {
            if (kind == AF_BATCH_NONE || kind == AF_BATCH_RHS) {
                oDims[d] = sd[d] + fd[d] - 1;
            } else {
                oDims[d] = (d < rank ? sd[d] + fd[d] - 1 : sd[d]);
            }
        }
    } else {
        oDims = sd;
        if (kind == AF_BATCH_RHS) {
            for (int i = rank; i < AF_MAX_DIMS; ++i) { oDims[i] = fd[i]; }
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);

    static const reorderFunc<T, convT> funcs[6] = {
        kernel::reorder<T, convT, 1, false>,
        kernel::reorder<T, convT, 2, false>,
        kernel::reorder<T, convT, 3, false>,
        kernel::reorder<T, convT, 1, true>,
        kernel::reorder<T, convT, 2, true>,
        kernel::reorder<T, convT, 3, true>,
    };

    getQueue().enqueue(funcs[expand * 3 + (rank - 1)], out, packed, filter,
                       sig_half_d0, fftScale, paddedSigDims, paddedSigStrides,
                       paddedFilDims, paddedFilStrides, kind);

    return out;
}

#define INSTANTIATE(T)                                                 \
    template Array<T> fftconvolve<T>(Array<T> const&, Array<T> const&, \
                                     const bool, AF_BATCH_KIND, const int);

INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(uint)
INSTANTIATE(int)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(uintl)
INSTANTIATE(intl)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace cpu
}  // namespace arrayfire

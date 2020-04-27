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
#include <fft.hpp>
#include <kernel/fftconvolve.hpp>
#include <af/dim4.hpp>

#include <cmath>
#include <type_traits>
#include <vector>

using af::dim4;
using std::ceil;
using std::conditional;
using std::is_integral;
using std::is_same;
using std::vector;

namespace opencl {

template<typename T>
static dim4 calcPackedSize(Array<T> const& i1, Array<T> const& i2,
                           const dim_t baseDim) {
    const dim4& i1d = i1.dims();
    const dim4& i2d = i2.dims();

    dim_t pd[4] = {1, 1, 1, 1};

    // Pack both signal and filter on same memory array, this will ensure
    // better use of batched cuFFT capabilities
    pd[0] = nextpow2(static_cast<unsigned>(
        static_cast<int>(ceil(i1d[0] / 2.f)) + i2d[0] - 1));

    for (dim_t k = 1; k < baseDim; k++) {
        pd[k] = nextpow2(static_cast<unsigned>(i1d[k] + i2d[k] - 1));
    }

    dim_t i1batch = 1;
    dim_t i2batch = 1;
    for (int k = baseDim; k < 4; k++) {
        i1batch *= i1d[k];
        i2batch *= i2d[k];
    }
    pd[baseDim] = (i1batch + i2batch);

    return dim4(pd[0], pd[1], pd[2], pd[3]);
}

template<typename T, dim_t baseDim>
Array<T> fftconvolve(Array<T> const& signal, Array<T> const& filter,
                     const bool expand, AF_BATCH_KIND kind) {
    using convT =
        typename conditional<is_integral<T>::value || is_same<T, float>::value,
                             float, double>::type;
    using cT = typename conditional<is_same<convT, float>::value, cfloat,
                                    cdouble>::type;

    const dim4& sDims = signal.dims();
    const dim4& fDims = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for (dim_t d = 0; d < 4; ++d) {
            if (kind == AF_BATCH_NONE || kind == AF_BATCH_RHS) {
                oDims[d] = sDims[d] + fDims[d] - 1;
            } else {
                oDims[d] = (d < baseDim ? sDims[d] + fDims[d] - 1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind == AF_BATCH_RHS) {
            for (dim_t i = baseDim; i < 4; ++i) { oDims[i] = fDims[i]; }
        }
    }

    const dim4 pDims = calcPackedSize<T>(signal, filter, baseDim);
    Array<cT> packed = createEmptyArray<cT>(pDims);

    kernel::packDataHelper<cT, T>(packed, signal, filter, baseDim, kind);
    fft_inplace<cT, baseDim, true>(packed);
    kernel::complexMultiplyHelper<cT, T>(packed, signal, filter, baseDim, kind);

    // Compute inverse FFT only on complex-multiplied data
    if (kind == AF_BATCH_RHS) {
        vector<af_seq> seqs;
        for (dim_t k = 0; k < 4; k++) {
            if (k < baseDim) {
                seqs.push_back({0., static_cast<double>(pDims[k] - 1), 1.});
            } else if (k == baseDim) {
                seqs.push_back({1., static_cast<double>(pDims[k] - 1), 1.});
            } else {
                seqs.push_back({0., 0., 1.});
            }
        }

        Array<cT> subPacked = createSubArray<cT>(packed, seqs);
        fft_inplace<cT, baseDim, false>(subPacked);
    } else {
        vector<af_seq> seqs;
        for (dim_t k = 0; k < 4; k++) {
            if (k < baseDim) {
                seqs.push_back({0., static_cast<double>(pDims[k]) - 1, 1.});
            } else if (k == baseDim) {
                seqs.push_back({0., static_cast<double>(pDims[k] - 2), 1.});
            } else {
                seqs.push_back({0., 0., 1.});
            }
        }

        Array<cT> subPacked = createSubArray<cT>(packed, seqs);
        fft_inplace<cT, baseDim, false>(subPacked);
    }

    Array<T> out = createEmptyArray<T>(oDims);

    kernel::reorderOutputHelper<T, cT>(out, packed, signal, filter, baseDim,
                                       kind, expand);
    return out;
}

#define INSTANTIATE(T)                                                     \
    template Array<T> fftconvolve<T, 1>(                                   \
        Array<T> const& signal, Array<T> const& filter, const bool expand, \
        AF_BATCH_KIND kind);                                               \
    template Array<T> fftconvolve<T, 2>(                                   \
        Array<T> const& signal, Array<T> const& filter, const bool expand, \
        AF_BATCH_KIND kind);                                               \
    template Array<T> fftconvolve<T, 3>(                                   \
        Array<T> const& signal, Array<T> const& filter, const bool expand, \
        AF_BATCH_KIND kind);

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

}  // namespace opencl

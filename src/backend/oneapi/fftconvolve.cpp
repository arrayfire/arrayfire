/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fftconvolve.hpp>

#include <Array.hpp>
#include <common/dispatch.hpp>
#include <err_oneapi.hpp>
#include <fft.hpp>
#include <af/dim4.hpp>

#include <kernel/fftconvolve_common.hpp>
#include <kernel/fftconvolve_multiply.hpp>
#include <kernel/fftconvolve_pack.hpp>
#include <kernel/fftconvolve_pad.hpp>
#include <kernel/fftconvolve_reorder.hpp>

#include <cmath>
#include <type_traits>
#include <vector>

using af::dim4;
using std::ceil;
using std::conditional;
using std::is_integral;
using std::is_same;
using std::vector;

namespace arrayfire {
namespace oneapi {

template<typename T>
dim4 calcPackedSize(Array<T> const& i1, Array<T> const& i2, const dim_t rank) {
    const dim4& i1d = i1.dims();
    const dim4& i2d = i2.dims();

    dim_t pd[4] = {1, 1, 1, 1};

    // Pack both signal and filter on same memory array, this will ensure
    // better use of batched cuFFT capabilities
    pd[0] = nextpow2(static_cast<unsigned>(
        static_cast<int>(ceil(i1d[0] / 2.f)) + i2d[0] - 1));

    for (dim_t k = 1; k < rank; k++) {
        pd[k] = nextpow2(static_cast<unsigned>(i1d[k] + i2d[k] - 1));
    }

    dim_t i1batch = 1;
    dim_t i2batch = 1;
    for (int k = rank; k < 4; k++) {
        i1batch *= i1d[k];
        i2batch *= i2d[k];
    }
    pd[rank] = (i1batch + i2batch);

    return dim4(pd[0], pd[1], pd[2], pd[3]);
}

template<typename T>
Array<T> fftconvolve(Array<T> const& signal, Array<T> const& filter,
                     const bool expand, AF_BATCH_KIND kind, const int rank) {
    using convT = typename conditional<is_integral<T>::value ||
                                           is_same<T, float>::value ||
                                           is_same<T, cfloat>::value,
                                       float, double>::type;
    using cT    = typename conditional<is_same<convT, float>::value, cfloat,
                                    cdouble>::type;

    const dim4& sDims = signal.dims();
    const dim4& fDims = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for (int d = 0; d < AF_MAX_DIMS; ++d) {
            if (kind == AF_BATCH_NONE || kind == AF_BATCH_RHS) {
                oDims[d] = sDims[d] + fDims[d] - 1;
            } else {
                oDims[d] = (d < rank ? sDims[d] + fDims[d] - 1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind == AF_BATCH_RHS) {
            for (int i = rank; i < AF_MAX_DIMS; ++i) { oDims[i] = fDims[i]; }
        }
    }

    const dim4 pDims = calcPackedSize<T>(signal, filter, rank);
    Array<cT> packed = createEmptyArray<cT>(pDims);

    kernel::packDataHelper<cT, T>(packed, signal, filter, rank, kind);
    kernel::padDataHelper<cT, T>(packed, signal, filter, rank, kind);

    fft_inplace<cT>(packed, rank, true);

    kernel::complexMultiplyHelper<cT, T>(packed, signal, filter, rank, kind);

    // Compute inverse FFT only on complex-multiplied data
    if (kind == AF_BATCH_RHS) {
        vector<af_seq> seqs;
        for (int k = 0; k < AF_MAX_DIMS; k++) {
            if (k < rank) {
                seqs.push_back({0., static_cast<double>(pDims[k] - 1), 1.});
            } else if (k == rank) {
                seqs.push_back({1., static_cast<double>(pDims[k] - 1), 1.});
            } else {
                seqs.push_back({0., 0., 1.});
            }
        }

        Array<cT> subPacked = createSubArray<cT>(packed, seqs);
        fft_inplace<cT>(subPacked, rank, false);
    } else {
        vector<af_seq> seqs;
        for (int k = 0; k < AF_MAX_DIMS; k++) {
            if (k < rank) {
                seqs.push_back({0., static_cast<double>(pDims[k]) - 1, 1.});
            } else if (k == rank) {
                seqs.push_back({0., static_cast<double>(pDims[k] - 2), 1.});
            } else {
                seqs.push_back({0., 0., 1.});
            }
        }

        Array<cT> subPacked = createSubArray<cT>(packed, seqs);
        fft_inplace<cT>(subPacked, rank, false);
    }

    Array<T> out = createEmptyArray<T>(oDims);

    kernel::reorderOutputHelper<T, cT>(out, packed, signal, filter, rank, kind,
                                       expand);

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

}  // namespace oneapi
}  // namespace arrayfire

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

#include <cmath>
#include <type_traits>
#include <vector>

using af::dim4;
using std::ceil;
using std::conditional;
using std::is_integral;
using std::is_same;
using std::vector;

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
    ONEAPI_NOT_SUPPORTED("");
    dim4 oDims(1);
    Array<T> out = createEmptyArray<T>(oDims);
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

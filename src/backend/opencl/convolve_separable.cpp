/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <convolve.hpp>

#include <Array.hpp>
#include <err_opencl.hpp>
#include <kernel/convolve_separable.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace opencl {

template<typename T, typename accT>
Array<T> convolve2(Array<T> const& signal, Array<accT> const& c_filter,
                   Array<accT> const& r_filter, const bool expand) {
    const auto cflen = c_filter.elements();
    const auto rflen = r_filter.elements();

    if ((cflen > kernel::MAX_SCONV_FILTER_LEN) ||
        (rflen > kernel::MAX_SCONV_FILTER_LEN)) {
        // TODO call upon fft
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\nOpenCL Separable convolution doesn't support %llu(coloumn) "
                 "%llu(row) filters\n",
                 cflen, rflen);
        OPENCL_NOT_SUPPORTED(errMessage);
    }

    const dim4& sDims = signal.dims();
    dim4 tDims        = sDims;
    dim4 oDims        = sDims;

    if (expand) {
        tDims[0] += cflen - 1;
        oDims[0] += cflen - 1;
        oDims[1] += rflen - 1;
    }

    Array<T> temp = createEmptyArray<T>(tDims);
    Array<T> out  = createEmptyArray<T>(oDims);

    kernel::convSep<T, accT>(temp, signal, c_filter, 0, expand);
    kernel::convSep<T, accT>(out, temp, r_filter, 1, expand);

    return out;
}

#define INSTANTIATE(T, accT)                                                  \
    template Array<T> convolve2<T, accT>(Array<T> const&, Array<accT> const&, \
                                         Array<accT> const&, const bool);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat, cfloat)
INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(uint, float)
INSTANTIATE(int, float)
INSTANTIATE(uchar, float)
INSTANTIATE(char, float)
INSTANTIATE(short, float)
INSTANTIATE(ushort, float)
INSTANTIATE(intl, float)
INSTANTIATE(uintl, float)

}  // namespace opencl
}  // namespace arrayfire

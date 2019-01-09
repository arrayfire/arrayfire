/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_opencl.hpp>
#include <kernel/meanshift.hpp>
#include <meanshift.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace opencl {
template <typename T>
Array<T> meanshift(const Array<T> &in, const float &spatialSigma,
                   const float &chromaticSigma, const unsigned &numIterations,
                   const bool &isColor) {
    const dim4 dims = in.dims();
    Array<T> out    = createEmptyArray<T>(dims);
    if (isColor)
        kernel::meanshift<T, true>(out, in, spatialSigma, chromaticSigma,
                                   numIterations);
    else
        kernel::meanshift<T, false>(out, in, spatialSigma, chromaticSigma,
                                    numIterations);
    return out;
}

#define INSTANTIATE(T)                                              \
    template Array<T> meanshift<T>(const Array<T> &, const float &, \
                                   const float &, const unsigned &, \
                                   const bool &);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)
}  // namespace opencl

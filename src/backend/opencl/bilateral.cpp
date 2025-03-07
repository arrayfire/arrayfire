/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <bilateral.hpp>
#include <kernel/bilateral.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace opencl {

template<typename inType, typename outType>
Array<outType> bilateral(const Array<inType> &in, const float &sSigma,
                         const float &cSigma) {
    Array<outType> out = createEmptyArray<outType>(in.dims());
    kernel::bilateral<inType, outType>(out, in, sSigma, cSigma);
    return out;
}

#define INSTANTIATE(inT, outT)                                    \
    template Array<outT> bilateral<inT, outT>(const Array<inT> &, \
                                              const float &, const float &);

INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(char, float)
INSTANTIATE(int, float)
INSTANTIATE(uint, float)
INSTANTIATE(uchar, float)
INSTANTIATE(short, float)
INSTANTIATE(ushort, float)

}  // namespace opencl
}  // namespace arrayfire

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <sobel.hpp>
#include <kernel/sobel.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename Ti, typename To>
std::pair< Array<To>, Array<To> >
sobelDerivatives(const Array<Ti> &img, const unsigned &ker_size)
{
    Array<To> dx = createEmptyArray<To>(img.dims());
    Array<To> dy = createEmptyArray<To>(img.dims());

    kernel::sobel<Ti, To>(dx, dy, img, ker_size);

    return std::make_pair(dx, dy);
}

#define INSTANTIATE(Ti, To)                                             \
    template std::pair< Array<To>, Array<To> >                          \
    sobelDerivatives(const Array<Ti> &img, const unsigned &ker_size);

INSTANTIATE(float , float)
INSTANTIATE(double, double)
INSTANTIATE(int   , int)
INSTANTIATE(uint  , int)
INSTANTIATE(char  , int)
INSTANTIATE(uchar , int)

}

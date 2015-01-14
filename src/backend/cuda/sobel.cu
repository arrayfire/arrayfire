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

template<typename T>
std::pair< Array<T>*, Array<T>* >
sobelDerivatives(const Array<T> &img, const unsigned &ker_size)
{
    Array<T> *dx = createEmptyArray<T>(img.dims());
    Array<T> *dy = createEmptyArray<T>(img.dims());

    kernel::sobel<T>(*dx, *dy, img, ker_size);

    return std::make_pair(dx, dy);
}

#define INSTANTIATE(T)\
    template std::pair< Array<T>*, Array<T>* > sobelDerivatives(const Array<T> &img, const unsigned &ker_size);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )

}

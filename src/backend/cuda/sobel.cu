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
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename T>
std::pair< Array<T>*, Array<T>* >
sobelDerivatives(const Array<T> &img, const unsigned &ker_size)
{
    CUDA_NOT_SUPPORTED();
}

#define INSTANTIATE(T)\
    template std::pair< Array<T>*, Array<T>* > sobelDerivatives(const Array<T> &img, const unsigned &ker_size);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}

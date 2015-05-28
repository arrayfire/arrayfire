/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/signal.h>
#include <af/array.h>
#include "error.hpp"
#include <algorithm>

namespace af
{

array fftConvolve(const array& signal, const array& filter, const convMode mode)
{
    unsigned sN = signal.numdims();
    unsigned fN = filter.numdims();

    switch(std::min(sN,fN)) {
        case 1:  return fftConvolve1(signal, filter, mode);
        case 2:  return fftConvolve2(signal, filter, mode);
        case 3:  return fftConvolve3(signal, filter, mode);
        default: return fftConvolve3(signal, filter, mode);
    }
}

array fftConvolve1(const array& signal, const array& filter, const convMode mode)
{
    af_array out = 0;
    AF_THROW(af_fft_convolve1(&out, signal.get(), filter.get(), mode));
    return array(out);
}

array fftConvolve2(const array& signal, const array& filter, const convMode mode)
{
    af_array out = 0;
    AF_THROW(af_fft_convolve2(&out, signal.get(), filter.get(), mode));
    return array(out);
}

array fftConvolve3(const array& signal, const array& filter, const convMode mode)
{
    af_array out = 0;
    AF_THROW(af_fft_convolve3(&out, signal.get(), filter.get(), mode));
    return array(out);
}

}

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/signal.h>
#include "error.hpp"
#include <algorithm>

namespace af
{

array fftconvolve(const array& signal, const array& filter)
{
    unsigned sN = signal.numdims();
    unsigned fN = filter.numdims();

    switch(std::min(sN,fN)) {
        case 1:  return fftconvolve1(signal, filter);
        case 2:  return fftconvolve2(signal, filter);
        case 3:  return fftconvolve3(signal, filter);
        default: return fftconvolve3(signal, filter);
    }
}

array fftconvolve1(const array& signal, const array& filter)
{
    af_array out = 0;
    AF_THROW(af_fftconvolve1(&out, signal.get(), filter.get()));
    return array(out);
}

array fftconvolve2(const array& signal, const array& filter)
{
    af_array out = 0;
    AF_THROW(af_fftconvolve2(&out, signal.get(), filter.get()));
    return array(out);
}

array fftconvolve3(const array& signal, const array& filter)
{
    af_array out = 0;
    AF_THROW(af_fftconvolve3(&out, signal.get(), filter.get()));
    return array(out);
}

}

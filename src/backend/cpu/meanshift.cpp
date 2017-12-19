/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <meanshift.hpp>
#include <cmath>
#include <algorithm>
#include <err_cpu.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <kernel/meanshift.hpp>

using af::dim4;
using std::vector;

namespace cpu
{
template<typename T>
Array<T>  meanshift(const Array<T> &in,
                    const float &spatialSigma, const float &chromaticSigma,
                    const unsigned& numInterations, const bool& isColor)
{
    in.eval();

    Array<T> out = createEmptyArray<T>(in.dims());

    if (isColor)
        getQueue().enqueue(kernel::meanShift<T, true>, out, in, spatialSigma, chromaticSigma, numInterations);
    else
        getQueue().enqueue(kernel::meanShift<T, false>, out, in, spatialSigma, chromaticSigma, numInterations);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T>  meanshift<T>(const Array<T>&, const float&, const float&, const unsigned&, const bool&);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )
INSTANTIATE(short )
INSTANTIATE(ushort)
INSTANTIATE(intl  )
INSTANTIATE(uintl )
}

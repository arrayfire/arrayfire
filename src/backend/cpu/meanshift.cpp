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

template<typename T, bool is_color>
Array<T>  meanshift(const Array<T> &in, const float &s_sigma, const float &c_sigma, const unsigned iter)
{
    in.eval();

    Array<T> out = createEmptyArray<T>(in.dims());

    getQueue().enqueue(kernel::meanShift<T, is_color>, out, in, s_sigma, c_sigma, iter);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T>  meanshift<T, true >(const Array<T> &in, const float &s_sigma, const float &c_sigma, const unsigned iter); \
    template Array<T>  meanshift<T, false>(const Array<T> &in, const float &s_sigma, const float &c_sigma, const unsigned iter);

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

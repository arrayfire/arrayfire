/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "morph_impl.hpp"

namespace opencl
{

#define INSTANTIATE(T, ISDILATE)                                        \
  template Array<T> morph  <T, ISDILATE>(const Array<T> &in, const Array<T> &mask);

    INSTANTIATE(float , false)
    INSTANTIATE(double, false)
    INSTANTIATE(char  , false)
    INSTANTIATE(int   , false)
    INSTANTIATE(uint  , false)
    INSTANTIATE(uchar , false)
    INSTANTIATE(short , false)
    INSTANTIATE(ushort, false)

#undef INSTANTIATE
}

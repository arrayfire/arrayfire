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
#include <regions.hpp>
#include <kernel/regions.hpp>
#include <err_opencl.hpp>

using af::dim4;

namespace opencl
{

template<typename T>
Array<T> regions(const Array<char> &in, af_connectivity connectivity)
{
    const af::dim4 dims = in.dims();

    Array<T> out  = createEmptyArray<T>(dims);

    switch(connectivity) {
        case AF_CONNECTIVITY_4:
            kernel::regions<T, false, 2>(out, in);
            break;
        case AF_CONNECTIVITY_8:
            kernel::regions<T, true,  2>(out, in);
            break;
    }

    return out;
}

#define INSTANTIATE(T)                                                                  \
    template Array<T> regions<T>(const Array<char> &in, af_connectivity connectivity);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(short)
INSTANTIATE(ushort)

}

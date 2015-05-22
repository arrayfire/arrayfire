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
#include <transpose.hpp>
#include <kernel/transpose.hpp>

using af::dim4;

namespace opencl
{

template<typename T>
Array<T> transpose(const Array<T> &in, const bool conjugate)
{
    const dim4 inDims   = in.dims();
    dim4 outDims  = dim4(inDims[1],inDims[0],inDims[2],inDims[3]);
    Array<T> out  = createEmptyArray<T>(outDims);

    if(conjugate) {
        if(inDims[0] % kernel::TILE_DIM == 0 && inDims[1] % kernel::TILE_DIM == 0)
            kernel::transpose<T, true, true>(out, in);
        else
            kernel::transpose<T, true, false>(out, in);
    } else {
        if(inDims[0] % kernel::TILE_DIM == 0 && inDims[1] % kernel::TILE_DIM == 0)
            kernel::transpose<T, false, true>(out, in);
        else
            kernel::transpose<T, false, false>(out, in);
    }

    return out;
}

#define INSTANTIATE(T)                                                          \
    template Array<T> transpose(const Array<T> &in, const bool conjugate);

INSTANTIATE(float  )
INSTANTIATE(cfloat )
INSTANTIATE(double )
INSTANTIATE(cdouble)
INSTANTIATE(char   )
INSTANTIATE(int    )
INSTANTIATE(uint   )
INSTANTIATE(uchar  )
INSTANTIATE(intl   )
INSTANTIATE(uintl  )

}

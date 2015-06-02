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
#include <kernel/transpose_inplace.hpp>

using af::dim4;

namespace opencl
{

template<typename T>
void transpose_inplace(Array<T> &in, const bool conjugate)
{
    dim4 iDims = in.dims();

    if(conjugate) {
        if(iDims[0] % kernel::TILE_DIM == 0 && iDims[1] % kernel::TILE_DIM == 0)
            kernel::transpose_inplace<T, true, true>(in);
        else
            kernel::transpose_inplace<T, true, false>(in);
    } else {
        if(iDims[0] % kernel::TILE_DIM == 0 && iDims[1] % kernel::TILE_DIM == 0)
            kernel::transpose_inplace<T, false, true>(in);
        else
            kernel::transpose_inplace<T, false, false>(in);
    }
}

#define INSTANTIATE(T)                                                          \
    template void transpose_inplace(Array<T> &in, const bool conjugate);

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

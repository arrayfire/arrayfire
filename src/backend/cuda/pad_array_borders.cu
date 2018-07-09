/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <copy.hpp>
#include <err_cuda.hpp>
#include <kernel/pad_array_borders.hpp>

namespace cuda
{
template<typename T>
Array<T> padArrayBorders(Array<T> const& in,
                         dim4 const& lowerBoundPadding,
                         dim4 const& upperBoundPadding,
                         const af::borderType btype)
{
    const dim4& iDims = in.dims();

    dim4 oDims(lowerBoundPadding[0] + iDims[0] + upperBoundPadding[0],
               lowerBoundPadding[1] + iDims[1] + upperBoundPadding[1],
               lowerBoundPadding[2] + iDims[2] + upperBoundPadding[2],
               lowerBoundPadding[3] + iDims[3] + upperBoundPadding[3]);

    auto ret = createEmptyArray<T>(oDims);

    kernel::padBorders<T>(ret, in, lowerBoundPadding, btype);

    return ret;
}

#define INSTANTIATE_PAD_ARRAY_BORDERS(T)                    \
    template Array<T> padArrayBorders<T>(Array<T> const&,   \
            dim4 const &, dim4 const &, const af::borderType);

INSTANTIATE_PAD_ARRAY_BORDERS(cfloat )
INSTANTIATE_PAD_ARRAY_BORDERS(cdouble)
INSTANTIATE_PAD_ARRAY_BORDERS(float  )
INSTANTIATE_PAD_ARRAY_BORDERS(double )
INSTANTIATE_PAD_ARRAY_BORDERS(int    )
INSTANTIATE_PAD_ARRAY_BORDERS(uint   )
INSTANTIATE_PAD_ARRAY_BORDERS(intl   )
INSTANTIATE_PAD_ARRAY_BORDERS(uintl  )
INSTANTIATE_PAD_ARRAY_BORDERS(uchar  )
INSTANTIATE_PAD_ARRAY_BORDERS(char   )
INSTANTIATE_PAD_ARRAY_BORDERS(ushort )
INSTANTIATE_PAD_ARRAY_BORDERS(short  )
}

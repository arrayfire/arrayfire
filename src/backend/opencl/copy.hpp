/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <Array.hpp>
#include <kernel/pad_array_borders.hpp>

namespace opencl
{
    template<typename T>
    void copyData(T *data, const Array<T> &A);

    template<typename T>
    Array<T> copyArray(const Array<T> &A);

    template<typename inType, typename outType>
    void copyArray(Array<outType> &out, const Array<inType> &in);

    template<typename inType, typename outType>
    Array<outType> padArray(Array<inType> const &in, dim4 const &dims,
                            outType default_value, double factor=1.0);

    template<typename T>
    Array<T> padArrayBorders(Array<T> const& in,
                             dim4 const& lowerBoundPadding,
                             dim4 const& upperBoundPadding,
                             const af::borderType btype)
    {
        auto iDims = in.dims();

        dim4 oDims(lowerBoundPadding[0] + iDims[0] + upperBoundPadding[0],
                lowerBoundPadding[1] + iDims[1] + upperBoundPadding[1],
                lowerBoundPadding[2] + iDims[2] + upperBoundPadding[2],
                lowerBoundPadding[3] + iDims[3] + upperBoundPadding[3]);

        auto ret = createEmptyArray<T>(oDims);

        switch(btype)
        {
            case AF_PAD_SYM:
                kernel::padBorders<T, AF_PAD_SYM>(ret, in, lowerBoundPadding);
                break;
            case AF_PAD_CLAMP_TO_EDGE:
                kernel::padBorders<T, AF_PAD_CLAMP_TO_EDGE>(ret, in,
                        lowerBoundPadding);
                break;
            default:
                kernel::padBorders<T, AF_PAD_ZERO>(ret, in, lowerBoundPadding);
                break;
        }

        return ret;
    }

    template<typename T>
    void multiply_inplace(Array<T> &in, double val);

    template<typename T>
    T getScalar(const Array<T> &in);
}

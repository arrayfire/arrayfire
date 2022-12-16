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
#include <math.hpp>
#include <queue.hpp>

namespace af {
class dim4;
}

namespace arrayfire {
namespace cpu {

template<typename T>
void copyData(T *to, const Array<T> &from);

template<typename T>
Array<T> copyArray(const Array<T> &A);

template<typename inType, typename outType>
void copyArray(Array<outType> &out, const Array<inType> &in);

// Resize Array to target dimensions and convert type
//
// Depending on the \p outDims, the output Array can be either truncated
// or padded (towards end of respective dimensions).
//
// While resizing copying, if output dimensions are larger than input, then
// elements beyond the input dimensions are set to the \p defaultValue.
//
// \param[in] in is input Array
// \param[in] outDims is the target output dimensions
// \param[in] defaultValue is the value to which padded locations are set.
// \param[in] scale is the value by which all output elements are scaled.
//
// \returns Array<outType>
template<typename inType, typename outType>
Array<outType> reshape(const Array<inType> &in, const dim4 &outDims,
                       outType defaultValue = outType(0), double scale = 1.0);

template<typename T>
Array<T> padArrayBorders(const Array<T> &in, const dim4 &lowerBoundPadding,
                         const dim4 &upperBoundPadding,
                         const af::borderType btype) {
    const dim4 &iDims = in.dims();

    dim4 oDims(lowerBoundPadding[0] + iDims[0] + upperBoundPadding[0],
               lowerBoundPadding[1] + iDims[1] + upperBoundPadding[1],
               lowerBoundPadding[2] + iDims[2] + upperBoundPadding[2],
               lowerBoundPadding[3] + iDims[3] + upperBoundPadding[3]);

    if (oDims == iDims) { return in; }

    auto ret = (btype == AF_PAD_ZERO ? createValueArray<T>(oDims, scalar<T>(0))
                                     : createEmptyArray<T>(oDims));

    getQueue().enqueue(kernel::padBorders<T>, ret, in, lowerBoundPadding,
                       upperBoundPadding, btype);
    return ret;
}

template<typename T>
void multiply_inplace(Array<T> &in, double val);

template<typename T>
T getScalar(const Array<T> &in);
}  // namespace cpu
}  // namespace arrayfire

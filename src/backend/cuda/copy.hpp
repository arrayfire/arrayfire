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

namespace cuda {
// Copies(blocking) data from an Array<T> object to a contiguous host side
// pointer.
//
// \param dst The destination pointer on the host system.
// \param src    The source array
template <typename T>
void copyData(T *dst, const Array<T> &src);

// Create a deep copy of the \p src Array with the same size and shape. The new
// Array will not maintain the subarray metadata of the \p src array.
//
// \param   src  The source Array<T> object.
// \returns      A new Array<T> object with the same shape and data as the
//               \p src Array<T>
template <typename T>
Array<T> copyArray(const Array<T> &src);

template <typename inType, typename outType>
void copyArray(Array<outType> &out, const Array<inType> &in);

template <typename inType, typename outType>
Array<outType> padArray(Array<inType> const &in, dim4 const &dims,
                        outType default_value, double factor = 1.0);

template <typename T>
Array<T> padArrayBorders(Array<T> const &in, dim4 const &lowerBoundPadding,
                         dim4 const &upperBoundPadding,
                         const af::borderType btype);

template <typename T>
void multiply_inplace(Array<T> &in, double val);

template <typename T>
T getScalar(const Array<T> &in);
}  // namespace cuda

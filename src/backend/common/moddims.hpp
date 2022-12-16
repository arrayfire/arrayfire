/*******************************************************
 * Copyright (c) 2021, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <af/dim4.hpp>

namespace arrayfire {
namespace common {

/// Modifies the shape of the Array<T> object to \p newDims
///
/// Modifies the shape of the Array<T> object to \p newDims. Depending on the
/// in Array, different operations will be performed.
///
/// * If the object is a linear array and it is an unevaluated JIT node, this
///   function will createa a JIT Node.
/// * If the object is not a JIT node but it is still linear, It will create a
///   reference to the internal array with the new shape.
/// * If the array is non-linear a moddims operation will be performed
///
/// \param in       The input array that who's shape will be modified
/// \param newDims  The new shape of the input Array<T>
///
/// \returns        a new Array<T> with the specified shape.
template<typename T>
detail::Array<T> modDims(const detail::Array<T> &in, const af::dim4 &newDims);

/// Calls moddims where all elements are in the first dimension of the array
///
/// \param in  The input Array to be flattened
///
/// \returns A new array where all elements are in the first dimension.
template<typename T>
detail::Array<T> flat(const detail::Array<T> &in);

}  // namespace common
}  // namespace arrayfire

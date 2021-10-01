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

namespace common {

/// Modifies the shape of the Array<T> object to \p newDims
///
/// Modifies the shape of the Array<T> object to \p newDims. If the object is a
/// linear array and it is an unevaluated JIT node, this Array will create a
/// JIT node. If the object is not a JIT node but it is still linear, It will
/// create a
template<typename T>
detail::Array<T> modDims(const detail::Array<T> &in, const af::dim4 &newDims);

template<typename T>
detail::Array<T> flat(const detail::Array<T> &in);

}  // namespace common

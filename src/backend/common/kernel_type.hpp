/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

namespace arrayfire {
namespace common {

/// \brief Maps a type between its data representation and the type used
///        during compute operations
///
/// This struct defines two types. The data type is used to reference the
/// data of an array. The compute type will be used during the computation.
/// The kernel is responsible for converting from the data type to the
/// computation type.
/// For most types these types will be the same. For fp16 type the compute
/// type will be float on platforms that don't support 16 bit floating point
/// operations.
template<typename T>
struct kernel_type {
    /// The type used to represent the data values
    using data = T;

    /// The type used when performing a computation
    using compute = T;

    /// The type defined by the compute framework for this type
    using native = compute;
};
}  // namespace common
}  // namespace arrayfire

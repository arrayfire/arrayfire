/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/SparseArray.hpp>
#include <sparse.hpp>

namespace arrayfire {
namespace opencl {

template<typename T>
Array<T> matmul(const common::SparseArray<T>& lhs, const Array<T>& rhs,
                af_mat_prop optLhs, af_mat_prop optRhs);

}  // namespace opencl
}  // namespace arrayfire

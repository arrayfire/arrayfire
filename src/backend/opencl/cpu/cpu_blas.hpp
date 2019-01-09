/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace opencl {
namespace cpu {
template <typename T>
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
                af_mat_prop optRhs);
}
}  // namespace opencl

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
#include <optypes.hpp>
#include <sparse.hpp>

namespace arrayfire {
namespace cuda {

// These two functions cannot be overloaded by return type.
// So have to give them separate names.
template<typename T, af_op_t op>
Array<T> arithOpD(const common::SparseArray<T> &lhs, const Array<T> &rhs,
                  const bool reverse = false);

template<typename T, af_op_t op>
common::SparseArray<T> arithOp(const common::SparseArray<T> &lhs,
                               const Array<T> &rhs, const bool reverse = false);

template<typename T, af_op_t op>
common::SparseArray<T> arithOp(const common::SparseArray<T> &lhs,
                               const common::SparseArray<T> &rhs);
}  // namespace cuda
}  // namespace arrayfire

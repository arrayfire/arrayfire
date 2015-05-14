/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/blas.h>
#include <Array.hpp>

namespace cuda
{

template<typename T>
Array<T> matmul(const Array<T> &lhs, const Array<T> &rhs,
                af_mat_prop optLhs, af_mat_prop optRhs);

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs,
             af_mat_prop optLhs, af_mat_prop optRhs);

template<typename T>
void trsm(const Array<T> &lhs, Array<T> &rhs, af_mat_prop trans = AF_MAT_NONE,
          bool is_upper = false, bool is_left = true, bool is_unit = false);

}

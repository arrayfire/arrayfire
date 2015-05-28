/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <Array.hpp>

namespace opencl
{
    template<typename T>
    Array<T> solve(const Array<T> &a, const Array<T> &b, const af_mat_prop options = AF_MAT_NONE);

    template<typename T>
    Array<T> solveLU(const Array<T> &a, const Array<int> &pivot,
                     const Array<T> &b, const af_mat_prop options = AF_MAT_NONE);
}

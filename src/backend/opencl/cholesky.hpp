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
    Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper);

    template<typename T>
    int cholesky_inplace(Array<T> &in, const bool is_upper);
}

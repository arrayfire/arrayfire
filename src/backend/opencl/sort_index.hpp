/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace opencl
{
    template<typename T>
    void sort_index(Array<T> &val, Array<unsigned> &idx, const Array<T> &in, const unsigned dim, bool isAscending);
}

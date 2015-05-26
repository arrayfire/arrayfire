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

namespace cuda
{
    template<typename T, bool is_upper, bool is_unit_diag>
    void triangle(Array<T> &out, const Array<T> &in);

    template<typename T, bool is_upper, bool is_unit_diag>
    Array<T> triangle(const Array<T> &in);
}

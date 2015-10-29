/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined (WITH_GRAPHICS)

#include <Array.hpp>
#include <graphics_common.hpp>

namespace cpu
{
    template<typename T>
    void copy_plot3(const Array<T> &P, fg::Plot3* plot3);
}

#endif


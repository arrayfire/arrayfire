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
#include <common/graphics_common.hpp>

namespace opencl
{
    template<typename T>
    void copy_surface(const Array<T> &P, forge::Surface* surface);
}

#endif



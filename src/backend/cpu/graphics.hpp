/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined (WITH_GRAPHICS)

#include <af/graphics.h>
#include <Array.hpp>

namespace cpu
{
    template<typename T>
    void copy_image(const Array<T> &in, const afgfx_image image);
}

#endif

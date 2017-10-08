/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#if defined (WITH_GRAPHICS)

#include <common/graphics_common.hpp>
#include <Array.hpp>

namespace cpu
{

template<typename T>
void copy_histogram(const Array<T> &data, const forge::Histogram* hist);

}

#endif


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

#include <graphics_common.hpp>
#include <Array.hpp>

namespace cuda
{

template<typename T>
void copy_histogram(const Array<T> &data, const fg::Histogram* hist);

}

#endif


/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <ostream>
#include <types.hpp>

namespace opencl
{
namespace kernel
{

    std::ostream&
    operator<<(std::ostream &out, const cfloat& var);

    std::ostream&
    operator<<(std::ostream &out, const cdouble& var);

    static const uint THREADS_PER_GROUP = 256;
    static const uint THREADS_X = 32;
    static const uint THREADS_Y = THREADS_PER_GROUP / THREADS_X;
    static const uint REPEAT    = 32;
}
}

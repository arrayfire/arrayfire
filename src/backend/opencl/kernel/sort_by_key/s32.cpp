/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/sort_by_key_impl.hpp>

namespace opencl
{
namespace kernel
{
    INSTANTIATE1(int,true)
    INSTANTIATE1(int,false)
}
}

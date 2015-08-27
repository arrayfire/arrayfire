/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/array.h>
#include <Array.hpp>
#include <ops.hpp>

namespace opencl
{
    template<af_op_t op, typename Ti, typename To>
    Array<To> reduce(const Array<Ti> &in, const int dim, bool change_nan=false, double nanval=0);

    template<af_op_t op, typename Ti, typename To>
    To reduce_all(const Array<Ti> &in, bool change_nan=false, double nanval=0);
}

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/array.h>
#include <Array.hpp>
#include <optypes.hpp>
#include <binary.hpp>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename T, af_op_t op>
    Array<uchar> *logicOp(const Array<T> &lhs, const Array<T> &rhs)
    {
        return createBinaryNode<uchar, T, op>(lhs, rhs);
    }
}

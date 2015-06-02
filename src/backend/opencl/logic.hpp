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
#include <af/dim4.hpp>
#include <Array.hpp>
#include <optypes.hpp>
#include <binary.hpp>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename T, af_op_t op>
    Array<char> logicOp(const Array<T> &lhs, const Array<T> &rhs, const af::dim4 &odims)
    {
        return createBinaryNode<char, T, op>(lhs, rhs, odims);
    }

    template<typename T, af_op_t op>
    Array<T> bitOp(const Array<T> &lhs, const Array<T> &rhs, const af::dim4 &odims)
    {
        return createBinaryNode<T, T, op>(lhs, rhs, odims);
    }
}

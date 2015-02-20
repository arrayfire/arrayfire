/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <Array.hpp>
#include <ops.hpp>

namespace cuda
{
    template<af_op_t op, typename T>
    void ireduce(Array<T> &out, Array<uint> &loc,
                 const Array<T> &in, const int dim);

    template<af_op_t op, typename T>
    T ireduce_all(unsigned *loc, const Array<T> &in);
}

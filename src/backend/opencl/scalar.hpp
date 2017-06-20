/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <optypes.hpp>
#include <math.hpp>
#include <JIT/ScalarNode.hpp>

namespace opencl
{

template<typename T>
Array<T> createScalarNode(const dim4 &size, const T val)
{
    return createNodeArray<T>(size, JIT::Node_ptr(new JIT::ScalarNode<T>(val)));
}

}

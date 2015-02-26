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

namespace cuda
{

template<typename T>
Array<T> createScalarNode(const dim4 &size, const T val)
{
    JIT::ScalarNode<T> *node = new JIT::ScalarNode<T>(val);
    return createNodeArray<T>(size, JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
}

}

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/jit/ScalarNode.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <memory>

namespace arrayfire {
namespace cuda {

template<typename T>
Array<T> createScalarNode(const dim4 &size, const T val) {
#if _MSC_VER > 1914
    // FIXME(pradeep) - Needed only in CUDA backend, didn't notice any
    // issues in other backends.
    // Either this gaurd or we need to enable extended alignment
    // by defining _ENABLE_EXTENDED_ALIGNED_STORAGE before <type_traits>
    // header is included
    using ScalarNode    = common::ScalarNode<T>;
    using ScalarNodePtr = std::shared_ptr<ScalarNode>;
    return createNodeArray<T>(size, ScalarNodePtr(new ScalarNode(val)));
#else
    return createNodeArray<T>(size,
                              std::make_shared<common::ScalarNode<T>>(val));
#endif
}

}  // namespace cuda
}  // namespace arrayfire

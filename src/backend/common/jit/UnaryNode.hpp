/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/jit/NaryNode.hpp>

namespace arrayfire {
namespace common {

class UnaryNode : public NaryNode {
   public:
    UnaryNode(const af::dtype type, const char *op_str, Node_ptr child,
              af_op_t op)
        : NaryNode(type, op_str, 1, {{child}}, op, child->getHeight() + 1) {
        static_assert(std::is_nothrow_move_assignable<UnaryNode>::value,
                      "UnaryNode is not move assignable");
        static_assert(std::is_nothrow_move_constructible<UnaryNode>::value,
                      "UnaryNode is not move constructible");
    }
};
}  // namespace common
}  // namespace arrayfire

/*******************************************************
 * Copyright (c) 2021, ArrayFire
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

class ModdimNode : public NaryNode {
   public:
    af::dim4 m_new_shape;
    ModdimNode(const af::dim4& new_shape, const af::dtype type, Node_ptr child)
        : NaryNode(type, "__noop", 1, {{child}}, af_moddims_t,
                   child->getHeight() + 1)
        , m_new_shape(new_shape) {
        static_assert(std::is_nothrow_move_assignable<ModdimNode>::value,
                      "ModdimNode is not move assignable");
        static_assert(std::is_nothrow_move_constructible<ModdimNode>::value,
                      "ModdimNode is not move constructible");
    }

    virtual std::unique_ptr<Node> clone() noexcept final {
        return std::make_unique<ModdimNode>(*this);
    }
};
}  // namespace common
}  // namespace arrayfire

/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include "NaryNode.hpp"
#include <iomanip>

namespace opencl
{

namespace JIT
{
    class UnaryNode : public NaryNode
    {
    public:
        UnaryNode(const char *out_type_str, const char *name_str,
                  const char *op_str,
                  Node_ptr child, int op)
            : NaryNode(out_type_str, name_str, op_str,
                       1, {{child}}, op, child->getHeight() + 1)
        {
        }
    };
}

}

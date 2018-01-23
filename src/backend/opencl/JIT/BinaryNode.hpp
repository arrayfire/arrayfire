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

    class BinaryNode : public NaryNode
    {
    public:
        BinaryNode(const char *out_type_str, const char *name_str,
                   const char *op_str,
                   Node_ptr lhs, Node_ptr rhs, int op)
            : NaryNode(out_type_str, name_str, op_str, 2, {{lhs, rhs}},
                       op, std::max(lhs->getHeight(), rhs->getHeight()) + 1)
        {
        }
    };

}

}
